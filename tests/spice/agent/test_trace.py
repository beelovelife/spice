from __future__ import annotations

import json
from pathlib import Path

import spice.cli.main as cli_main
from spice.agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from spice.agent.trace import RunTraceWriter
from spice.llm.config import SpiceConfig
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.tools.base import tool_result


class FakeSession:
    def __init__(self, cwd: Path, *, session_id: str | None = "session-1") -> None:
        self.cwd = cwd
        self.model = Model(id="gpt-5.1", provider="openai")
        self.config = SpiceConfig(provider="openai", model="gpt-5.1", temperature=0.2)
        self.subagents_enabled = True
        self.messages = [Message(role="system", content="system prompt")]
        self.session = type("SessionInfo", (), {"id": session_id})() if session_id else None
        self.session_label = session_id or "new"
        self._listeners = []

    def get_active_tools(self):
        return ["read_file", "bash"]

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def emit(self, event):
        for listener in list(self._listeners):
            listener(event)


def test_run_trace_writer_records_runtime_snapshot_without_text_delta_events(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    path = tmp_path / "run.trace.json"
    writer = RunTraceWriter(path, session)

    writer.record(AgentStartEvent(session_id="session-1"))
    writer.record(TextDeltaEvent("hello "))
    writer.record(TextDeltaEvent("world"))
    call = ToolCall(id="tc1", name="read_file", arguments={"path": "README.md"})
    session.messages.append(Message(role="assistant", content="", tool_calls=[call], provider="openai", model="gpt-5.1"))
    writer.record(AssistantMessageEvent(text="", tool_calls=[call]))
    writer.record(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="read_file", args={"path": "README.md"}))
    session.messages.append(Message(role="tool", content="# README", tool_call_id="tc1", name="read_file"))
    writer.record(ToolExecutionEndEvent(tool_call_id="tc1", tool_name="read_file", result=tool_result("# README")))
    writer.record(TurnEndEvent(text="done", stop_reason="stop"))
    writer.record(AgentEndEvent(session_id="session-1"))

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["format"] == "spice-trace-1"
    assert data["kind"] == "agent_trajectory"
    assert data["source"] == "runtime"
    assert data["session_id"] == "session-1"
    assert data["model"] == {"provider": "openai", "id": "gpt-5.1", "display_name": "gpt-5.1"}
    assert data["runtime"]["active_tools"] == ["read_file", "bash"]
    assert [message["role"] for message in data["messages"]] == ["system", "assistant", "tool"]
    assert "text_delta" not in {event["type"] for event in data["events"]}
    assert data["summary"]["status"] == "completed"
    assert data["summary"]["stop_reason"] == "stop"
    assert data["summary"]["tool_calls"] == 1
    assert data["summary"]["text_delta_events"] == 2
    assert data["summary"]["text_delta_chars"] == len("hello world")


def test_run_command_trace_file_wires_runtime_writer(monkeypatch, tmp_path: Path) -> None:
    trace_path = tmp_path / "cli.trace.json"
    fake_session = FakeSession(tmp_path, session_id=None)

    class FakeAgentSession:
        def __new__(cls, **kwargs):
            return fake_session

    async def fake_render_prompt(session, renderer, prompt):
        session.session = type("SessionInfo", (), {"id": "created-session"})()
        session.messages.append(Message(role="user", content=prompt))
        session.emit(AgentStartEvent(session_id="created-session"))
        session.emit(TurnEndEvent(text="done", stop_reason="stop"))
        session.emit(AgentEndEvent(session_id="created-session"))

    monkeypatch.setattr(cli_main, "AgentSession", FakeAgentSession)
    monkeypatch.setattr(cli_main, "render_prompt", fake_render_prompt)

    cli_main.run("hello", trace_file=trace_path)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["session_id"] == "created-session"
    assert data["messages"][-1] == {"role": "user", "content": "hello"}
    assert data["summary"]["status"] == "completed"
