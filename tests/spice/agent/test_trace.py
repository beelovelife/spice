from __future__ import annotations

import json
import stat
from pathlib import Path

import spice.cli.main as cli_main
from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    RoundCompleteEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from spice.agent.trace import RunTraceWriter, attach_trace_writer
from spice.llm.config import SpiceConfig
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.llm.usage import ModelUsageRecord, TokenUsage
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


def test_attach_trace_writer_uses_unique_timestamped_path(monkeypatch, tmp_path: Path) -> None:
    import spice.agent.trace as trace_module

    monkeypatch.setattr(trace_module, "TRACE_DIR", tmp_path)
    session = FakeSession(tmp_path)

    first_writer, first_path = attach_trace_writer(session)
    second_writer, second_path = attach_trace_writer(session)

    assert first_writer.path == first_path
    assert second_writer.path == second_path
    assert first_path != second_path
    assert first_path.parent == tmp_path
    assert first_path.name.endswith("-session-1.trace.json")


def test_trace_writer_can_follow_replacement_session(tmp_path: Path) -> None:
    first = FakeSession(tmp_path, session_id="first")
    second = FakeSession(tmp_path, session_id="second")
    writer, path = attach_trace_writer(first, path=tmp_path / "chat.json")
    first.emit(AgentStartEvent("first"))

    writer.bind(second)
    first.emit(TurnEndEvent("ignored"))
    second.emit(AgentStartEvent("second"))

    data = json.loads(path.read_text(encoding="utf-8"))
    starts = [event["session_id"] for event in data["events"] if event["type"] == "agent_start"]
    assert starts == ["first", "second"]
    assert data["session_id"] == "second"


def test_run_trace_writer_records_runtime_snapshot_without_text_delta_events(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    path = tmp_path / "run.trace.json"
    writer = RunTraceWriter(path, session)

    writer.record(AgentStartEvent(session_id="session-1"))
    writer.record(TextDeltaEvent("hello "))
    writer.record(TextDeltaEvent("world"))
    call = ToolCall(id="tc1", name="read_file", arguments={"path": "README.md"})
    session.messages.append(Message(role="assistant", content="", tool_calls=[call], provider="openai", model="gpt-5.1"))
    usage = ModelUsageRecord(
        provider="openai",
        model="gpt-5.1",
        tokens=TokenUsage(input_tokens=100, output_tokens=10),
        duration_ms=25,
        estimated_cost_usd="0.001",
    )
    writer.record(AssistantMessageEvent(text="", tool_calls=[call], usage=usage))
    writer.record(ModelRetryEvent("openai", "gpt-5.1", 1, 2, 3, 0.5, "temporary"))
    writer.record(ModelFallbackEvent("primary", "openai", "gpt-5.1", "backup", "anthropic", "claude", "server", 0, 1))
    writer.record(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="read_file", args={"path": "README.md"}))
    session.messages.append(Message(role="tool", content="# README", tool_call_id="tc1", name="read_file"))
    writer.record(ToolExecutionEndEvent(tool_call_id="tc1", tool_name="read_file", result=tool_result("# README")))
    writer.record(RoundCompleteEvent(1))
    writer.record(TurnEndEvent(text="done", stop_reason="stop"))
    writer.record(AgentEndEvent(session_id="session-1"))

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["format"] == "spice-trace-1"
    assert data["spice_version"]
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
    assert data["summary"]["usage"]["model_calls"] == 1
    assert data["summary"]["usage"]["input_tokens"] == 100
    assistant_event = next(event for event in data["events"] if event["type"] == "assistant_message")
    assert assistant_event["usage"]["estimated_cost_usd"] == "0.001"
    assert "model_retry" in {event["type"] for event in data["events"]}
    assert "model_fallback" in {event["type"] for event in data["events"]}
    assert "round_complete" in {event["type"] for event in data["events"]}


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
    monkeypatch.setattr(cli_main, "run_prompt", fake_render_prompt)

    cli_main.run("hello", trace_file=trace_path)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["session_id"] == "created-session"
    assert data["messages"][-1] == {"role": "user", "content": "hello"}
    assert data["summary"]["status"] == "completed"


def test_user_interruption_has_distinct_trace_status(tmp_path: Path) -> None:
    writer = RunTraceWriter(tmp_path / "interrupted.json", FakeSession(tmp_path))

    writer.record(AgentErrorEvent("cancelled", kind="user_interrupted"))
    writer.record(AgentEndEvent(session_id="session-1"))

    assert writer.snapshot()["summary"]["status"] == "interrupted"

    writer.record(AgentStartEvent(session_id="session-1"))
    writer.record(TurnEndEvent(text="continued", stop_reason="stop"))
    writer.record(AgentEndEvent(session_id="session-1"))

    assert writer.snapshot()["summary"]["status"] == "completed"


def test_trace_redacts_secrets_drops_full_output_and_uses_private_permissions(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    session.messages.append(
        Message(
            role="tool",
            content="Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            metadata={"_full_tool_output": "private full output", "tool_result": {"api_key": "secret-value"}},
        )
    )
    path = tmp_path / "private.trace.json"
    writer = RunTraceWriter(path, session)
    writer.record(AgentEndEvent(session_id="session-1"))

    raw = path.read_text(encoding="utf-8")
    assert "private full output" not in raw
    assert "secret-value" not in raw
    assert "abcdefghijklmnopqrstuvwxyz" not in raw
    assert "[REDACTED]" in raw
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
