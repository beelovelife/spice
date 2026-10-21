from __future__ import annotations

import asyncio
from functools import partial

import pytest

import spice.agent.agent_session as session_module
from spice.agent.agent_session import AgentSession
from spice.agent.events import AgentEndEvent, AgentErrorEvent, TurnEndEvent
from spice.agent.loop import run_turn
from spice.agent.sessions import SessionStore, message_from_dict
from spice.agent.turn_outcome import TURN_OUTCOME_ENTRY, continuation_context, latest_turn_outcome
from spice.llm.config import SpiceConfig
from spice.llm.types import Done, StreamError, TextDelta, ToolCallEvent
from spice.storage.sqlite_sessions import SqliteSessionStore
from spice.tools.base import Tool, tool_result


@pytest.fixture(params=["jsonl", "sqlite"])
def store(tmp_path, request):
    state = tmp_path / "state"
    state.mkdir()
    if request.param == "jsonl":
        return SessionStore(state / "sessions", cwd=tmp_path)
    return SqliteSessionStore(state / "spice.db", cwd=tmp_path)


def make_session(tmp_path, store, **kwargs):
    return AgentSession(
        cwd=tmp_path, session_store=store,
        config=SpiceConfig(memory_enabled=False, subagents_enabled=False), **kwargs,
    )


def install_stream(monkeypatch, stream):
    original = session_module.build_model_route

    def build(*args, **kwargs):
        route = original(*args, **kwargs)
        route.stream_factory = stream
        return route

    monkeypatch.setattr(session_module, "build_model_route", build)
    monkeypatch.setattr(session_module, "run_turn", partial(run_turn, max_tool_rounds=1))


def test_paused_checkpoint_survives_resume_and_completed_turn_supersedes_it(tmp_path, store, monkeypatch):
    session = make_session(tmp_path, store)
    writes, resumed_requests = [], []

    async def stream(model, messages, tools, options):
        if tools:
            yield ToolCallEvent("write", "write", {"path": "scene.js"})
            yield Done("tool_calls")
        else:
            yield TextDelta("已创建 scene.js；剩余工作：验证最终场景。")
            yield Done("stop")

    async def execute(args, _context):
        writes.append(args["path"])
        (tmp_path / args["path"]).write_text("scene")
        return tool_result("Wrote scene.js")

    tool = Tool("write", "write", {"type": "object"}, execute)
    session._edit_tools = [tool]
    session.todo_state.replace([{"id": "verify", "content": "验证最终场景", "status": "in_progress"}])
    session._persist_todo_state()
    install_stream(monkeypatch, stream)

    def observe(event):
        if isinstance(event, TurnEndEvent):
            saved = latest_turn_outcome(store.path_entries(session.session_id))
            assert saved["status"] == event.status
            assert saved["checkpoint"]["pending_tasks"] == ["验证最终场景"]

    session.subscribe(observe)

    async def first_turn():
        return [e async for e in session.prompt("创建场景")]

    events = asyncio.run(first_turn())
    assert isinstance(events[-1], AgentEndEvent)
    assert not session.listener_errors
    assert not any(isinstance(e, AgentErrorEvent) for e in events)
    assert writes == ["scene.js"]
    entries = store.path_entries(session.session_id)
    saved = latest_turn_outcome(entries)
    assert saved["status"] == "paused"
    assert saved["checkpoint"]["verification_status"] == "unverified"
    assert saved["checkpoint"]["summary_source"] == "model"
    assert len([e for e in entries if e.data.get("customType") == TURN_OUTCOME_ENTRY]) == 1
    history = [message_from_dict(e.data["message"]) for e in entries if e.type == "message"]
    assert history[-1].content == saved["text"]
    assert history[-1].tool_calls == []
    user_entry = next(e.id for e in entries if e.type == "message" and e.data["message"]["role"] == "user")

    # A real compaction entry keeps continuation facts on the same branch,
    # even though only the kept messages enter the provider conversation.
    store.append_compaction(session.session_id, summary="已创建场景代码", first_kept_entry_id=user_entry, tokens_before=100)
    resumed = make_session(tmp_path, store, session_id=session.session_id)
    resumed._edit_tools = [tool]

    async def continue_stream(model, messages, tools, options):
        resumed_requests.append(messages)
        assert any("Continuation checkpoint:" in m.content for m in messages if m.role == "system")
        assert any("验证最终场景" in m.content for m in messages if m.role == "system")
        yield TextDelta("完成本次检查说明")
        yield Done("stop")

    # Update the installed factory, not the wrapper a second time.
    installed_build = session_module.build_model_route

    def continue_route(*args, **kwargs):
        route = installed_build(*args, **kwargs)
        route.stream_factory = continue_stream
        return route

    monkeypatch.setattr(session_module, "build_model_route", continue_route)

    async def next_turn():
        return [e async for e in resumed.prompt("继续，先说明已完成内容")]

    resumed_events = asyncio.run(next_turn())
    assert writes == ["scene.js"]  # Completed effects are not replayed on resume.
    assert len(resumed_requests) == 1
    assert next(e for e in resumed_events if isinstance(e, TurnEndEvent)).status == "completed"
    latest = latest_turn_outcome(store.path_entries(session.session_id))
    assert latest["status"] == "completed" and latest["checkpoint"] is None
    assert continuation_context(latest) is None
    resumed.rewind(user_entry)
    assert "Continuation checkpoint:" not in (resumed._runtime_context() or "")


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_regular_turn_outcomes_are_persisted_without_finalization(tmp_path, store, monkeypatch, outcome):
    session = make_session(tmp_path, store)
    requests = []

    async def stream(*_args):
        requests.append(True)
        if outcome == "failed":
            yield StreamError("service unavailable")
        else:
            yield TextDelta("answer")
            yield Done("stop")

    install_stream(monkeypatch, stream)

    async def run():
        return [e async for e in session.prompt("hello")]

    events = asyncio.run(run())
    saved = latest_turn_outcome(store.path_entries(session.session_id))
    assert requests == [True]
    assert saved["status"] == outcome
    assert saved["checkpoint"] is None
    assert next(e for e in events if isinstance(e, TurnEndEvent)).status == outcome


def test_cancelling_finalization_keeps_completed_tools_and_records_interruption(tmp_path, store, monkeypatch):
    session = make_session(tmp_path, store)
    summarizing = asyncio.Event()
    closed = []

    async def stream(model, messages, tools, options):
        if tools:
            yield ToolCallEvent("write", "write", {})
            yield Done("tool_calls")
        else:
            try:
                summarizing.set()
                await asyncio.Event().wait()
            finally:
                closed.append(True)

    async def execute(*_args):
        (tmp_path / "completed.txt").write_text("done")
        return tool_result("Wrote completed.txt")

    session._edit_tools = [Tool("write", "write", {"type": "object"}, execute)]
    install_stream(monkeypatch, stream)

    async def run():
        async def collect():
            return [e async for e in session.prompt("work")]

        task = asyncio.create_task(collect())
        await summarizing.wait()
        task.cancel()
        return await task

    events = asyncio.run(run())
    assert closed == [True]
    assert (tmp_path / "completed.txt").read_text() == "done"
    assert next(e for e in events if isinstance(e, TurnEndEvent)).status == "interrupted"
    saved = latest_turn_outcome(store.path_entries(session.session_id))
    assert saved["status"] == "interrupted"
    assert saved["checkpoint"] is None
    tools = [e.data["message"] for e in store.path_entries(session.session_id)
             if e.type == "message" and e.data["message"]["role"] == "tool"]
    assert len(tools) == 1 and tools[0]["content"] == "Wrote completed.txt"
