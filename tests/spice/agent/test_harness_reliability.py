"""Fault-injection scenarios for the actual session -> loop -> tool chain."""

from __future__ import annotations

import asyncio

import pytest

import spice.agent.agent_session as session_module
from spice.agent.agent_session import AgentSession
from spice.agent.events import (
    AgentEndEvent, AgentErrorEvent, AssistantMessageEvent,
    ToolExecutionEndEvent, ToolExecutionStartEvent, TurnStartEvent,
)
from spice.agent.sessions import SessionStore, message_from_dict
from spice.agent.turn_journal import TOOL_CHECKPOINT_ENTRY, TurnJournal, recover_interrupted_tools
from spice.agent.trace import RunTraceWriter
from spice.llm.config import SpiceConfig
from spice.llm.messages import Message, ToolCall
from spice.llm.types import Done, TextDelta, ToolCallEvent
from spice.storage.sqlite_sessions import SqliteSessionStore
from spice.tools.base import Tool, tool_result


@pytest.fixture(params=["jsonl", "sqlite"])
def store(tmp_path, request):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    if request.param == "jsonl":
        return SessionStore(state_dir / "sessions", cwd=tmp_path)
    return SqliteSessionStore(state_dir / "spice.db", cwd=tmp_path)


def make_session(tmp_path, store, **kwargs):
    return AgentSession(
        cwd=tmp_path, session_store=store,
        config=SpiceConfig(memory_enabled=False, subagents_enabled=False), **kwargs,
    )


def set_stream(monkeypatch, stream):
    build_route = session_module.build_model_route

    def build(*args, **kwargs):
        route = build_route(*args, **kwargs)
        route.stream_factory = stream
        return route

    monkeypatch.setattr(session_module, "build_model_route", build)


def raw_messages(store, session_id):
    return [message_from_dict(entry.data["message"]) for entry in store.path_entries(session_id)
            if entry.type == "message"]


def test_calls_are_persisted_before_execution_and_results_before_observers(tmp_path, store, monkeypatch):
    session = make_session(tmp_path, store)
    requests = 0

    async def stream(*_args):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield ToolCallEvent("call", "write", {})
            yield Done("tool_calls")
        else:
            yield TextDelta("done")
            yield Done("stop")

    async def execute(_args, _context):
        messages = raw_messages(store, session.session_id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[-1].tool_calls[0].id == "call"
        (tmp_path / "side-effect").write_text("done")
        return tool_result("write completed")

    def observer(event):
        if isinstance(event, ToolExecutionEndEvent):
            checkpoints = [entry for entry in store.path_entries(session.session_id)
                           if entry.data.get("customType") == TOOL_CHECKPOINT_ENTRY]
            assert checkpoints[-1].data["message"]["content"] == "write completed"

    session._edit_tools = [Tool("write", "write", {"type": "object"}, execute)]
    session.subscribe(observer)
    set_stream(monkeypatch, stream)

    async def run():
        return [event async for event in session.prompt("run")]

    events = asyncio.run(run())
    assert not session.listener_errors
    assert isinstance(events[-1], AgentEndEvent)
    assert [message.role for message in raw_messages(store, session.session_id)] == [
        "user", "assistant", "tool", "assistant",
    ]


@pytest.mark.parametrize("mode", ["close", "cancel"])
def test_parallel_interruption_preserves_finished_sibling_and_stops_worker(tmp_path, store, monkeypatch, mode):
    session = make_session(tmp_path, store)
    cancelled = asyncio.Event()
    slow_started = asyncio.Event()

    async def stream(*_args):
        yield ToolCallEvent("slow", "slow", {})
        yield ToolCallEvent("fast", "fast", {})
        yield Done("tool_calls")

    async def slow(*_args):
        slow_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fast(*_args):
        await slow_started.wait()
        return tool_result("completed before interruption")

    session._edit_tools = [Tool(name, name, {"type": "object"}, execute, concurrency="parallel")
                           for name, execute in (("slow", slow), ("fast", fast))]
    set_stream(monkeypatch, stream)

    async def run():
        events = session.prompt("run")
        async for event in events:
            if isinstance(event, ToolExecutionEndEvent):
                assert event.tool_call_id == "fast"
                break
        if mode == "close":
            await events.aclose()
        else:
            async def drain():
                return [event async for event in events]

            consumer = asyncio.create_task(drain())
            await asyncio.sleep(0)
            consumer.cancel()
            remaining = await consumer
            assert any(isinstance(event, AgentErrorEvent) and event.kind == "user_interrupted"
                       for event in remaining)
        assert cancelled.is_set()

    asyncio.run(run())
    results = {message.tool_call_id: message for message in raw_messages(store, session.session_id)
               if message.role == "tool"}
    assert set(results) == {"slow", "fast"}
    assert results["fast"].content == "completed before interruption"
    assert not results["fast"].is_error
    assert results["slow"].is_error
    assert "not retried automatically" in results["slow"].content
    resumed = make_session(tmp_path, store, session_id=session.session_id)
    assert len([message for message in resumed.messages if message.role == "tool"]) == 2


@pytest.mark.parametrize("stop_at", [TurnStartEvent, AssistantMessageEvent, ToolExecutionStartEvent, ToolExecutionEndEvent])
def test_closing_at_each_execution_boundary_leaves_a_reusable_session(tmp_path, store, monkeypatch, stop_at):
    session = make_session(tmp_path, store)
    executions = []

    async def stream(*_args):
        yield ToolCallEvent("call", "write", {})
        yield Done("tool_calls")

    async def execute(*_args):
        executions.append("ran")
        return tool_result("actual result")

    session._edit_tools = [Tool("write", "write", {"type": "object"}, execute)]
    set_stream(monkeypatch, stream)

    async def run():
        events = session.prompt("run")
        async for event in events:
            if isinstance(event, stop_at):
                break
        await events.aclose()
        # The writer lock must be released even on GeneratorExit.
        next_prompt = session.prompt("next")
        await anext(next_prompt)
        await next_prompt.aclose()

    asyncio.run(run())
    messages = raw_messages(store, session.session_id)
    if stop_at is TurnStartEvent:
        assert [message.role for message in messages] == ["user"]
    else:
        assert [message.role for message in messages] == ["user", "assistant", "tool"]
        assert messages[-1].is_error == (stop_at is not ToolExecutionEndEvent)
    assert executions == (["ran"] if stop_at is ToolExecutionEndEvent else [])


def test_concurrent_prompts_are_rejected_before_mutating_history(tmp_path, store):
    session = make_session(tmp_path, store)

    async def run():
        first = session.prompt("first")
        await anext(first)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await anext(session.prompt("second"))
        finally:
            await first.aclose()

    asyncio.run(run())
    assert raw_messages(store, session.session_id) == []


def test_closed_consumer_is_traced_as_interrupted(tmp_path, store):
    session = make_session(tmp_path, store)
    trace = RunTraceWriter(tmp_path / "trace.json", session)
    trace.bind(session)

    async def run():
        events = session.prompt("run")
        await anext(events)
        await events.aclose()

    asyncio.run(run())
    assert trace.status == "interrupted"
    assert trace.stop_reason == "consumer_closed"


def test_recovery_uses_checkpoints_without_reexecuting_unknown_calls(tmp_path, store):
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    calls = [ToolCall("unknown", "write", {}), ToolCall("known", "read", {})]
    messages = [Message(role="user", content="run"), Message(role="assistant", tool_calls=calls)]
    journal = TurnJournal(store, info.id, cwd=tmp_path, cursor=0, provider="openai", model="gpt-5.1")
    journal.flush(messages)
    journal.checkpoint(ToolExecutionEndEvent("known", "read", tool_result("durable result")))
    # Simulate process loss before ordered tool messages are flushed.
    resumed = make_session(tmp_path, store, session_id=info.id)
    results = [message for message in resumed.messages if message.role == "tool"]
    assert [message.tool_call_id for message in results] == ["unknown", "known"]
    assert results[0].is_error and "Side effects may have occurred" in results[0].content
    assert results[1].content == "durable result" and not results[1].is_error
    assert recover_interrupted_tools(store, info.id) == 0


def test_checkpoint_results_do_not_cross_rounds_or_branches(tmp_path, store):
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    call = ToolCall("repeated", "write", {})
    messages = [Message(role="assistant", tool_calls=[call])]
    journal = TurnJournal(store, info.id, cwd=tmp_path, cursor=0, provider="openai", model="gpt-5.1")
    first_id = journal.flush(messages)
    journal.checkpoint(ToolExecutionEndEvent(call.id, call.name, tool_result("old result")))
    assert recover_interrupted_tools(store, info.id) == 1
    store.append_message(info.id, Message(role="assistant", tool_calls=[call]))
    assert recover_interrupted_tools(store, info.id) == 1
    assert raw_messages(store, info.id)[-1].is_error
    store.set_leaf(info.id, first_id)
    assert recover_interrupted_tools(store, info.id) == 1
    assert raw_messages(store, info.id)[-1].is_error


def test_partial_storage_failure_does_not_duplicate_successful_prefix(tmp_path, store, monkeypatch):
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    journal = TurnJournal(store, info.id, cwd=tmp_path, cursor=0, provider="openai", model="gpt-5.1")
    messages = [Message(role="user", content="question"), Message(role="assistant", content="answer")]
    append = store.append_message
    writes = 0

    def append_with_failure(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        return append(*args, **kwargs)

    monkeypatch.setattr(store, "append_message", append_with_failure)
    with pytest.raises(OSError, match="disk full"):
        journal.flush(messages)
    journal.flush(messages)
    assert [message.content for message in raw_messages(store, info.id)] == ["question", "answer"]


def test_persistence_failure_prevents_tool_side_effects(tmp_path, store, monkeypatch):
    session = make_session(tmp_path, store)
    executions = []

    async def stream(*_args):
        yield ToolCallEvent("call", "write", {})
        yield Done("tool_calls")

    async def execute(*_args):
        executions.append("ran")
        return tool_result("done")

    append = store.append_message

    def fail_assistant(session_id, message, **kwargs):
        if message.role == "assistant":
            raise OSError("disk full")
        return append(session_id, message, **kwargs)

    session._edit_tools = [Tool("write", "write", {"type": "object"}, execute)]
    set_stream(monkeypatch, stream)
    monkeypatch.setattr(store, "append_message", fail_assistant)

    async def run():
        return [event async for event in session.prompt("run")]

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(run())
    assert executions == []
    assert [message.role for message in raw_messages(store, session.session_id)] == ["user"]
    with pytest.raises(RuntimeError, match="resume the session"):
        asyncio.run(run())


def test_jsonl_recovers_after_truncated_final_row(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    store.append_message(info.id, Message(role="assistant", tool_calls=[ToolCall("call", "write", {})]))
    with store.path_for(info.id).open("ab") as handle:
        handle.write(b'{"type":"message", "id":')
    assert recover_interrupted_tools(store, info.id) == 1
    assert recover_interrupted_tools(store, info.id) == 0
    assert raw_messages(store, info.id)[-1].is_error
