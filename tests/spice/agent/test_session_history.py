from __future__ import annotations

import asyncio

import pytest

import spice.agent.agent_session as session_module
from spice.agent.agent_session import AgentSession
from spice.agent.events import AgentErrorEvent, TurnStartEvent
from spice.agent.file_references import FileReferenceError
from spice.agent.session_history import AGENT_ERROR_ENTRY, HistoryError, history_from_entries, session_history
from spice.agent.sessions import SessionStore
from spice.agent.turn_outcome import TURN_OUTCOME_ENTRY
from spice.llm.config import SpiceConfig
from spice.llm.messages import Message
from spice.llm.types import Done, StreamError, TextDelta
from spice.storage.sqlite_sessions import SqliteSessionStore


@pytest.fixture(params=["jsonl", "sqlite"])
def store(tmp_path, request):
    if request.param == "jsonl":
        return SessionStore(tmp_path / "sessions", cwd=tmp_path)
    return SqliteSessionStore(tmp_path / "spice.db", cwd=tmp_path)


def make_session(tmp_path, store, **kwargs):
    return AgentSession(
        cwd=tmp_path, session_store=store,
        config=SpiceConfig(memory_enabled=False, subagents_enabled=False), **kwargs,
    )


@pytest.mark.parametrize("partial", [False, True])
def test_model_error_survives_resume_in_order_without_entering_model_context(tmp_path, store, monkeypatch, partial):
    requests = []
    original = session_module.build_model_route

    async def stream(model, messages, tools, options):
        requests.append(list(messages))
        if len(requests) == 1:
            if partial:
                yield TextDelta("partial answer")
            yield StreamError("length limit exceeded; api_key=private-value")
        else:
            yield TextDelta("continued successfully")
            yield Done("stop")

    def build(*args, **kwargs):
        route = original(*args, **kwargs)
        route.stream_factory = stream
        return route

    monkeypatch.setattr(session_module, "build_model_route", build)
    session = make_session(tmp_path, store)
    observed = []

    def observe(event):
        if isinstance(event, AgentErrorEvent):
            observed.extend(item for item in session_history(session) if isinstance(item, HistoryError))

    session.subscribe(observe)

    async def fail():
        return [event async for event in session.prompt("first question")]

    asyncio.run(fail())
    assert not session.listener_errors and len(observed) == 1
    entries = store.path_entries(session.session_id)
    errors = [entry for entry in entries if entry.data.get("customType") == AGENT_ERROR_ENTRY]
    assert len(errors) == 1
    assert "private-value" not in str(errors[0].data)
    assert errors[0].timestamp
    assert len([entry for entry in entries if entry.data.get("customType") == TURN_OUTCOME_ENTRY]) == 1

    resumed = make_session(tmp_path, store, session_id=session.session_id)

    async def continue_turn():
        return [event async for event in resumed.prompt("continue")]

    asyncio.run(continue_turn())
    history = session_history(resumed)
    assert [item.message if isinstance(item, HistoryError) else item.content for item in history] == [
        "first question", *(["partial answer"] if partial else []),
        "length limit exceeded; api_key=[redacted]", "continue", "continued successfully",
    ]
    assert not any("length limit exceeded" in message.content for message in requests[-1])
    assert not any("length limit exceeded" in message.content for message in store.load_messages(session.session_id))
    user_id = next(entry.id for entry in entries if entry.type == "message")
    resumed.rewind(user_id)
    assert not any(isinstance(item, HistoryError) for item in session_history(resumed))


@pytest.mark.parametrize("failure", ["runtime", "cancelled"])
def test_runtime_and_cancellation_errors_are_persisted(tmp_path, store, monkeypatch, failure):
    async def run_turn(*, prompt, messages, **kwargs):
        messages.append(Message(role="user", content=prompt))
        yield TurnStartEvent(prompt)
        if failure == "runtime":
            raise RuntimeError("unexpected failure; api_key=private-value")
        raise asyncio.CancelledError()

    monkeypatch.setattr(session_module, "run_turn", run_turn)
    session = make_session(tmp_path, store)

    async def run():
        return [event async for event in session.prompt("work")]

    if failure == "runtime":
        with pytest.raises(RuntimeError):
            asyncio.run(run())
    else:
        asyncio.run(run())

    history = session_history(session)
    assert len(history) == 2 and isinstance(history[1], HistoryError)
    assert history[1].kind == ("runtime_error" if failure == "runtime" else "user_interrupted")
    assert "private-value" not in history[1].message
    assert len(store.load_messages(session.session_id)) == 1


def test_old_failed_outcome_has_a_fallback_but_recorded_errors_are_not_duplicated(tmp_path, store):
    info = store.create(cwd=tmp_path, provider="fake", model="fake")
    store.append_message(info.id, Message(role="user", content="old request"))
    store.append_custom(info.id, {"customType": TURN_OUTCOME_ENTRY, "status": "failed", "stop_reason": "error"})
    store.append_message(info.id, Message(role="user", content="new request"))
    store.append_custom(info.id, {"customType": AGENT_ERROR_ENTRY, "message": "specific failure", "kind": "error"})
    store.append_custom(info.id, {"customType": TURN_OUTCOME_ENTRY, "status": "failed", "stop_reason": "error"})
    store.append_message(info.id, Message(role="assistant", content="later success"))
    store.append_compaction(info.id, summary="summary", first_kept_entry_id=store.path_entries(info.id)[-1].id, tokens_before=100)

    history = history_from_entries(store.path_entries(info.id))
    errors = [item for item in history if isinstance(item, HistoryError)]
    assert len(errors) == 2
    assert "未保存具体错误" in errors[0].message
    assert errors[1].message == "specific failure"
    assert isinstance(history[0], Message) and history[0].content == "old request"
    assert isinstance(history[-1], Message) and history[-1].content == "later success"


def test_input_validation_failure_is_recorded_with_its_user_request(tmp_path, store, monkeypatch):
    def invalid_reference(*args, **kwargs):
        raise FileReferenceError("Model does not support image input")

    monkeypatch.setattr(session_module, "expand_file_references", invalid_reference)
    session = make_session(tmp_path, store)

    async def run():
        return [event async for event in session.prompt("inspect @ui.png")]

    events = asyncio.run(run())
    assert len(events) == 1 and isinstance(events[0], AgentErrorEvent)
    history = session_history(session)
    assert len(history) == 2
    assert isinstance(history[0], Message) and history[0].content == "inspect @ui.png"
    assert isinstance(history[1], HistoryError) and history[1].message == events[0].message
