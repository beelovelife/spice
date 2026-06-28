from __future__ import annotations

import asyncio

from spice.agent.long_task import LongTaskState, LongTaskStore
from spice.agent.long_task_tools import create_long_task_tools
from spice.storage.sqlite_long_task import SqliteLongTaskStore
from spice.tools.base import ToolContext


def test_long_task_state_defaults_to_idle() -> None:
    state = LongTaskState()

    assert state.objective == ""
    assert state.status == "idle"
    assert state.max_continuation_rounds == 12
    assert state.remaining_continuations == 12
    assert state.needs_user_attention is False


def test_create_long_task_tools_registers_tool() -> None:
    tools = create_long_task_tools()

    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"complete_long_task"}


def test_complete_long_task_marks_active_goal_completed(tmp_path) -> None:
    from spice.agent.long_task import LongTaskState

    state = LongTaskState()
    tools = {tool.name: tool for tool in create_long_task_tools(get_state=lambda: state, set_state=lambda new: state.__dict__.update(new.__dict__))}

    state.start("restore spice")
    result = asyncio.run(tools["complete_long_task"].execute({"note": "verified"}, ToolContext(cwd=tmp_path)))

    assert result.is_error is False
    assert result.details["status"] == "completed"
    assert "verified" in result.details["notes"]


def test_long_task_state_tracks_continuation_boundary() -> None:
    state = LongTaskState()

    state.start("restore spice", max_continuation_rounds=2)
    state.record_continuation(stop_reason="max_tool_rounds")
    state.record_continuation(stop_reason="max_tool_rounds")

    assert state.continuation_rounds == 2
    assert state.remaining_continuations == 0
    assert state.can_continue is False
    assert state.needs_user_attention is True
    assert state.last_stop_reason == "max_tool_rounds"
    assert "Do not continue indefinitely" in (state.runtime_context() or "")


def test_long_task_state_round_trips_boundary_fields() -> None:
    loaded = LongTaskState.from_dict(
        {
            "customType": "long_task_state",
            "objective": "restore spice",
            "status": "active",
            "notes": ["started"],
            "continuationRounds": "2",
            "maxContinuationRounds": "5",
            "needsUserAttention": True,
            "lastStopReason": "max_tool_rounds",
            "completionCandidate": True,
        }
    )

    assert loaded.continuation_rounds == 2
    assert loaded.max_continuation_rounds == 5
    assert loaded.needs_user_attention is True
    assert loaded.last_stop_reason == "max_tool_rounds"
    assert loaded.completion_candidate is True
    assert loaded.to_dict()["continuationRounds"] == 2


def test_long_task_store_writes_state_checkpoint_and_events(tmp_path) -> None:
    store = LongTaskStore(tmp_path / "tasks")

    state = store.create(objective="restore spice", session_id="session-1", max_continuation_rounds=3)
    store.write_checkpoint(
        state,
        todos=[{"id": "1", "content": "inspect", "status": "completed"}],
        last_action="turn_end",
    )
    store.append_event(state.task_id, "completed", {"note": "verified"})

    task_dir = tmp_path / "tasks" / state.task_id
    assert (task_dir / "state.json").exists()
    assert (task_dir / "checkpoint.json").exists()
    assert (task_dir / "events.jsonl").exists()
    assert store.load_state(state.task_id).objective == "restore spice"


def test_sqlite_long_task_store_writes_state_checkpoint_and_events(tmp_path) -> None:
    store = SqliteLongTaskStore(tmp_path / "spice.db")

    state = store.create(objective="restore spice", session_id="session-1", max_continuation_rounds=3)
    store.write_checkpoint(
        state,
        todos=[{"id": "1", "content": "inspect", "status": "completed"}],
        last_action="turn_end",
    )
    store.append_event(state.task_id, "completed", {"note": "verified"})

    assert store.load_state(state.task_id).objective == "restore spice"
    assert store.load_checkpoint(state.task_id)["lastAction"] == "turn_end"
    assert [event["type"] for event in store.read_events(state.task_id)] == ["created", "completed"]
