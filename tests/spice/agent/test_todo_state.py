from __future__ import annotations

import pytest

from spice.agent.todo_state import MAX_TODO_CONTENT_CHARS, TodoState, todos_from_plan_steps


def test_todo_state_replace_and_summary() -> None:
    state = TodoState()

    state.replace(
        [
            {"id": "1", "content": "Read files", "status": "completed"},
            {"id": "2", "content": "Run tests", "status": "in_progress"},
        ]
    )

    assert state.read() == [
        {"id": "1", "content": "Read files", "status": "completed"},
        {"id": "2", "content": "Run tests", "status": "in_progress"},
    ]
    assert state.summary() == {
        "total": 2,
        "pending": 0,
        "in_progress": 1,
        "completed": 1,
        "cancelled": 0,
    }


def test_todo_state_merge_updates_existing_items() -> None:
    state = TodoState()
    state.replace(
        [
            {"id": "1", "content": "Read files", "status": "pending"},
            {"id": "2", "content": "Run tests", "status": "pending"},
        ]
    )

    state.merge(
        [
            {"id": "1", "content": "Read files", "status": "completed"},
            {"id": "3", "content": "Summarize", "status": "pending"},
        ]
    )

    assert state.read() == [
        {"id": "1", "content": "Read files", "status": "completed"},
        {"id": "2", "content": "Run tests", "status": "pending"},
        {"id": "3", "content": "Summarize", "status": "pending"},
    ]


def test_todo_state_rejects_multiple_in_progress_items() -> None:
    state = TodoState()

    with pytest.raises(ValueError, match="Only one todo item"):
        state.replace(
            [
                {"id": "1", "content": "One", "status": "in_progress"},
                {"id": "2", "content": "Two", "status": "in_progress"},
            ]
        )


def test_todo_state_normalizes_invalid_status_and_truncates_content() -> None:
    state = TodoState()

    state.replace([{"id": "1", "content": "x" * (MAX_TODO_CONTENT_CHARS + 20), "status": "bad"}])

    item = state.read()[0]
    assert item["status"] == "pending"
    assert len(item["content"]) == MAX_TODO_CONTENT_CHARS
    assert item["content"].endswith("... [truncated]")


def test_todo_runtime_context_includes_full_list_with_history_instruction() -> None:
    state = TodoState()
    state.replace(
        [
            {"id": "1", "content": "Done", "status": "completed"},
            {"id": "2", "content": "Current", "status": "in_progress"},
            {"id": "3", "content": "Next", "status": "pending"},
            {"id": "4", "content": "Old path", "status": "cancelled"},
        ]
    )

    context = state.runtime_context()

    assert context is not None
    assert "Current todo list:" in context
    assert "[x] 1. Done (completed)" in context
    assert "Current" in context
    assert "Next" in context
    assert "[~] 4. Old path (cancelled)" in context
    assert "history only" in context


def test_todos_from_plan_steps_creates_pending_items() -> None:
    state = todos_from_plan_steps(["Read files", "Run tests"])

    assert state.read() == [
        {"id": "1", "content": "Read files", "status": "pending"},
        {"id": "2", "content": "Run tests", "status": "pending"},
    ]
