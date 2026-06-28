from __future__ import annotations

import json

from spice.agent.todo_state import TodoState
from spice.tools.base import ToolContext
from spice.tools.todo import create_update_todo_tool


def test_update_todo_tool_writes_and_reads_state(tmp_path) -> None:
    state = TodoState()

    def set_state(next_state: TodoState) -> None:
        nonlocal state
        state = next_state

    tool = create_update_todo_tool(get_state=lambda: state, set_state=set_state)

    import asyncio

    result = asyncio.run(
        tool.execute(
            {
                "todos": [
                    {"id": "1", "content": "Read files", "status": "in_progress"},
                    {"id": "2", "content": "Run tests", "status": "pending"},
                ]
            },
            ToolContext(cwd=tmp_path),
        )
    )
    payload = json.loads(result.content)

    assert not result.is_error
    assert payload["summary"]["total"] == 2
    assert state.read()[0]["status"] == "in_progress"

    read_result = asyncio.run(tool.execute({}, ToolContext(cwd=tmp_path)))
    read_payload = json.loads(read_result.content)

    assert read_payload["todos"] == state.read()


def test_update_todo_tool_reports_multiple_in_progress_error(tmp_path) -> None:
    state = TodoState()
    tool = create_update_todo_tool(get_state=lambda: state, set_state=lambda next_state: None)

    import asyncio

    result = asyncio.run(
        tool.execute(
            {
                "todos": [
                    {"id": "1", "content": "One", "status": "in_progress"},
                    {"id": "2", "content": "Two", "status": "in_progress"},
                ]
            },
            ToolContext(cwd=tmp_path),
        )
    )

    assert result.is_error
    assert "Only one todo item" in result.content


def test_update_todo_tool_describes_new_task_replace_rule() -> None:
    state = TodoState()
    tool = create_update_todo_tool(get_state=lambda: state, set_state=lambda next_state: None)

    assert "For a new complex task" in tool.description
    assert "merge=false" in tool.description
    assert "Todo ids are stable identifiers" in tool.description
    assert "do not renumber or rename todo ids" in tool.description
    assert "replace the old list directly" in tool.description
    assert "Use merge=true only" in tool.description
    assert "completed and cancelled items" in tool.description
    assert "continue from in_progress and pending" in tool.description
    assert "when the user switches tasks" in tool.parameters["properties"]["merge"]["description"]
