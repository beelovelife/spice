"""Todo progress tracking tool."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from spice.agent.todo_state import TodoState
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result


def create_update_todo_tool(
    *,
    get_state: Callable[[], TodoState],
    set_state: Callable[[TodoState], None],
) -> Tool:
    async def update_todo(args: dict[str, Any], context: ToolContext) -> ToolResult:
        state = get_state()
        todos = args.get("todos")
        merge = bool(args.get("merge", False))
        try:
            if todos is not None:
                if not isinstance(todos, list):
                    return tool_error("todos must be an array of todo items.")
                if merge:
                    state.merge(todos)
                else:
                    state.replace(todos)
                set_state(state)
        except ValueError as exc:
            return tool_error(str(exc))

        payload = {
            "todos": state.read(),
            "summary": state.summary(),
        }
        return tool_result(json.dumps(payload, ensure_ascii=False), details=payload)

    return Tool(
        name="update_todo",
        description=(
            "Manage the task list for the current session. Use for complex tasks with 3+ meaningful "
            "steps, multi-file changes, or approved plans. Call with no parameters to read the current "
            "list. Call with todos to replace or update the list. Each item is {id, content, status}; "
            "status is pending, in_progress, completed, or cancelled. List order is priority. Keep "
            "exactly one item in_progress while working. Mark items completed immediately after "
            "finishing them. Todo ids are stable identifiers within the current task; when updating "
            "status, reuse the existing id exactly and do not renumber or rename todo ids. "
            "For a new complex task, call this with merge=false and a fresh todo list; "
            "do not merge new-task todos into a completed old task. If the user interrupts or switches "
            "to a different task, replace the old list directly with merge=false and the new task's todos. "
            "Use merge=true only to update statuses or append follow-up steps within the same active task. "
            "Runtime todo context may show completed and cancelled items to preserve step identity; treat "
            "those as history only and continue from in_progress and pending items unless explicitly asked. "
            "Avoid this tool for simple one-shot answers, translations, explanations, small lookups, "
            "or obvious single-step edits. Always returns the full current list."
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "content": {"type": "string", "minLength": 1},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                        },
                        "required": ["id", "content", "status"],
                        "additionalProperties": False,
                    },
                    "maxItems": 256,
                },
                "merge": {
                    "type": "boolean",
                    "description": "When true, update existing items by id and append new items for the same active task. When false, replace the whole list; use false for every new complex task or when the user switches tasks.",
                },
            },
            "additionalProperties": False,
        },
        execute=update_todo,
    )
