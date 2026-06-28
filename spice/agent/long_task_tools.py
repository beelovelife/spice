"""State-bound tools for active long tasks."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from spice.agent.long_task import LongTaskState
from spice.tools.base import Tool, ToolContext, ToolResult, tool_result

SetStateFn = Callable[[LongTaskState], LongTaskState | None]
CanCompleteFn = Callable[[], tuple[bool, str]]


def create_long_task_tools(
    *,
    get_state: Callable[[], LongTaskState] | None = None,
    set_state: SetStateFn | None = None,
    can_complete: CanCompleteFn | None = None,
) -> list[Tool]:
    async def complete_long_task(args: dict[str, Any], context: ToolContext) -> ToolResult:
        note = str(args.get("note") or "").strip()
        state = get_state() if get_state else LongTaskState()
        if not state.is_active:
            return ToolResult("No sustained goal is active.", is_error=True)
        if can_complete:
            ok, message = can_complete()
            if not ok:
                return ToolResult(message, is_error=True)
        state.complete(note=note)
        if set_state:
            state = set_state(state) or state
        payload = state.to_dict()
        return tool_result(json.dumps(payload, ensure_ascii=False), payload)

    return [
        Tool(
            name="complete_long_task",
            description=(
                "Mark the active sustained goal complete after the work is actually done and verified. "
                "Use this only when no further continuation is needed for the current goal."
            ),
            parameters={
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "additionalProperties": False,
            },
            execute=complete_long_task,
        ),
    ]
