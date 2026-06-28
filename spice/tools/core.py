"""Core tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from spice.tools.base import Tool, ToolContext, ToolResult, tool_result


async def get_current_time(args: dict[str, Any], context: ToolContext) -> ToolResult:
    return tool_result(datetime.now(timezone.utc).isoformat())


def create_core_tools() -> list[Tool]:
    return [
        Tool(
            name="get_current_time",
            description="Return the current UTC time as an ISO-8601 timestamp.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=get_current_time,
        )
    ]
