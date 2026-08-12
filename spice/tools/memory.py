"""Memory tool."""

from __future__ import annotations

import json
from typing import Any

from spice.llm.config import load_config
from spice.storage.factory import create_memory_store
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result


async def memory(args: dict[str, Any], context: ToolContext) -> ToolResult:
    action = str(args.get("action") or "read")
    target = str(args.get("target") or "global")
    if target == "global":
        target = "memory"
    content = str(args.get("content") or "")
    old = str(args.get("old") or args.get("old_text") or "")
    store = create_memory_store(load_config(), workspace=context.cwd)
    try:
        if action == "read":
            result = store.read(target)
        elif action == "add":
            result = store.add(target, content)
        elif action == "replace":
            result = store.replace(target, old, content)
        elif action == "remove":
            result = store.remove(target, content or old)
        else:
            return tool_error(f"Unknown memory action: {action}")
    except ValueError as exc:
        return tool_error(str(exc))
    if not result.get("success"):
        return tool_error(str(result.get("error") or "Memory operation failed."), result)
    return tool_result(json.dumps(result, ensure_ascii=False), result)


def create_memory_tools() -> list[Tool]:
    return [
        Tool(
            name="memory",
            description=(
                "Read or update persistent long-term memory only when memory is enabled. "
                "Use target=user for stable user preferences, profile, communication style, and workflow habits. "
                "Choose exactly one target for each atomic fact. Use target=global for cross-project environment facts and reusable operational knowledge. "
                "Use target=project for commands, conventions, architecture decisions, and debugging facts specific to the current workspace. "
                "Do not save secrets, API keys, raw logs, stack traces, temporary task progress, guesses, or facts easily rediscovered from files. "
                "Use action=read before replace/remove when you need the exact existing entry; replace/remove require a unique substring in old/old_text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "add", "replace", "remove"]},
                    "target": {"type": "string", "enum": ["user", "global", "project"]},
                    "content": {"type": "string"},
                    "old": {"type": "string"},
                    "old_text": {"type": "string"},
                },
                "required": ["action", "target"],
                "additionalProperties": False,
            },
            execute=memory,
        )
    ]
