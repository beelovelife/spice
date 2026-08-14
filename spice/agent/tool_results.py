"""Session metadata helpers for tool results."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from spice.llm.messages import Message
from spice.tools.base import ToolResult, truncate_head_tail

MAX_INLINE_TOOL_OUTPUT = 12_000


def build_tool_result_metadata(tool_name: str, arguments: dict[str, Any], result: ToolResult) -> dict[str, Any]:
    display = _display_text(tool_name, arguments, result)
    return {
        "tool_result": {
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "display": display,
            "is_error": result.is_error,
            "disposition": result.disposition,
            "error_code": result.error_code,
            "details": result.details,
            "truncated": False,
            "original_chars": len(result.content),
        }
    }


def tool_display_text(tool_name: str | Message, arguments: dict[str, Any] | None = None, result: ToolResult | None = None) -> str:
    if isinstance(tool_name, Message):
        message = tool_name
        tool_meta = message.metadata.get("tool_result") if isinstance(message.metadata, dict) else None
        if isinstance(tool_meta, dict):
            display = tool_meta.get("display")
            meta_name = tool_meta.get("tool_name")
            if isinstance(display, str) and display and display != meta_name:
                return display
        return ""
    arguments = arguments or {}
    if result is not None:
        return _display_text(tool_name, arguments, result)
    path = arguments.get("path")
    if path:
        return f"{tool_name}: {path}"
    return tool_name


def prepare_tool_message_for_session(message: Message, *, cwd: Path, session_id: str) -> Message:
    if message.role != "tool":
        return message
    metadata = dict(message.metadata)
    tool_meta = dict(metadata.get("tool_result") or {})
    full_content = metadata.pop("_full_tool_output", None)
    source_content = full_content if isinstance(full_content, str) else message.content
    content = message.content
    if len(source_content) > MAX_INLINE_TOOL_OUTPUT:
        artifact_path = _artifact_path(cwd, session_id, message.tool_call_id or "tool")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(source_content, encoding="utf-8")
        content = (
            truncate_head_tail(source_content, MAX_INLINE_TOOL_OUTPUT)
            + f"\n\n[tool output truncated] Full output saved to {artifact_path}"
        )
        tool_meta.update(
            {
                "truncated": True,
                "original_chars": len(source_content),
                "artifact_path": str(artifact_path),
            }
        )
        metadata["tool_result"] = tool_meta
    return Message(
        role=message.role,
        content=content,
        tool_calls=list(message.tool_calls),
        tool_call_id=message.tool_call_id,
        name=message.name,
        is_error=message.is_error,
        provider=message.provider,
        model=message.model,
        metadata=metadata,
    )


def cleanup_tool_results(cwd: Path, session_id: str) -> None:
    shutil.rmtree(cwd / ".spice" / "tool-results" / session_id, ignore_errors=True)


def _artifact_path(cwd: Path, session_id: str, tool_call_id: str) -> Path:
    safe_call = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tool_call_id)
    return cwd / ".spice" / "tool-results" / session_id / f"{safe_call}.txt"


def _display_text(tool_name: str, arguments: dict[str, Any], result: ToolResult) -> str:
    path = arguments.get("path") or result.details.get("path")
    if path:
        return f"{tool_name}: {path}"
    return tool_name
