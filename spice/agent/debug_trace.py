"""Verbose per-turn trace logging for debugging model/tool flow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from spice.llm.config import CONFIG_DIR, load_config
from spice.llm.messages import ToolCall
from spice.tools.base import ToolResult

DEFAULT_TRACE_PATH = CONFIG_DIR / "spice.debug.log"
_TOOL_RESULT_PREVIEW_CHARS = 4000


def trace_path() -> Path:
    raw = os.environ.get("SPICE_DEBUG_TRACE_PATH")
    return Path(raw).expanduser() if raw else DEFAULT_TRACE_PATH


def trace_enabled() -> bool:
    env_value = os.environ.get("SPICE_DEBUG_TRACE")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return bool(load_config().debug_trace)


def write_trace(text: str) -> None:
    if not trace_enabled():
        return
    path = trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def trace_turn_start(*, session_label: str | None, prompt: str, message_count: int, tool_count: int) -> None:
    write_trace(
        "\n"
        f"======== turn start session={session_label or '<unknown>'} messages={message_count} tools={tool_count} ========\n"
        "[prompt]\n"
        f"{prompt}\n"
        "[/prompt]\n"
    )


def trace_round_start(round_index: int, *, message_count: int) -> None:
    write_trace(f"======== round {round_index} start messages={message_count} ========")


def trace_round_end(
    round_index: int,
    *,
    duration_ms: int,
    assistant_text: str,
    tool_calls: list[ToolCall],
    finish_reason: str | None,
) -> None:
    write_trace(
        "[assistant text]\n"
        f"{assistant_text}\n"
        "[/assistant text]\n"
        "[tool calls]\n"
        f"{_format_tool_calls(tool_calls)}"
        "[/tool calls]\n"
        f"======== round {round_index} end duration_ms={duration_ms} "
        f"text_chars={len(assistant_text)} tool_calls={len(tool_calls)} finish_reason={finish_reason or '<unknown>'} ========\n"
    )


def trace_tool_start(round_index: int, call: ToolCall) -> None:
    write_trace(
        f"---- tool start round={round_index} name={call.name} id={call.id} ----\n"
        "[arguments]\n"
        f"{_json(call.arguments)}\n"
        "[/arguments]\n"
    )


def trace_tool_end(round_index: int, call: ToolCall, result: ToolResult, *, duration_ms: int) -> None:
    content = result.content or ""
    preview = _truncate(content, _TOOL_RESULT_PREVIEW_CHARS)
    write_trace(
        f"[tool result] round={round_index} name={call.name} id={call.id} "
        f"ok={not result.is_error} duration_ms={duration_ms} content_chars={len(content)}\n"
        f"{preview}\n"
        "[/tool result]\n"
        f"---- tool end round={round_index} name={call.name} id={call.id} ----\n"
    )


def trace_turn_end(*, rounds: int, text_chars: int, tool_calls: int) -> None:
    write_trace(f"======== turn end rounds={rounds} text_chars={text_chars} tool_calls={tool_calls} ========\n")


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return "(none)\n"
    lines = []
    for index, call in enumerate(tool_calls, start=1):
        lines.append(f"{index}. {call.name} id={call.id} args={_json(call.arguments)}")
    return "\n".join(lines) + "\n"


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... <truncated {omitted} chars>"
