"""Runtime trace writer for non-interactive runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.llm.config import CONFIG_DIR
from spice.llm.messages import Message, ToolCall
from spice.tools.base import ToolResult

DEFAULT_TRACE_PATH = CONFIG_DIR / "last_trace.json"
TRACE_FORMAT = "spice-trace-1"


class RunTraceWriter:
    """Collect agent events and atomically write a JSON run trace."""

    def __init__(self, path: Path, session) -> None:
        self.path = path
        self.session = session
        self.created_at = _now()
        self.events: list[dict[str, Any]] = []
        self.status = "running"
        self.stop_reason: str | None = None
        self.error: str | None = None
        self.tool_calls = 0
        self.text_delta_events = 0
        self.text_delta_chars = 0
        self.flush()

    def record(self, event: AgentEvent) -> None:
        should_flush = True
        if isinstance(event, TextDeltaEvent):
            self.text_delta_events += 1
            self.text_delta_chars += len(event.text)
            return

        if isinstance(event, AgentStartEvent):
            self.events.append({"type": "agent_start", "session_id": event.session_id, "timestamp": _now()})
        elif isinstance(event, TurnStartEvent):
            self.events.append({"type": "turn_start", "prompt": event.prompt, "timestamp": _now()})
        elif isinstance(event, AssistantMessageEvent):
            self.events.append(
                {
                    "type": "assistant_message",
                    "text": event.text,
                    "tool_calls": [_tool_call_to_dict(call) for call in event.tool_calls],
                    "timestamp": _now(),
                }
            )
        elif isinstance(event, ModelRetryEvent):
            self.events.append({"type": "model_retry", **asdict(event), "timestamp": _now()})
        elif isinstance(event, ModelFallbackEvent):
            self.events.append({"type": "model_fallback", **asdict(event), "timestamp": _now()})
        elif isinstance(event, ToolExecutionStartEvent):
            self.tool_calls += 1
            self.events.append(
                {
                    "type": "tool_start",
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "args": _json_safe(event.args),
                    "timestamp": _now(),
                }
            )
            should_flush = False
        elif isinstance(event, ToolExecutionEndEvent):
            self.events.append(
                {
                    "type": "tool_end",
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "result": _tool_result_to_dict(event.result),
                    "timestamp": _now(),
                }
            )
        elif isinstance(event, TurnEndEvent):
            self.stop_reason = event.stop_reason
            self.events.append(
                {
                    "type": "turn_end",
                    "text": event.text,
                    "stop_reason": event.stop_reason,
                    "timestamp": _now(),
                }
            )
        elif isinstance(event, AgentErrorEvent):
            self.status = "error"
            self.error = event.message
            self.events.append({"type": "agent_error", "message": event.message, "kind": event.kind, "timestamp": _now()})
        elif isinstance(event, AgentEndEvent):
            if self.status != "error":
                self.status = "completed"
            self.events.append({"type": "agent_end", "session_id": event.session_id, "timestamp": _now()})
        else:
            self.events.append({"type": type(event).__name__, "timestamp": _now()})

        if should_flush:
            self.flush()

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": TRACE_FORMAT,
            "kind": "agent_trajectory",
            "source": "runtime",
            "created_at": self.created_at,
            "updated_at": _now(),
            "cwd": str(self.session.cwd),
            "session_id": self.session.session.id if self.session.session is not None else None,
            "model": {
                "provider": self.session.model.provider,
                "id": self.session.model.id,
                "display_name": self.session.model.display_name,
            },
            "runtime": {
                "temperature": self.session.config.temperature,
                "max_tokens": self.session.model.output_tokens,
                "base_url": self.session.config.base_url or self.session.model.base_url,
                "memory_enabled": self.session.config.memory_enabled,
                "subagents_enabled": self.session.subagents_enabled,
                "active_tools": self.session.get_active_tools(),
            },
            "messages": [_message_to_dict(message) for message in self.session.messages],
            "events": self.events,
            "summary": {
                "status": self.status,
                "stop_reason": self.stop_reason,
                "error": self.error,
                "tool_calls": self.tool_calls,
                "text_delta_events": self.text_delta_events,
                "text_delta_chars": self.text_delta_chars,
            },
        }

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        tmp_path.write_text(json.dumps(_json_safe(self.snapshot()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.path)


def _message_to_dict(message: Message) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        data["tool_calls"] = [_tool_call_to_dict(call) for call in message.tool_calls]
    for attr in ("tool_call_id", "name", "provider", "model"):
        value = getattr(message, attr)
        if value is not None:
            data[attr] = value
    if message.is_error:
        data["is_error"] = True
    if message.metadata:
        data["metadata"] = _json_safe(message.metadata)
    return data


def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": _json_safe(call.arguments)}


def _tool_result_to_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "content": result.content,
        "is_error": result.is_error,
        "details": _json_safe(result.details),
        "disposition": result.disposition,
        "error_code": result.error_code,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
