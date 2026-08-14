"""Runtime trace writer for non-interactive runs."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    RoundCompleteEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.llm.config import CONFIG_DIR
from spice.llm.messages import Message, ToolCall
from spice.llm.usage import ModelUsageRecord, aggregate_usage_records
from spice.tools.base import ToolResult
from spice.version import __version__

DEFAULT_TRACE_PATH = CONFIG_DIR / "last_trace.json"
TRACE_DIR = CONFIG_DIR / "traces"
TRACE_FORMAT = "spice-trace-1"
TRACE_FLUSH_INTERVAL_SECONDS = 0.5
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|sk-proj|sk-ant)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)\b((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+"),
)


def timestamped_trace_path(session_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_session = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in session_id
    )
    return TRACE_DIR / f"{timestamp}-{safe_session}.trace.json"


def attach_trace_writer(
    session, *, path: Path | None = None
) -> tuple["RunTraceWriter", Path]:
    resolved = (
        path.expanduser()
        if path is not None
        else timestamped_trace_path(session.session_label)
    )
    writer = RunTraceWriter(resolved, session)
    writer.bind(session)
    return writer, resolved


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
        self.reasoning_delta_events = 0
        self.reasoning_delta_chars = 0
        self.usage_records: list[ModelUsageRecord] = []
        self._unsubscribe = None
        self._last_flush_at = 0.0
        self.flush()

    def bind(self, session) -> None:
        """Follow a replacement session without losing this invocation's events."""
        if self._unsubscribe is not None:
            self._unsubscribe()
        self.session = session
        self._unsubscribe = session.subscribe(self.record)
        self.flush()

    def record(self, event: AgentEvent) -> None:
        should_flush = False
        if isinstance(event, TextDeltaEvent):
            self.text_delta_events += 1
            self.text_delta_chars += len(event.text)
            return
        if isinstance(event, ReasoningDeltaEvent):
            # Reasoning can contain sensitive intermediate material. Keep only
            # aggregate diagnostics and never write its contents to the trace.
            self.reasoning_delta_events += 1
            self.reasoning_delta_chars += len(event.text)
            return

        if isinstance(event, AgentStartEvent):
            self.status = "running"
            self.stop_reason = None
            self.error = None
            self.events.append(
                {
                    "type": "agent_start",
                    "session_id": event.session_id,
                    "timestamp": _now(),
                }
            )
            should_flush = True
        elif isinstance(event, TurnStartEvent):
            self.events.append(
                {"type": "turn_start", "prompt": event.prompt, "timestamp": _now()}
            )
        elif isinstance(event, RoundCompleteEvent):
            self.events.append(
                {
                    "type": "round_complete",
                    "round_index": event.round_index,
                    "timestamp": _now(),
                }
            )
            should_flush = True
        elif isinstance(event, AssistantMessageEvent):
            if event.usage is not None:
                self.usage_records.append(event.usage)
            self.events.append(
                {
                    "type": "assistant_message",
                    "text": event.text,
                    "tool_calls": [
                        _tool_call_to_dict(call) for call in event.tool_calls
                    ],
                    "usage": event.usage.to_dict() if event.usage is not None else None,
                    "timestamp": _now(),
                }
            )
        elif isinstance(event, ModelRetryEvent):
            self.events.append(
                {"type": "model_retry", **asdict(event), "timestamp": _now()}
            )
        elif isinstance(event, ModelFallbackEvent):
            self.events.append(
                {"type": "model_fallback", **asdict(event), "timestamp": _now()}
            )
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
            should_flush = True
        elif isinstance(event, AgentErrorEvent):
            self.status = "interrupted" if event.kind == "user_interrupted" else "error"
            self.error = event.message
            self.events.append(
                {
                    "type": "agent_error",
                    "message": event.message,
                    "kind": event.kind,
                    "timestamp": _now(),
                }
            )
            should_flush = True
        elif isinstance(event, AgentEndEvent):
            if self.status not in {"error", "interrupted"}:
                self.status = "completed"
            self.events.append(
                {
                    "type": "agent_end",
                    "session_id": event.session_id,
                    "timestamp": _now(),
                }
            )
            should_flush = True
        else:
            self.events.append({"type": type(event).__name__, "timestamp": _now()})

        if (
            should_flush
            or time.monotonic() - self._last_flush_at >= TRACE_FLUSH_INTERVAL_SECONDS
        ):
            self.flush()

    def snapshot(self) -> dict[str, Any]:
        usage = aggregate_usage_records(self.usage_records)
        return {
            "format": TRACE_FORMAT,
            "spice_version": __version__,
            "kind": "agent_trajectory",
            "source": "runtime",
            "created_at": self.created_at,
            "updated_at": _now(),
            "cwd": str(self.session.cwd),
            "session_id": self.session.session.id
            if self.session.session is not None
            else None,
            "model": {
                "provider": self.session.model.provider,
                "id": self.session.model.id,
                "display_name": self.session.model.display_name,
            },
            "runtime": {
                "temperature": self.session.config.temperature,
                "max_tokens": self.session.model.output_tokens,
                "base_url": self.session.config.base_url or self.session.model.base_url,
                "protocol": self.session.model.protocol,
                "environment": getattr(
                    getattr(self.session, "environment", None), "name", None
                ),
                "memory_enabled": self.session.config.memory_enabled,
                "subagents_enabled": self.session.subagents_enabled,
                "active_tools": self.session.get_active_tools(),
            },
            "messages": [
                _message_to_dict(message) for message in self.session.messages
            ],
            "events": self.events,
            "summary": {
                "status": self.status,
                "stop_reason": self.stop_reason,
                "error": self.error,
                "tool_calls": self.tool_calls,
                "text_delta_events": self.text_delta_events,
                "text_delta_chars": self.text_delta_chars,
                "reasoning_delta_events": self.reasoning_delta_events,
                "reasoning_delta_chars": self.reasoning_delta_chars,
                "usage": asdict(usage),
            },
        }

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(_json_safe(self.snapshot()), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        if os.name != "nt":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self.path)
        self._last_flush_at = time.monotonic()


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
        metadata = {
            key: value
            for key, value in message.metadata.items()
            if key != "_full_tool_output"
        }
        data["metadata"] = _json_safe(metadata)
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
        return _json_safe(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(str(key)) else _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in _SENSITIVE_KEY_PARTS:
        return True
    return normalized.endswith(
        ("_api_key", "_apikey", "_password", "_secret", "_token")
    )


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
