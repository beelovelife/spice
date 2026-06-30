"""Agent-level events consumed by frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spice.llm.messages import ToolCall
from spice.tools.base import ToolResult


class AgentEvent:
    """Base class for frontend-consumable agent events."""


@dataclass
class AgentStartEvent(AgentEvent):
    session_id: str


@dataclass
class AgentEndEvent(AgentEvent):
    session_id: str


@dataclass
class TurnStartEvent(AgentEvent):
    prompt: str


@dataclass
class TextDeltaEvent(AgentEvent):
    text: str


@dataclass
class AssistantMessageEvent(AgentEvent):
    text: str
    tool_calls: list[ToolCall]


@dataclass
class ToolExecutionStartEvent(AgentEvent):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass
class ToolExecutionUpdateEvent(AgentEvent):
    tool_call_id: str
    tool_name: str
    text: str


@dataclass
class ToolExecutionEndEvent(AgentEvent):
    tool_call_id: str
    tool_name: str
    result: ToolResult


@dataclass
class ModelRetryEvent(AgentEvent):
    provider: str
    model: str
    failed_attempt: int
    next_attempt: int
    max_attempts: int
    delay_seconds: float
    error: str


@dataclass
class ModelFallbackEvent(AgentEvent):
    from_profile: str
    from_provider: str
    from_model: str
    to_profile: str
    to_provider: str
    to_model: str
    reason: str
    fallback_index: int
    fallback_count: int


@dataclass
class TurnEndEvent(AgentEvent):
    text: str
    stop_reason: str = "stop"


@dataclass
class AgentErrorEvent(AgentEvent):
    message: str
    kind: str = "error"
