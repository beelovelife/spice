"""Provider-facing stream event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from spice.llm.usage import ModelUsageRecord, TokenUsage

StreamErrorKind = Literal[
    "network",
    "timeout",
    "rate_limit",
    "server",
    "authentication",
    "invalid_request",
    "unsupported",
    "cancelled",
    "unknown",
]


class StreamEvent:
    """Base class for provider events."""


@dataclass
class TextDelta(StreamEvent):
    text: str


@dataclass
class ReasoningDelta(StreamEvent):
    """Ephemeral reasoning output for display, never provider conversation history."""

    text: str
    kind: Literal["reasoning", "summary"] = "reasoning"


@dataclass
class ToolCallEvent(StreamEvent):
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Done(StreamEvent):
    """Terminal provider signal, which may still describe an incomplete response."""

    finish_reason: str | None = None
    usage: TokenUsage | None = None
    usage_record: ModelUsageRecord | None = None
    incomplete_reason: str | None = None
    output_token_limit: int | None = None

    @property
    def is_complete(self) -> bool:
        return not self.incomplete_reason and self.finish_reason not in {
            "length", "max_tokens", "max_output_tokens", "content_filter",
            "incomplete", "error",
        }


@dataclass
class StreamError(StreamEvent):
    error: str
    kind: StreamErrorKind = "unknown"
    retryable: bool = False
    status_code: int | None = None
    retry_after_seconds: float | None = None
    provider: str | None = None
    model: str | None = None


@dataclass
class ModelRetryNotice(StreamEvent):
    provider: str
    model: str
    failed_attempt: int
    next_attempt: int
    max_attempts: int
    delay_seconds: float
    error: str


@dataclass
class ModelFallbackNotice(StreamEvent):
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
class ToolSchema:
    """Tool description visible to providers; execution stays in the agent layer."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ModelRequestOptions:
    api_key: str | None = None
    temperature: float = 0.5
    max_tokens: int = 4096
    base_url: str | None = None
