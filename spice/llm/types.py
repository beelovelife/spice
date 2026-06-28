"""Provider-facing stream event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class StreamEvent:
    """Base class for provider events."""


@dataclass
class TextDelta(StreamEvent):
    text: str


@dataclass
class ToolCallEvent(StreamEvent):
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Done(StreamEvent):
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class StreamError(StreamEvent):
    error: str


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
