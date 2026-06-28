"""Provider protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.types import StreamEvent, ModelRequestOptions, ToolSchema


class Provider(Protocol):
    def astream(
        self,
        model: Model,
        messages: list[Message],
        tools: list[ToolSchema],
        options: ModelRequestOptions,
    ) -> AsyncIterator[StreamEvent]:
        """Stream provider events."""
        ...
