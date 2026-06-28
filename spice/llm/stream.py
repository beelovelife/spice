"""Model streaming dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator

from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.provider_registry import get_provider
from spice.llm.types import StreamError, StreamEvent, ModelRequestOptions, ToolSchema


async def stream_model(
    model: Model,
    messages: list[Message],
    tools: list[ToolSchema],
    options: ModelRequestOptions,
) -> AsyncIterator[StreamEvent]:
    provider = get_provider(model.provider, protocol=model.protocol, model=model)
    if not provider:
        yield StreamError(f"Unsupported provider: {model.provider}")
        return
    async for event in provider.astream(model, messages, tools, options):
        yield event
