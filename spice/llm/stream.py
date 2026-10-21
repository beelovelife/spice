"""Model streaming dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator
from spice.async_utils import closing_iterator

from spice.llm.image_context import project_image_context
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
    messages = project_image_context(messages)
    if any(message.images for message in messages) and not model.supports_vision:
        yield StreamError(
            f"Model {model.provider}/{model.id} does not support image input. "
            "Select a vision-capable model to continue this conversation.",
            kind="unsupported", provider=model.provider, model=model.id,
        )
        return
    provider = get_provider(model.provider, protocol=model.protocol, model=model)
    if not provider:
        yield StreamError(f"Unsupported provider: {model.provider}")
        return
    async with closing_iterator(provider.astream(model, messages, tools, options)) as events:
        async for event in events:
            yield event
