"""Anthropic provider."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast

from spice.agent.logging_config import get_logger
from spice.llm.error_safety import stream_error_from_exception
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.types import Done, StreamError, StreamEvent, ModelRequestOptions, TextDelta, ToolCallEvent, ToolSchema

logger = get_logger(__name__)


class AnthropicProvider:
    def __init__(self, *, default_base_url: str | None = None) -> None:
        self.default_base_url = default_base_url

    async def astream(
        self,
        model: Model,
        messages: list[Message],
        tools: list[ToolSchema],
        options: ModelRequestOptions,
    ) -> AsyncIterator[StreamEvent]:
        if not options.api_key:
            yield StreamError(
                "Missing Anthropic API key. Set ANTHROPIC_API_KEY or run `spice config set api-key <key>`.",
                kind="authentication",
                provider=model.provider,
                model=model.id,
            )
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            yield StreamError("Package `anthropic` is not installed. Run `uv add anthropic`.", kind="unsupported")
            return

        system, anthropic_messages = _messages_to_anthropic(messages)
        base_url = options.base_url or self.default_base_url
        client = AsyncAnthropic(api_key=options.api_key, base_url=base_url, max_retries=0)
        started = time.perf_counter()
        try:
            logger.info(
                "provider_request_start provider=Anthropic model=%s messages=%d tools=%d base_url=%s",
                model.id,
                len(messages),
                len(tools),
                bool(base_url),
            )
            stream_messages = cast(Any, client.messages.stream)
            async with stream_messages(
                model=model.id,
                system=system or None,
                messages=anthropic_messages,
                tools=[_tool_to_anthropic(tool) for tool in tools] or None,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            ) as stream:
                async for event in stream:
                    # The SDK's MessageStream emits a high-level "text" event per text delta.
                    if event.type == "text":
                        yield TextDelta(event.text)
                message = await stream.get_final_message()
                for block in message.content:
                    if getattr(block, "type", None) == "tool_use":
                        yield ToolCallEvent(id=block.id, name=block.name, arguments=block.input or {})
                logger.info(
                    "provider_request_end provider=Anthropic model=%s stop_reason=%s duration_ms=%d",
                    model.id,
                    message.stop_reason,
                    int((time.perf_counter() - started) * 1000),
                )
                yield Done(message.stop_reason)
        except Exception as exc:
            logger.exception(
                "provider_request_error provider=Anthropic model=%s duration_ms=%d",
                model.id,
                int((time.perf_counter() - started) * 1000),
            )
            yield stream_error_from_exception(
                exc,
                prefix="Provider request failed",
                provider=model.provider,
                model=model.id,
            )


def _messages_to_anthropic(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    output: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role == "tool":
            output.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                            "is_error": message.is_error,
                        }
                    ],
                }
            )
        elif message.role == "assistant":
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tc in message.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            output.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
        else:
            output.append({"role": "user", "content": message.content})
    return "\n\n".join(system_parts), output


def _tool_to_anthropic(tool: ToolSchema) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }
