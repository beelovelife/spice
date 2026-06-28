"""OpenAI provider."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from spice.agent.logging_config import get_logger
from spice.llm.error_safety import public_exception_message
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.types import Done, StreamError, StreamEvent, ModelRequestOptions, TextDelta, ToolCallEvent, ToolSchema

logger = get_logger(__name__)

class OpenAIProvider:
    def __init__(
        self,
        *,
        provider_name: str = "OpenAI",
        api_key_hint: str = "OPENAI_API_KEY",
        default_base_url: str | None = None,
        use_responses: bool = False,
    ) -> None:
        self.provider_name = provider_name
        self.api_key_hint = api_key_hint
        self.default_base_url = default_base_url
        self.use_responses = use_responses

    async def astream(
        self,
        model: Model,
        messages: list[Message],
        tools: list[ToolSchema],
        options: ModelRequestOptions,
    ) -> AsyncIterator[StreamEvent]:
        if not options.api_key:
            yield StreamError(f"Missing {self.provider_name} API key. Set {self.api_key_hint} or run `spice config set api-key <key>`.")
            return
        try:
            from openai import AsyncOpenAI
        except ImportError:
            yield StreamError("Package `openai` is not installed. Run `uv add openai`.")
            return

        client = AsyncOpenAI(api_key=options.api_key, base_url=options.base_url or self.default_base_url)
        started = time.perf_counter()
        try:
            logger.info(
                "provider_request_start provider=%s model=%s messages=%d tools=%d base_url=%s",
                self.provider_name,
                model.id,
                len(messages),
                len(tools),
                bool(options.base_url or self.default_base_url),
            )
            if self.use_responses:
                async for event in self._astream_responses(client, model, messages, tools, options, started):
                    yield event
            else:
                async for event in self._astream_chat_completions(client, model, messages, tools, options, started):
                    yield event
        except Exception as exc:
            logger.exception(
                "provider_request_error provider=%s model=%s duration_ms=%d",
                self.provider_name,
                model.id,
                int((time.perf_counter() - started) * 1000),
            )
            yield StreamError(public_exception_message(exc, prefix="Provider request failed"))

    async def _astream_responses(
        self,
        client: Any,
        model: Model,
        messages: list[Message],
        tools: list[ToolSchema],
        options: ModelRequestOptions,
        started: float,
    ) -> AsyncIterator[StreamEvent]:
        create_response = cast(Any, client.responses.create)
        response_input, instructions = _messages_to_responses(messages)
        stream = await create_response(
            model=model.id,
            input=response_input,
            instructions=instructions,
            tools=[_tool_to_response(tool) for tool in tools] or None,
            temperature=options.temperature,
            max_output_tokens=options.max_tokens,
            stream=True,
        )
        async for event in stream:
            event_type = _get_event_type(event)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    yield TextDelta(str(delta))
            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if _get_item_type(item) == "function_call":
                    yield ToolCallEvent(
                        id=str(getattr(item, "call_id", "") or getattr(item, "id", "")),
                        name=str(getattr(item, "name", "")),
                        arguments=_parse_json_object(str(getattr(item, "arguments", "") or "")),
                    )
            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                tool_count = _count_response_function_calls(response)
                logger.info(
                    "provider_request_end provider=%s model=%s finish_reason=%s duration_ms=%d tool_calls=%d",
                    self.provider_name,
                    model.id,
                    "completed",
                    int((time.perf_counter() - started) * 1000),
                    tool_count,
                )
                yield Done("completed")

    async def _astream_chat_completions(
        self,
        client: Any,
        model: Model,
        messages: list[Message],
        tools: list[ToolSchema],
        options: ModelRequestOptions,
        started: float,
    ) -> AsyncIterator[StreamEvent]:
        tool_calls: dict[int, dict[str, Any]] = {}
        create_completion = cast(Any, client.chat.completions.create)
        stream = await create_completion(
            model=model.id,
            messages=_messages_to_openai(messages),
            tools=[_tool_to_openai(tool) for tool in tools] or None,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            if delta.content:
                yield TextDelta(delta.content)
            for tc in delta.tool_calls or []:
                index = tc.index
                current = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    current["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        current["name"] = tc.function.name
                    if tc.function.arguments:
                        current["arguments"] += tc.function.arguments
            if choice.finish_reason:
                for current in tool_calls.values():
                    yield ToolCallEvent(
                        id=current["id"],
                        name=current["name"],
                        arguments=_parse_json_object(current["arguments"]),
                    )
                logger.info(
                    "provider_request_end provider=%s model=%s finish_reason=%s duration_ms=%d tool_calls=%d",
                    self.provider_name,
                    model.id,
                    choice.finish_reason,
                    int((time.perf_counter() - started) * 1000),
                    len(tool_calls),
                )
                yield Done(choice.finish_reason)


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in message.tool_calls
                ]
            output.append(item)
        elif message.role == "tool":
            output.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content})
        else:
            output.append({"role": message.role, "content": message.content})
    return output


def _messages_to_responses(messages: list[Message]) -> tuple[list[dict[str, Any]], str | None]:
    input_items: list[dict[str, Any]] = []
    instructions: list[str] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                instructions.append(message.content)
        elif message.role == "assistant":
            if message.content:
                input_items.append({"role": "assistant", "content": message.content})
            for tc in message.tool_calls:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    }
                )
        elif message.role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        else:
            input_items.append({"role": message.role, "content": message.content})
    return input_items, "\n\n".join(instructions) or None


def _tool_to_openai(tool: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _tool_to_response(tool: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _get_event_type(event: Any) -> str:
    return str(getattr(event, "type", "") or "")


def _get_item_type(item: Any) -> str:
    if item is None:
        return ""
    return str(getattr(item, "type", "") or "")


def _count_response_function_calls(response: Any) -> int:
    output = getattr(response, "output", None) or []
    return sum(1 for item in output if _get_item_type(item) == "function_call")


def _parse_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return data if isinstance(data, dict) else {"value": data}
