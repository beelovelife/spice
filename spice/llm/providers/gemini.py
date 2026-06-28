"""Gemini provider."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from spice.agent.logging_config import get_logger
from spice.llm.error_safety import public_exception_message
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.types import Done, StreamError, StreamEvent, ModelRequestOptions, TextDelta, ToolCallEvent, ToolSchema

logger = get_logger(__name__)


class GeminiProvider:
    async def astream(
        self,
        model: Model,
        messages: list[Message],
        tools: list[ToolSchema],
        options: ModelRequestOptions,
    ) -> AsyncIterator[StreamEvent]:
        if not options.api_key:
            yield StreamError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
            return
        try:
            from google import genai
        except ImportError:
            yield StreamError("Package `google-genai` is not installed. Run `uv add google-genai`.")
            return

        started = time.perf_counter()
        try:
            logger.info(
                "provider_request_start provider=Gemini model=%s messages=%d tools=%d",
                model.id,
                len(messages),
                len(tools),
            )
            client = genai.Client(api_key=options.api_key)
            system, contents = _messages_to_gemini(messages)
            config: dict[str, Any] = {
                "temperature": options.temperature,
                "max_output_tokens": options.max_tokens,
            }
            if system:
                config["system_instruction"] = system
            if tools:
                config["tools"] = [{"function_declarations": [_tool_to_gemini(tool) for tool in tools]}]
            stream = client.aio.models.generate_content_stream(
                model=model.id,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                for candidate in chunk.candidates or []:
                    content = candidate.content
                    for part in content.parts if content and content.parts else []:
                        text = getattr(part, "text", None)
                        if text:
                            yield TextDelta(text)
                        function_call = getattr(part, "function_call", None)
                        if function_call:
                            fc_id = getattr(function_call, "id", "") or _fallback_tool_call_id(function_call.name)
                            yield ToolCallEvent(
                                id=fc_id,
                                name=function_call.name,
                                arguments=dict(function_call.args or {}),
                            )
            logger.info(
                "provider_request_end provider=Gemini model=%s duration_ms=%d",
                model.id,
                int((time.perf_counter() - started) * 1000),
            )
            yield Done("stop")
        except Exception as exc:
            logger.exception(
                "provider_request_error provider=Gemini model=%s duration_ms=%d",
                model.id,
                int((time.perf_counter() - started) * 1000),
            )
            yield StreamError(public_exception_message(exc, prefix="Provider request failed"))


def _messages_to_gemini(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role == "tool":
            response_dict: dict[str, Any] = {
                "name": message.name or "tool",
                "response": {"output": message.content, "is_error": message.is_error},
            }
            if message.tool_call_id:
                response_dict["id"] = message.tool_call_id
            contents.append(
                {
                    "role": "user",
                    "parts": [{"function_response": response_dict}],
                }
            )
        elif message.role == "assistant":
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for tc in message.tool_calls:
                parts.append({"function_call": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        else:
            contents.append({"role": "user", "parts": [{"text": message.content}]})
    return "\n\n".join(system_parts), contents


def _tool_to_gemini(tool: ToolSchema) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _fallback_tool_call_id(name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name) or "tool")
    return f"{safe_name}_{uuid4().hex[:12]}"
