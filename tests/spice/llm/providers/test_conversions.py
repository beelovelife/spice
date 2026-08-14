from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.llm.providers.anthropic import _messages_to_anthropic
import spice.llm.providers.gemini as gemini_module
from spice.llm.providers.gemini import (
    GeminiProvider,
    _fallback_tool_call_id,
    _messages_to_gemini,
)
from spice.llm.providers.openai import (
    OpenAIProvider,
    _messages_to_openai,
    _messages_to_responses,
    _tool_to_response,
)
from spice.llm.types import (
    Done,
    ModelRequestOptions,
    ReasoningDelta,
    TextDelta,
    ToolSchema,
)


class ProviderConversionTests(unittest.TestCase):
    def test_openai_tool_roundtrip_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})],
            ),
            Message(role="tool", tool_call_id="tc1", name="demo", content="done"),
        ]
        converted = _messages_to_openai(messages)
        self.assertEqual(converted[2]["tool_calls"][0]["function"]["name"], "demo")
        self.assertEqual(converted[3]["role"], "tool")
        self.assertEqual(converted[3]["tool_call_id"], "tc1")

    def test_openai_chat_roundtrip_includes_ephemeral_reasoning_for_tool_turn(
        self,
    ) -> None:
        messages = [
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc1", name="demo", arguments={})],
                metadata={"_reasoning_content": "check the inputs"},
            )
        ]

        converted = _messages_to_openai(messages)

        self.assertEqual(converted[0]["reasoning_content"], "check the inputs")

    def test_openai_chat_stream_emits_reasoning_delta(self) -> None:
        async def chunks():
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content="checking",
                            content=None,
                            tool_calls=[],
                        ),
                        finish_reason=None,
                    )
                ],
            )
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content="done",
                            tool_calls=[],
                        ),
                        finish_reason="stop",
                    )
                ],
            )

        async def create(**_kwargs):
            return chunks()

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        async def collect():
            provider = OpenAIProvider(provider_name="DeepSeek")
            return [
                event
                async for event in provider._astream_chat_completions(
                    client,
                    Model(id="deepseek-v4-pro", provider="deepseek"),
                    [Message(role="user", content="hi")],
                    [],
                    ModelRequestOptions(api_key="test-key"),
                    0.0,
                )
            ]

        events = asyncio.run(collect())

        self.assertEqual(
            [event.text for event in events if isinstance(event, ReasoningDelta)],
            ["checking"],
        )
        self.assertEqual(
            [event.text for event in events if isinstance(event, TextDelta)],
            ["done"],
        )

    def test_responses_stream_emits_raw_reasoning_and_requests_reasoning(self) -> None:
        request = {}

        async def chunks():
            yield SimpleNamespace(
                type="response.reasoning_text.delta",
                delta="checking",
            )
            yield SimpleNamespace(type="response.output_text.delta", delta="done")
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output=[], usage=None),
            )

        async def create(**kwargs):
            request.update(kwargs)
            return chunks()

        client = SimpleNamespace(responses=SimpleNamespace(create=create))

        async def collect():
            return [
                event
                async for event in OpenAIProvider(
                    provider_name="DeepSeek", use_responses=True
                )._astream_responses(
                    client,
                    Model(
                        id="deepseek-v4-pro",
                        provider="deepseek",
                        supports_reasoning=True,
                    ),
                    [Message(role="user", content="hi")],
                    [],
                    ModelRequestOptions(api_key="test-key"),
                    0.0,
                )
            ]

        events = asyncio.run(collect())

        self.assertEqual(request["reasoning"], {"summary": "auto"})
        self.assertEqual(
            [event.text for event in events if isinstance(event, ReasoningDelta)],
            ["checking"],
        )
        self.assertEqual(
            [event.text for event in events if isinstance(event, TextDelta)],
            ["done"],
        )
        self.assertIsInstance(events[-1], Done)

    def test_openai_responses_tool_roundtrip_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="checking"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})],
            ),
            Message(role="tool", tool_call_id="tc1", name="demo", content="done"),
        ]
        converted, instructions = _messages_to_responses(messages)

        self.assertEqual(instructions, "sys")
        self.assertEqual(converted[0], {"role": "user", "content": "hi"})
        self.assertEqual(converted[1], {"role": "assistant", "content": "checking"})
        self.assertEqual(converted[2]["type"], "function_call")
        self.assertEqual(converted[2]["call_id"], "tc1")
        self.assertEqual(converted[2]["name"], "demo")
        self.assertEqual(converted[3]["type"], "function_call_output")
        self.assertEqual(converted[3]["call_id"], "tc1")

    def test_openai_responses_roundtrip_includes_reasoning_item_for_tool_turn(
        self,
    ) -> None:
        messages = [
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc1", name="demo", arguments={})],
                metadata={"_reasoning_content": "check the inputs"},
            ),
            Message(role="tool", tool_call_id="tc1", content="done"),
        ]

        converted, _ = _messages_to_responses(messages)

        self.assertEqual(converted[0]["type"], "reasoning")
        self.assertEqual(
            converted[0]["content"],
            [{"type": "reasoning_text", "text": "check the inputs"}],
        )
        self.assertEqual(converted[1]["type"], "function_call")
        self.assertEqual(converted[2]["type"], "function_call_output")

    def test_openai_responses_tool_schema_is_flat_function_shape(self) -> None:
        converted = _tool_to_response(
            ToolSchema("demo", "Demo tool.", {"type": "object", "properties": {}})
        )

        self.assertEqual(converted["type"], "function")
        self.assertEqual(converted["name"], "demo")
        self.assertEqual(converted["parameters"]["type"], "object")

    def test_anthropic_tool_roundtrip_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})],
            ),
            Message(role="tool", tool_call_id="tc1", name="demo", content="done"),
        ]
        system, converted = _messages_to_anthropic(messages)
        self.assertEqual(system, "sys")
        self.assertEqual(converted[0]["content"][0]["type"], "tool_use")
        self.assertEqual(converted[1]["content"][0]["type"], "tool_result")
        self.assertEqual(converted[1]["content"][0]["tool_use_id"], "tc1")

    def test_gemini_tool_roundtrip_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})],
            ),
            Message(role="tool", tool_call_id="tc1", name="demo", content="done"),
        ]
        system, converted = _messages_to_gemini(messages)
        self.assertEqual(system, "sys")
        self.assertEqual(converted[0]["parts"][0]["function_call"]["name"], "demo")
        self.assertEqual(converted[1]["parts"][0]["function_response"]["name"], "demo")

    def test_gemini_function_response_includes_id_when_tool_call_id_present(
        self,
    ) -> None:
        messages = [
            Message(
                role="tool", tool_call_id="call_42", name="read_file", content="hello"
            ),
        ]
        _, converted = _messages_to_gemini(messages)
        response = converted[0]["parts"][0]["function_response"]
        self.assertEqual(response["id"], "call_42")
        self.assertEqual(response["name"], "read_file")

    def test_gemini_function_response_omits_id_when_no_tool_call_id(self) -> None:
        messages = [
            Message(role="tool", name="read_file", content="hello"),
        ]
        _, converted = _messages_to_gemini(messages)
        response = converted[0]["parts"][0]["function_response"]
        self.assertNotIn("id", response)
        self.assertEqual(response["name"], "read_file")

    def test_gemini_multiple_same_name_tool_calls_distinct_ids(self) -> None:
        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="read_file_0", name="read_file", arguments={"path": "a.txt"}
                    ),
                    ToolCall(
                        id="read_file_1", name="read_file", arguments={"path": "b.txt"}
                    ),
                ],
            ),
        ]
        _, converted = _messages_to_gemini(messages)
        parts = converted[0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["function_call"]["name"], "read_file")
        self.assertEqual(parts[0]["function_call"]["args"], {"path": "a.txt"})
        self.assertEqual(parts[1]["function_call"]["name"], "read_file")
        self.assertEqual(parts[1]["function_call"]["args"], {"path": "b.txt"})

    def test_gemini_fallback_tool_call_id_uses_uuid_not_turn_local_index(self) -> None:
        class FakeUuid:
            hex = "abcdef1234567890"

        original_uuid4 = gemini_module.uuid4
        try:
            gemini_module.uuid4 = lambda: FakeUuid()
            self.assertEqual(
                _fallback_tool_call_id("read file"), "read_file_abcdef123456"
            )
        finally:
            gemini_module.uuid4 = original_uuid4

    def test_gemini_provider_awaits_stream_creation_before_iterating(self) -> None:
        awaited = False

        async def chunks():
            part = SimpleNamespace(text="hello", function_call=None)
            content = SimpleNamespace(parts=[part])
            yield SimpleNamespace(
                usage_metadata=None, candidates=[SimpleNamespace(content=content)]
            )

        async def generate_content_stream(**_kwargs):
            nonlocal awaited
            awaited = True
            return chunks()

        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content_stream=generate_content_stream)
            )
        )

        async def collect():
            with patch("google.genai.Client", return_value=client):
                return [
                    event
                    async for event in GeminiProvider().astream(
                        Model(id="gemini-test", provider="gemini"),
                        [Message(role="user", content="hi")],
                        [],
                        ModelRequestOptions(api_key="test-key"),
                    )
                ]

        events = asyncio.run(collect())
        self.assertTrue(awaited)
        self.assertEqual(
            [event.text for event in events if isinstance(event, TextDelta)], ["hello"]
        )
        self.assertIsInstance(events[-1], Done)
