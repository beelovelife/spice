from __future__ import annotations

import unittest

from spice.llm.messages import Message, ToolCall
from spice.llm.providers.anthropic import _messages_to_anthropic
import spice.llm.providers.gemini as gemini_module
from spice.llm.providers.gemini import _fallback_tool_call_id, _messages_to_gemini
from spice.llm.providers.openai import _messages_to_openai, _messages_to_responses, _tool_to_response
from spice.llm.types import ToolSchema


class ProviderConversionTests(unittest.TestCase):
    def test_openai_tool_roundtrip_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})]),
            Message(role="tool", tool_call_id="tc1", name="demo", content="done"),
        ]
        converted = _messages_to_openai(messages)
        self.assertEqual(converted[2]["tool_calls"][0]["function"]["name"], "demo")
        self.assertEqual(converted[3]["role"], "tool")
        self.assertEqual(converted[3]["tool_call_id"], "tc1")

    def test_openai_responses_tool_roundtrip_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="checking"),
            Message(role="assistant", tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})]),
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
            Message(role="assistant", tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})]),
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
            Message(role="assistant", tool_calls=[ToolCall(id="tc1", name="demo", arguments={"x": 1})]),
            Message(role="tool", tool_call_id="tc1", name="demo", content="done"),
        ]
        system, converted = _messages_to_gemini(messages)
        self.assertEqual(system, "sys")
        self.assertEqual(converted[0]["parts"][0]["function_call"]["name"], "demo")
        self.assertEqual(converted[1]["parts"][0]["function_response"]["name"], "demo")

    def test_gemini_function_response_includes_id_when_tool_call_id_present(self) -> None:
        messages = [
            Message(role="tool", tool_call_id="call_42", name="read_file", content="hello"),
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
                    ToolCall(id="read_file_0", name="read_file", arguments={"path": "a.txt"}),
                    ToolCall(id="read_file_1", name="read_file", arguments={"path": "b.txt"}),
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
            self.assertEqual(_fallback_tool_call_id("read file"), "read_file_abcdef123456")
        finally:
            gemini_module.uuid4 = original_uuid4
