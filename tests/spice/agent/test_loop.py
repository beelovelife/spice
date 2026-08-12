from __future__ import annotations

import unittest
from pathlib import Path

import spice.agent.loop as loop_module
from spice.agent.events import AgentErrorEvent, AssistantMessageEvent, ToolExecutionEndEvent, TurnEndEvent
from spice.agent.loop import run_turn
from spice.llm.messages import Message
from spice.llm.models import Model, ModelPricing
from spice.llm.types import Done, ModelRequestOptions, StreamError, TextDelta, ToolCallEvent
from spice.llm.usage import TokenUsage
from spice.tools.base import Tool, ToolContext, tool_result


async def _ok_tool(args: dict, context: ToolContext):
    return tool_result(f"ok:{args['value']}")


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_stream_model = loop_module.stream_model

    async def asyncTearDown(self) -> None:
        loop_module.stream_model = self.original_stream_model

    async def test_records_model_usage_on_assistant_message(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield TextDelta("done")
            yield Done("stop", TokenUsage(input_tokens=100, output_tokens=20, cache_read_tokens=60, cache_metrics_available=True))

        loop_module.stream_model = fake_stream_model
        messages = [Message(role="system", content="")]
        model = Model(
            id="fake",
            provider="fake",
            pricing=ModelPricing("1", "2", cache_read_per_million_usd="0.1"),
        )

        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=messages,
                model=model,
                tools=[],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
            )
        ]

        assistant = next(event for event in events if isinstance(event, AssistantMessageEvent))
        persisted = messages[-1].metadata["usage"]
        self.assertEqual(assistant.usage.tokens.input_tokens, 100)
        self.assertEqual(persisted["model_calls"], 1)
        self.assertEqual(persisted["cache_read_tokens"], 60)
        self.assertEqual(persisted["estimated_cost_usd"], "0.000086")

    async def test_requires_confirmation_defaults_to_deny(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield ToolCallEvent(id="tc1", name="write_file", arguments={"path": "x.txt", "content": "x"})
            yield Done("tool_calls")

        loop_module.stream_model = fake_stream_model
        tool = Tool(
            name="write_file",
            description="write",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            execute=_ok_tool,
            requires_confirmation=True,
        )
        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=[Message(role="system", content="")],
                model=Model(id="fake", provider="fake"),
                tools=[tool],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
            )
        ]
        end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
        self.assertTrue(end.result.is_error)
        self.assertIn("requires confirmation", end.result.content)

    async def test_invalid_tool_arguments_are_rejected_before_execution(self) -> None:
        executed = False

        async def fake_stream_model(model, messages, tools, options):
            yield ToolCallEvent(id="tc1", name="demo", arguments={})
            yield Done("tool_calls")

        async def execute(args: dict, context: ToolContext):
            nonlocal executed
            executed = True
            return tool_result("should not run")

        loop_module.stream_model = fake_stream_model
        tool = Tool(
            name="demo",
            description="demo",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            execute=execute,
        )
        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=[Message(role="system", content="")],
                model=Model(id="fake", provider="fake"),
                tools=[tool],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
            )
        ]
        end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
        self.assertFalse(executed)
        self.assertTrue(end.result.is_error)
        self.assertIn("missing required argument", end.result.content)

    async def test_tool_round_limit_stops_loop(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield ToolCallEvent(id="tc1", name="demo", arguments={"value": "x"})
            yield Done("tool_calls")

        loop_module.stream_model = fake_stream_model
        tool = Tool(
            name="demo",
            description="demo",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            execute=_ok_tool,
        )
        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=[Message(role="system", content="")],
                model=Model(id="fake", provider="fake"),
                tools=[tool],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
            )
        ]
        self.assertIsInstance(events[-1], AgentErrorEvent)
        self.assertIn("Stopped after", events[-1].message)
        self.assertIn("30 tool rounds", events[-1].message)

    async def test_subagent_manager_is_passed_to_tool_context(self) -> None:
        sentinel = object()
        seen = None
        calls = 0

        async def fake_stream_model(model, messages, tools, options):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield ToolCallEvent(id="tc1", name="demo", arguments={})
                yield Done("tool_calls")
            else:
                yield TextDelta("done")
                yield Done("stop")

        async def execute(args: dict, context: ToolContext):
            nonlocal seen
            seen = context.subagent_manager
            return tool_result("ok")

        loop_module.stream_model = fake_stream_model
        tool = Tool(
            name="demo",
            description="demo",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
        [
            event
            async for event in run_turn(
                prompt="hi",
                messages=[Message(role="system", content="")],
                model=Model(id="fake", provider="fake"),
                tools=[tool],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
                subagent_manager=sentinel,
            )
        ]

        self.assertIs(seen, sentinel)

    async def test_runtime_context_is_model_only_and_not_persisted(self) -> None:
        seen_messages = []

        async def fake_stream_model(model, messages, tools, options):
            seen_messages.append(list(messages))
            yield TextDelta("done")
            yield Done("stop")

        loop_module.stream_model = fake_stream_model
        messages = [Message(role="system", content="system")]
        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=messages,
                model=Model(id="fake", provider="fake"),
                tools=[],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
                runtime_context="Runtime state for this turn only.",
            )
        ]

        self.assertEqual(events[-1].text, "done")
        self.assertEqual([message.content for message in messages if message.role == "user"], ["hi"])
        self.assertNotIn("Runtime state", "\n".join(message.content for message in messages))
        self.assertEqual(seen_messages[0][-2].role, "system")
        self.assertIn("Runtime state", seen_messages[0][-2].content)

    async def test_tool_exception_is_redacted_before_tool_result(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield ToolCallEvent(id="tc1", name="demo", arguments={})
            yield Done("tool_calls")

        async def execute(args: dict, context: ToolContext):
            raise RuntimeError("Authorization: Bearer sk-proj-abcdefghijklmnop api_key=super-secret-token")

        loop_module.stream_model = fake_stream_model
        tool = Tool(
            name="demo",
            description="demo",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )

        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=[Message(role="system", content="")],
                model=Model(id="fake", provider="fake"),
                tools=[tool],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
                max_tool_rounds=1,
            )
        ]

        end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
        self.assertTrue(end.result.is_error)
        self.assertIn("Tool failed", end.result.content)
        self.assertIn("[redacted]", end.result.content)
        self.assertNotIn("sk-proj", end.result.content)
        self.assertNotIn("super-secret-token", end.result.content)

    async def test_stream_error_emits_partial_message_and_turn_end(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield TextDelta("partial")
            yield StreamError("provider failed")

        loop_module.stream_model = fake_stream_model
        messages = [Message(role="system", content="")]

        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=messages,
                model=Model(id="fake", provider="fake"),
                tools=[],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
            )
        ]

        assistant = next(event for event in events if isinstance(event, AssistantMessageEvent))
        error = next(event for event in events if isinstance(event, AgentErrorEvent))
        end = events[-1]
        self.assertEqual(assistant.text, "partial")
        self.assertEqual(error.message, "provider failed")
        self.assertIsInstance(end, TurnEndEvent)
        self.assertEqual(end.stop_reason, "error")
        self.assertEqual(end.text, "partial")
        self.assertEqual(messages[-1].role, "assistant")
        self.assertEqual(messages[-1].content, "partial")
