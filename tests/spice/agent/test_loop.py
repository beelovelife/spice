from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

import pytest

import spice.agent.loop as loop_module
from spice.agent.events import (
    AgentErrorEvent,
    AssistantMessageEvent,
    ReasoningDeltaEvent,
    ToolExecutionEndEvent,
    TurnEndEvent,
)
from spice.agent.loop import run_turn
from spice.llm.messages import Message
from spice.llm.models import Model, ModelPricing
from spice.llm.retry import ModelRetryPolicy
from spice.llm.routing import ModelCandidate, ModelRoute
from spice.llm.types import (
    Done,
    ModelRequestOptions,
    ReasoningDelta,
    StreamError,
    TextDelta,
    ToolCallEvent,
)
from spice.llm.usage import TokenUsage
from spice.tools.base import Tool, ToolContext, tool_error, tool_result


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
            yield Done(
                "stop",
                TokenUsage(
                    input_tokens=100,
                    output_tokens=20,
                    cache_read_tokens=60,
                    cache_metrics_available=True,
                ),
            )

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

        assistant = next(
            event for event in events if isinstance(event, AssistantMessageEvent)
        )
        persisted = messages[-1].metadata["usage"]
        self.assertEqual(assistant.usage.tokens.input_tokens, 100)
        self.assertEqual(persisted["model_calls"], 1)
        self.assertEqual(persisted["cache_read_tokens"], 60)
        self.assertEqual(persisted["estimated_cost_usd"], "0.000086")

    async def test_reasoning_is_displayed_and_only_kept_for_live_tool_roundtrip(
        self,
    ) -> None:
        calls = 0
        seen_second_round: list[Message] = []

        async def fake_stream_model(model, messages, tools, options):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield ReasoningDelta("check inputs")
                yield ToolCallEvent(id="tc1", name="demo", arguments={"value": "x"})
                yield Done("tool_calls")
            else:
                seen_second_round.extend(messages)
                yield TextDelta("done")
                yield Done("stop")

        loop_module.stream_model = fake_stream_model
        messages = [Message(role="system", content="")]
        tool = Tool(
            name="demo",
            description="demo",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            execute=_ok_tool,
        )

        events = [
            event
            async for event in run_turn(
                prompt="hi",
                messages=messages,
                model=Model(id="fake", provider="fake"),
                tools=[tool],
                options=ModelRequestOptions(),
                cwd=Path.cwd(),
                confirm=None,
            )
        ]

        self.assertEqual(
            [event.text for event in events if isinstance(event, ReasoningDeltaEvent)],
            ["check inputs"],
        )
        persisted_tool_turn = next(
            message
            for message in messages
            if message.role == "assistant" and message.tool_calls
        )
        live_tool_turn = next(
            message
            for message in seen_second_round
            if message.role == "assistant" and message.tool_calls
        )
        self.assertNotIn("_reasoning_content", persisted_tool_turn.metadata)
        self.assertEqual(live_tool_turn.metadata["_reasoning_content"], "check inputs")

    async def test_requires_confirmation_defaults_to_deny(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield ToolCallEvent(
                id="tc1", name="write_file", arguments={"path": "x.txt", "content": "x"}
            )
            yield Done("tool_calls")

        loop_module.stream_model = fake_stream_model
        tool = Tool(
            name="write_file",
            description="write",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
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
        end = next(
            event for event in events if isinstance(event, ToolExecutionEndEvent)
        )
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
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
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
        end = next(
            event for event in events if isinstance(event, ToolExecutionEndEvent)
        )
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
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
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
        self.assertIsInstance(events[-2], AssistantMessageEvent)
        self.assertIsInstance(events[-1], TurnEndEvent)
        self.assertEqual(events[-1].stop_reason, "max_tool_rounds")
        self.assertEqual(events[-1].status, "paused")
        self.assertIn("100 轮工具预算", events[-1].text)
        self.assertFalse(any(isinstance(event, AgentErrorEvent) for event in events))
        self.assertEqual(
            sum(isinstance(event, ToolExecutionEndEvent) for event in events), 100
        )

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
        self.assertEqual(
            [message.content for message in messages if message.role == "user"], ["hi"]
        )
        self.assertNotIn(
            "Runtime state", "\n".join(message.content for message in messages)
        )
        self.assertEqual(seen_messages[0][-2].role, "system")
        self.assertIn("Runtime state", seen_messages[0][-2].content)

    async def test_tool_exception_is_redacted_before_tool_result(self) -> None:
        async def fake_stream_model(model, messages, tools, options):
            yield ToolCallEvent(id="tc1", name="demo", arguments={})
            yield Done("tool_calls")

        async def execute(args: dict, context: ToolContext):
            raise RuntimeError(
                "Authorization: Bearer sk-proj-abcdefghijklmnop api_key=super-secret-token"
            )

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

        end = next(
            event for event in events if isinstance(event, ToolExecutionEndEvent)
        )
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

        assistant = next(
            event for event in events if isinstance(event, AssistantMessageEvent)
        )
        error = next(event for event in events if isinstance(event, AgentErrorEvent))
        end = events[-1]
        self.assertEqual(assistant.text, "partial")
        self.assertEqual(error.message, "provider failed")
        self.assertIsInstance(end, TurnEndEvent)
        self.assertEqual(end.stop_reason, "error")
        self.assertEqual(end.text, "partial")
        self.assertEqual(messages[-1].role, "assistant")
        self.assertEqual(messages[-1].content, "partial")


@pytest.mark.parametrize("source", ["prompt", "runtime", "schema", "reasoning"])
def test_context_guard_counts_full_request_without_compactor(tmp_path, monkeypatch, source):
    calls = []

    async def stream(model, messages, tools, options):
        calls.append(messages)
        yield Done("stop")

    monkeypatch.setattr(loop_module, "stream_model", stream)
    messages = [Message(role="system", content="rules")]
    if source == "reasoning":
        messages.append(Message(role="assistant", metadata={"_reasoning_content": "x" * 2000}))
    tools = [Tool("demo", "x" * 2000, {"type": "object"}, _ok_tool)] if source == "schema" else []

    async def run():
        return [event async for event in run_turn(
            prompt="x" * 2000 if source == "prompt" else "continue",
            messages=messages, model=Model(id="tiny", provider="fake", context_window=1000, output_tokens=100),
            tools=tools, options=ModelRequestOptions(), cwd=tmp_path, confirm=None,
            runtime_context="x" * 2000 if source == "runtime" else None,
        )]

    events = asyncio.run(run())
    assert calls == []
    assert isinstance(events[-2], AgentErrorEvent)
    assert events[-2].kind == "context_limit"
    assert events[-1].stop_reason == "context_limit"


@pytest.mark.parametrize("stop", ["fatal", "round_limit"])
def test_context_guard_does_not_compact_after_terminal_round(tmp_path, monkeypatch, stop):
    compacted = []

    async def stream(model, messages, tools, options):
        yield ToolCallEvent("tc1", "demo", {})
        yield Done("tool_calls")

    async def execute(args, context):
        if stop == "fatal":
            return tool_error("x" * 2000, disposition="fatal")
        return tool_result("x" * 2000)

    async def compact(model, options):
        compacted.append(model)

    monkeypatch.setattr(loop_module, "stream_model", stream)

    async def run():
        return [event async for event in run_turn(
            prompt="continue", messages=[Message(role="system", content="rules")],
            model=Model(id="tiny", provider="fake", context_window=1000, output_tokens=100),
            tools=[Tool("demo", "demo", {"type": "object"}, execute)],
            options=ModelRequestOptions(), cwd=tmp_path, confirm=None,
            max_tool_rounds=1 if stop == "round_limit" else 3, compact_context=compact,
        )]

    events = asyncio.run(run())
    assert compacted == []
    assert [e.stop_reason for e in events if isinstance(e, TurnEndEvent)] == [
        "fatal_tool_error" if stop == "fatal" else "max_tool_rounds"
    ]


def test_context_guard_uses_active_fallback_model_budget(tmp_path):
    calls = []
    primary = Model(id="primary", provider="fake", context_window=10_000, output_tokens=100)
    fallback = Model(id="fallback", provider="fake", context_window=1000, output_tokens=100)
    options = ModelRequestOptions(max_tokens=100)

    async def stream(model, messages, tools, options):
        calls.append(model.id)
        if model is primary:
            yield StreamError("unavailable", retryable=True)
        else:
            yield ToolCallEvent("tc1", "demo", {})
            yield Done("tool_calls")

    async def execute(args, context):
        return tool_result("x" * 2000)

    route = ModelRoute(
        [ModelCandidate(m.id, m, options) for m in [primary, fallback]],
        retry_policy=ModelRetryPolicy(enabled=False, max_attempts=1),
        fallback_enabled=True, stream_factory=stream,
    )

    async def run():
        return [event async for event in run_turn(
            prompt="continue", messages=[Message(role="system", content="rules")],
            model=primary, model_route=route,
            tools=[Tool("demo", "demo", {"type": "object"}, execute)],
            options=options, cwd=tmp_path, confirm=None,
        )]

    events = asyncio.run(run())
    assert calls == ["primary", "fallback"]
    assert events[-1].stop_reason == "context_limit"
