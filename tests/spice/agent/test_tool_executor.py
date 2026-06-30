from __future__ import annotations

import asyncio
import time
from functools import wraps
from pathlib import Path

import spice.agent.loop as loop_module
from spice.agent.events import AgentErrorEvent, ToolExecutionEndEvent, TurnEndEvent
from spice.agent.loop import run_turn
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.types import Done, ModelRequestOptions, ToolCallEvent
from spice.tools.base import Tool, ToolContext, fatal_tool_error, tool_result


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _tool(name: str, execute, *, concurrency: str = "parallel", timeout: float | None = None) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
        concurrency=concurrency,
        timeout_seconds=timeout,
    )


@async_test
async def test_parallel_completion_events_but_ordered_tool_messages(monkeypatch) -> None:
    requests = 0
    observed_tool_order: list[str] = []

    async def stream_model(model, messages, tools, options):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield ToolCallEvent("slow-id", "slow", {})
            yield ToolCallEvent("fast-id", "fast", {})
            yield Done("tool_calls")
        else:
            observed_tool_order.extend(message.tool_call_id for message in messages if message.role == "tool")
            yield Done("stop")

    async def slow(args: dict, context: ToolContext):
        await asyncio.sleep(0.05)
        return tool_result("slow")

    async def fast(args: dict, context: ToolContext):
        await asyncio.sleep(0.005)
        return tool_result("fast")

    monkeypatch.setattr(loop_module, "stream_model", stream_model)
    events = [
        event
        async for event in run_turn(
            prompt="run",
            messages=[Message(role="system", content="")],
            model=Model(id="test", provider="test"),
            tools=[_tool("slow", slow), _tool("fast", fast)],
            options=ModelRequestOptions(),
            cwd=Path.cwd(),
            confirm=None,
        )
    ]

    completed = [event.tool_call_id for event in events if isinstance(event, ToolExecutionEndEvent)]
    assert completed == ["fast-id", "slow-id"]
    assert observed_tool_order == ["slow-id", "fast-id"]


@async_test
async def test_serial_tool_is_a_barrier(monkeypatch) -> None:
    timeline: list[str] = []
    requests = 0

    async def stream_model(model, messages, tools, options):
        nonlocal requests
        requests += 1
        if requests == 1:
            for call_id, name in (("a", "read_a"), ("b", "read_b"), ("c", "write"), ("d", "read_d")):
                yield ToolCallEvent(call_id, name, {})
            yield Done("tool_calls")
        else:
            yield Done("stop")

    def execute(name: str, delay: float = 0):
        async def inner(args: dict, context: ToolContext):
            timeline.append(f"{name}:start")
            await asyncio.sleep(delay)
            timeline.append(f"{name}:end")
            return tool_result(name)

        return inner

    monkeypatch.setattr(loop_module, "stream_model", stream_model)
    await _collect(
        run_turn(
            prompt="run",
            messages=[Message(role="system", content="")],
            model=Model(id="test", provider="test"),
            tools=[
                _tool("read_a", execute("a", 0.02)),
                _tool("read_b", execute("b", 0.01)),
                _tool("write", execute("c"), concurrency="serial"),
                _tool("read_d", execute("d")),
            ],
            options=ModelRequestOptions(),
            cwd=Path.cwd(),
            confirm=None,
        )
    )

    assert timeline.index("c:start") > timeline.index("a:end")
    assert timeline.index("c:start") > timeline.index("b:end")
    assert timeline.index("d:start") > timeline.index("c:end")


@async_test
async def test_tool_timeout_is_recoverable(monkeypatch) -> None:
    requests = 0

    async def stream_model(model, messages, tools, options):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield ToolCallEvent("tc1", "wait", {})
            yield Done("tool_calls")
        else:
            yield Done("stop")

    async def wait(args: dict, context: ToolContext):
        await asyncio.sleep(1)
        return tool_result("late")

    monkeypatch.setattr(loop_module, "stream_model", stream_model)
    events = await _collect(
        run_turn(
            prompt="run",
            messages=[Message(role="system", content="")],
            model=Model(id="test", provider="test"),
            tools=[_tool("wait", wait, timeout=0.01)],
            options=ModelRequestOptions(),
            cwd=Path.cwd(),
            confirm=None,
        )
    )

    result = next(event.result for event in events if isinstance(event, ToolExecutionEndEvent))
    assert result.error_code == "tool_timeout"
    assert result.disposition == "recoverable"
    assert requests == 2


@async_test
async def test_fatal_parallel_result_cancels_sibling_and_turn(monkeypatch) -> None:
    sibling_cancelled = asyncio.Event()
    requests = 0

    async def stream_model(model, messages, tools, options):
        nonlocal requests
        requests += 1
        yield ToolCallEvent("fatal-id", "fatal", {})
        yield ToolCallEvent("sibling-id", "sibling", {})
        yield Done("tool_calls")

    async def fatal(args: dict, context: ToolContext):
        await asyncio.sleep(0.01)
        return fatal_tool_error("stop", code="unsafe")

    async def sibling(args: dict, context: ToolContext):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return tool_result("late")

    monkeypatch.setattr(loop_module, "stream_model", stream_model)
    started = time.monotonic()
    events = await _collect(
        run_turn(
            prompt="run",
            messages=[Message(role="system", content="")],
            model=Model(id="test", provider="test"),
            tools=[_tool("fatal", fatal), _tool("sibling", sibling)],
            options=ModelRequestOptions(),
            cwd=Path.cwd(),
            confirm=None,
        )
    )

    assert time.monotonic() - started < 0.5
    assert sibling_cancelled.is_set()
    assert requests == 1
    assert any(isinstance(event, AgentErrorEvent) for event in events)
    assert next(event for event in events if isinstance(event, TurnEndEvent)).stop_reason == "fatal_tool_error"
    results = {event.tool_call_id: event.result for event in events if isinstance(event, ToolExecutionEndEvent)}
    assert results["fatal-id"].error_code == "unsafe"
    assert results["sibling-id"].error_code == "tool_batch_cancelled"


@async_test
async def test_fatal_parallel_result_preserves_already_completed_sibling(monkeypatch) -> None:
    async def stream_model(model, messages, tools, options):
        yield ToolCallEvent("fatal-id", "fatal", {})
        yield ToolCallEvent("finished-id", "finished", {})
        yield Done("tool_calls")

    async def fatal(args: dict, context: ToolContext):
        await asyncio.sleep(0.02)
        return fatal_tool_error("stop", code="unsafe")

    async def finished(args: dict, context: ToolContext):
        await asyncio.sleep(0.005)
        return tool_result("actually completed")

    monkeypatch.setattr(loop_module, "stream_model", stream_model)
    events = await _collect(
        run_turn(
            prompt="run",
            messages=[Message(role="system", content="")],
            model=Model(id="test", provider="test"),
            tools=[_tool("fatal", fatal), _tool("finished", finished)],
            options=ModelRequestOptions(),
            cwd=Path.cwd(),
            confirm=None,
        )
    )

    results = {event.tool_call_id: event.result for event in events if isinstance(event, ToolExecutionEndEvent)}
    assert results["finished-id"].content == "actually completed"
    assert results["finished-id"].error_code is None


async def _collect(iterator):
    return [event async for event in iterator]
