from __future__ import annotations

import asyncio

import pytest

import spice.agent.loop as loop_module
from spice.agent.events import AgentErrorEvent, TurnEndEvent
from spice.agent.loop import run_turn
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy, stream_with_retry
from spice.llm.routing import ModelCandidate, ModelRoute
from spice.llm.types import Done, ModelRequestOptions, ModelRetryNotice, StreamError, TextDelta, ToolCallEvent
from spice.tools.base import Tool, tool_result


@pytest.mark.parametrize("ending", ["eof", "exception", "length", "max_tokens", "incomplete"])
def test_incomplete_response_never_executes_tools(tmp_path, monkeypatch, ending):
    executed = []
    closed = []

    async def stream(*_args):
        try:
            yield TextDelta("partial response")
            yield ToolCallEvent("call", "write", {})
            if ending == "exception":
                raise ConnectionError("disconnected")
            if ending != "eof":
                yield Done(ending)
        finally:
            closed.append(True)

    async def execute(*_args):
        executed.append(True)
        return tool_result("done")

    monkeypatch.setattr(loop_module, "stream_model", stream)
    messages = [Message(role="system", content="rules")]

    async def run():
        return [event async for event in run_turn(
            prompt="run", messages=messages, model=Model(id="fake", provider="fake"),
            tools=[Tool("write", "write", {"type": "object"}, execute)],
            options=ModelRequestOptions(), cwd=tmp_path, confirm=None,
        )]

    events = asyncio.run(run())
    assert executed == []
    assert closed == [True]
    assert messages[-1].content == "partial response"
    assert messages[-1].tool_calls == []
    assert isinstance(events[-2], AgentErrorEvent)
    assert isinstance(events[-1], TurnEndEvent) and events[-1].stop_reason == "error"


def test_eof_retries_only_before_committing_output():
    attempts = 0
    closed = []

    async def stream():
        nonlocal attempts
        attempts += 1
        try:
            if attempts == 2:
                yield TextDelta("ok")
                yield Done("stop")
        finally:
            closed.append(attempts)

    async def run():
        return [event async for event in stream_with_retry(
            stream, policy=ModelRetryPolicy(max_attempts=2, base_delay_ms=0), provider="fake", model="fake",
        )]

    events = asyncio.run(run())
    assert [type(event) for event in events] == [ModelRetryNotice, TextDelta, Done]
    assert closed == [1, 2]


@pytest.mark.parametrize("finish,reason", [
    ("incomplete", "max_output_tokens"), ("length", None), ("max_tokens", None),
])
def test_output_limit_error_explains_recovery_without_repeating_request(finish, reason):
    attempts = []

    async def stream():
        attempts.append(True)
        yield Done(finish, incomplete_reason=reason, output_token_limit=16000)

    async def run():
        return [event async for event in stream_with_retry(
            stream, policy=ModelRetryPolicy(max_attempts=3, base_delay_ms=0),
            provider="fake", model="fake",
        )]

    events = asyncio.run(run())
    assert attempts == [True]
    assert len(events) == 1 and isinstance(events[0], StreamError)
    assert "16,000 tokens" in events[0].error
    assert "smaller tool calls" in events[0].error
    assert "output_tokens" in events[0].error
    assert "Tools were not executed" in events[0].error
    assert not events[0].retryable


def test_cancellation_closes_stream_without_retry():
    closed = []

    async def stream():
        try:
            yield TextDelta("partial")
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    async def run():
        events = stream_with_retry(stream, policy=ModelRetryPolicy(), provider="fake", model="fake")
        await anext(events)
        await events.aclose()

    asyncio.run(run())
    assert closed == [True]


def test_unexpected_stream_errors_are_sanitized():
    async def stream():
        raise RuntimeError("api_key=private-value")
        yield Done("stop")

    async def run():
        return [event async for event in stream_with_retry(
            stream, policy=ModelRetryPolicy(), provider="fake", model="fake",
        )]

    events = asyncio.run(run())
    assert len(events) == 1 and isinstance(events[0], StreamError)
    assert "private-value" not in events[0].error


@pytest.mark.parametrize("can_compact", [False, True])
def test_smaller_fallback_is_guarded_before_its_first_request(tmp_path, can_compact):
    primary = Model(id="primary", provider="fake", context_window=100_000, output_tokens=100)
    fallback = Model(id="fallback", provider="fake", context_window=1000, output_tokens=100)
    options = ModelRequestOptions(max_tokens=100)
    messages = [Message(role="system", content="rules"), Message(role="user", content="x" * 4000)]
    requests = []
    compactions = []

    async def stream(model, request_messages, *_args):
        requests.append(model.id)
        if model is primary:
            yield StreamError("unavailable", kind="server", retryable=True)
        else:
            assert all(len(message.content) < 100 for message in request_messages)
            yield Done("stop")

    async def compact(model, _options):
        compactions.append(model.id)
        messages[:] = [Message(role="system", content="checkpoint"), Message(role="user", content="continue")]

    route = ModelRoute(
        [ModelCandidate(model.id, model, options) for model in (primary, fallback)],
        retry_policy=ModelRetryPolicy(enabled=False), fallback_enabled=True, stream_factory=stream,
    )

    async def run():
        return [event async for event in run_turn(
            prompt="continue", messages=messages, model=primary, tools=[],
            options=options, cwd=tmp_path, confirm=None, model_route=route,
            compact_context=compact if can_compact else None,
        )]

    events = asyncio.run(run())
    assert requests == (["primary", "fallback"] if can_compact else ["primary"])
    assert compactions == (["fallback"] if can_compact else [])
    assert events[-1].stop_reason == ("stop" if can_compact else "context_limit")
