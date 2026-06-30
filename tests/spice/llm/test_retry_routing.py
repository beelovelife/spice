from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps

import pytest

from spice.llm.error_safety import stream_error_from_exception
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy, stream_with_retry
from spice.llm.routing import ModelCandidate, ModelRoute
from spice.llm.types import (
    Done,
    ModelFallbackNotice,
    ModelRequestOptions,
    ModelRetryNotice,
    StreamError,
    TextDelta,
    ToolCallEvent,
)


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@async_test
async def test_retry_recovers_before_first_observable_event() -> None:
    attempts = 0

    async def stream():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield StreamError("temporary", kind="server", retryable=True)
        else:
            yield TextDelta("ok")
            yield Done("stop")

    events = [
        event
        async for event in stream_with_retry(
            stream,
            policy=ModelRetryPolicy(max_attempts=2, base_delay_ms=0),
            provider="test",
            model="primary",
        )
    ]

    assert attempts == 2
    assert [type(event) for event in events] == [ModelRetryNotice, TextDelta, Done]


@pytest.mark.parametrize("observable", [TextDelta("partial"), ToolCallEvent("tc1", "read_file", {})])
@async_test
async def test_retry_never_replays_after_observable_event(observable) -> None:
    attempts = 0

    async def stream():
        nonlocal attempts
        attempts += 1
        yield observable
        yield StreamError("disconnected", kind="network", retryable=True)

    events = [
        event
        async for event in stream_with_retry(
            stream,
            policy=ModelRetryPolicy(max_attempts=3, base_delay_ms=0),
            provider="test",
            model="primary",
        )
    ]

    assert attempts == 1
    assert events == [observable, StreamError("disconnected", kind="network", retryable=True)]


@async_test
async def test_retry_after_takes_priority(monkeypatch) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    attempts = 0

    async def stream():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield StreamError("rate limited", kind="rate_limit", retryable=True, retry_after_seconds=2.5)
        else:
            yield Done("stop")

    monkeypatch.setattr("spice.llm.retry.asyncio.sleep", fake_sleep)
    events = [
        event
        async for event in stream_with_retry(
            stream,
            policy=ModelRetryPolicy(max_attempts=2, base_delay_ms=10_000),
            provider="test",
            model="primary",
        )
    ]

    assert delays == [2.5]
    assert isinstance(events[0], ModelRetryNotice)
    assert events[0].delay_seconds == 2.5


@async_test
async def test_fallback_runs_after_retry_exhaustion_and_sticks_for_turn() -> None:
    primary = Model(id="primary", provider="one", profile_key="primary")
    fallback = Model(id="fallback", provider="two", profile_key="fallback")
    calls: list[str] = []

    async def stream_factory(model, messages, tools, options):
        calls.append(model.id)
        if model.id == "primary":
            yield StreamError("unavailable", kind="server", retryable=True)
        elif calls.count("fallback") == 1:
            yield ToolCallEvent("tc1", "read_file", {"path": "README.md"})
            yield Done("tool_calls")
        else:
            yield TextDelta("finished")
            yield Done("stop")

    route = ModelRoute(
        [
            ModelCandidate("primary", primary, ModelRequestOptions()),
            ModelCandidate("fallback", fallback, ModelRequestOptions()),
        ],
        retry_policy=ModelRetryPolicy(max_attempts=2, base_delay_ms=0),
        fallback_enabled=True,
        stream_factory=stream_factory,
    )

    first = [event async for event in route.stream([], [])]
    second = [event async for event in route.stream([], [])]

    assert calls == ["primary", "primary", "fallback", "fallback"]
    assert sum(isinstance(event, ModelFallbackNotice) for event in first) == 1
    assert any(isinstance(event, ToolCallEvent) for event in first)
    assert any(isinstance(event, TextDelta) for event in second)
    assert route.actual.model is fallback


@dataclass
class _Response:
    status_code: int
    headers: dict[str, str]


class _HttpError(Exception):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.response = _Response(status, headers or {})


def test_provider_error_classification_is_structured() -> None:
    retryable = stream_error_from_exception(
        _HttpError(429, {"Retry-After": "3"}), prefix="Request failed", provider="openai", model="m"
    )
    fatal = stream_error_from_exception(
        _HttpError(401), prefix="Request failed", provider="openai", model="m"
    )
    invalid = stream_error_from_exception(
        _HttpError(400), prefix="Request failed", provider="openai", model="m"
    )

    assert (retryable.kind, retryable.retryable, retryable.retry_after_seconds) == ("rate_limit", True, 3.0)
    assert (fatal.kind, fatal.retryable) == ("authentication", False)
    assert (invalid.kind, invalid.retryable) == ("invalid_request", False)
