"""Provider-neutral model stream retries."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from spice.llm.types import ModelRetryNotice, StreamError, StreamEvent, TextDelta, ToolCallEvent


@dataclass(frozen=True)
class ModelRetryPolicy:
    enabled: bool = True
    max_attempts: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 10_000
    multiplier: float = 2.0
    honor_retry_after: bool = True

    @classmethod
    def from_settings(cls, settings: dict | None) -> "ModelRetryPolicy":
        values = settings or {}
        return cls(
            enabled=bool(values.get("enabled", True)),
            max_attempts=min(max(int(values.get("maxAttempts", 3)), 1), 5),
            base_delay_ms=min(max(int(values.get("baseDelayMs", 500)), 0), 60_000),
            max_delay_ms=min(max(int(values.get("maxDelayMs", 10_000)), 0), 60_000),
            multiplier=max(float(values.get("multiplier", 2.0)), 1.0),
            honor_retry_after=bool(values.get("honorRetryAfter", True)),
        )


async def stream_with_retry(
    stream_factory: Callable[[], AsyncIterator[StreamEvent]],
    *,
    policy: ModelRetryPolicy,
    provider: str,
    model: str,
) -> AsyncIterator[StreamEvent]:
    max_attempts = policy.max_attempts if policy.enabled else 1
    for attempt in range(1, max_attempts + 1):
        committed = False
        failure: StreamError | None = None
        async for event in stream_factory():
            if isinstance(event, (TextDelta, ToolCallEvent)):
                committed = True
                yield event
            elif isinstance(event, StreamError):
                failure = event
                break
            else:
                yield event

        if failure is None:
            return
        if committed or not failure.retryable or attempt >= max_attempts:
            yield failure
            return

        delay = _retry_delay(policy, attempt, failure)
        yield ModelRetryNotice(
            provider=provider,
            model=model,
            failed_attempt=attempt,
            next_attempt=attempt + 1,
            max_attempts=max_attempts,
            delay_seconds=delay,
            error=failure.error,
        )
        await asyncio.sleep(delay)


def _retry_delay(policy: ModelRetryPolicy, failed_attempt: int, error: StreamError) -> float:
    if policy.honor_retry_after and error.retry_after_seconds is not None:
        return min(max(error.retry_after_seconds, 0.0), 60.0)
    raw_ms = min(
        policy.base_delay_ms * (policy.multiplier ** max(failed_attempt - 1, 0)),
        policy.max_delay_ms,
    )
    return random.uniform(raw_ms / 2, raw_ms) / 1000 if raw_ms > 0 else 0.0

