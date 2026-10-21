"""Per-turn model retry and fallback routing."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from spice.async_utils import closing_iterator

from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy, stream_with_retry
from spice.llm.stream import stream_model
from spice.llm.usage import track_model_stream
from spice.llm.types import (
    Done,
    ModelFallbackNotice,
    ModelRequestOptions,
    StreamError,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    ToolSchema,
)

StreamFactory = Callable[
    [Model, list[Message], list[ToolSchema], ModelRequestOptions],
    AsyncIterator[StreamEvent],
]


@dataclass(frozen=True)
class ModelCandidate:
    profile_key: str
    model: Model
    options: ModelRequestOptions


BeforeModelRequest = Callable[[ModelCandidate, list[Message], list[ToolSchema]], Awaitable[None]]


class ModelRoute:
    def __init__(
        self,
        candidates: list[ModelCandidate],
        *,
        retry_policy: ModelRetryPolicy,
        fallback_enabled: bool,
        stream_factory: StreamFactory = stream_model,
        usage_source: str = "main",
    ) -> None:
        if not candidates:
            raise ValueError("ModelRoute requires at least one candidate.")
        self.candidates = candidates
        self.retry_policy = retry_policy
        self.fallback_enabled = fallback_enabled
        self.stream_factory = stream_factory
        self.usage_source = usage_source
        self.active_index: int | None = None
        self.committed = False

    @property
    def actual(self) -> ModelCandidate:
        return self.candidates[self.active_index or 0]

    async def stream(
        self, messages: list[Message], tools: list[ToolSchema], *,
        before_request: BeforeModelRequest | None = None,
    ) -> AsyncIterator[StreamEvent]:
        indexes = [self.active_index] if self.active_index is not None else list(range(len(self.candidates)))
        for position, candidate_index in enumerate(indexes):
            assert candidate_index is not None
            candidate = self.candidates[candidate_index]
            if before_request is not None:
                await before_request(candidate, messages, tools)
            failure: StreamError | None = None

            async def create_stream() -> AsyncIterator[StreamEvent]:
                async with closing_iterator(track_model_stream(
                    candidate.model,
                    lambda: self.stream_factory(candidate.model, messages, tools, candidate.options),
                    source=self.usage_source,
                )) as stream:
                    async for item in stream:
                        yield item

            async with closing_iterator(stream_with_retry(
                create_stream,
                policy=self.retry_policy,
                provider=candidate.model.provider,
                model=candidate.model.id,
            )) as stream:
                async for event in stream:
                    if isinstance(event, (TextDelta, ToolCallEvent)):
                        self.active_index = candidate_index
                        self.committed = True
                        yield event
                    elif isinstance(event, Done):
                        self.active_index = candidate_index
                        yield event
                        return
                    elif isinstance(event, StreamError):
                        failure = event
                    else:
                        yield event

            if failure is None:
                return
            can_fallback = (
                self.fallback_enabled
                and self.active_index is None
                and not self.committed
                and failure.retryable
                and position + 1 < len(indexes)
            )
            if not can_fallback:
                yield failure
                return

            next_index = indexes[position + 1]
            assert next_index is not None
            next_candidate = self.candidates[next_index]
            yield ModelFallbackNotice(
                from_profile=candidate.profile_key,
                from_provider=candidate.model.provider,
                from_model=candidate.model.id,
                to_profile=next_candidate.profile_key,
                to_provider=next_candidate.model.provider,
                to_model=next_candidate.model.id,
                reason=failure.kind,
                fallback_index=position,
                fallback_count=len(indexes) - 1,
            )


def build_model_route(
    primary: Model,
    *,
    routing_settings: dict,
    resolve_model: Callable[[str], Model | None],
    options_factory: Callable[[Model], ModelRequestOptions],
    usage_source: str = "main",
) -> ModelRoute:
    retry_policy = ModelRetryPolicy.from_settings(routing_settings.get("retry"))
    raw_fallback = routing_settings.get("fallback")
    fallback = raw_fallback if isinstance(raw_fallback, dict) else {}
    fallback_enabled = bool(fallback.get("enabled", False))
    models = [primary]
    seen = {(primary.provider, primary.id)}
    if fallback_enabled:
        for profile in fallback.get("profiles") or []:
            candidate = resolve_model(str(profile))
            if candidate is None or (candidate.provider, candidate.id) in seen:
                continue
            seen.add((candidate.provider, candidate.id))
            models.append(candidate)
    candidates = [
        ModelCandidate(model.profile_key or model.id, model, options_factory(model))
        for model in models
    ]
    return ModelRoute(candidates, retry_policy=retry_policy, fallback_enabled=fallback_enabled, usage_source=usage_source)
