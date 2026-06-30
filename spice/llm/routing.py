"""Per-turn model retry and fallback routing."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy, stream_with_retry
from spice.llm.stream import stream_model
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


class ModelRoute:
    def __init__(
        self,
        candidates: list[ModelCandidate],
        *,
        retry_policy: ModelRetryPolicy,
        fallback_enabled: bool,
        stream_factory: StreamFactory = stream_model,
    ) -> None:
        if not candidates:
            raise ValueError("ModelRoute requires at least one candidate.")
        self.candidates = candidates
        self.retry_policy = retry_policy
        self.fallback_enabled = fallback_enabled
        self.stream_factory = stream_factory
        self.active_index: int | None = None
        self.committed = False

    @property
    def actual(self) -> ModelCandidate:
        return self.candidates[self.active_index or 0]

    async def stream(self, messages: list[Message], tools: list[ToolSchema]) -> AsyncIterator[StreamEvent]:
        indexes = [self.active_index] if self.active_index is not None else list(range(len(self.candidates)))
        for position, candidate_index in enumerate(indexes):
            assert candidate_index is not None
            candidate = self.candidates[candidate_index]
            failure: StreamError | None = None

            async def create_stream() -> AsyncIterator[StreamEvent]:
                async for item in self.stream_factory(candidate.model, messages, tools, candidate.options):
                    yield item

            async for event in stream_with_retry(
                create_stream,
                policy=self.retry_policy,
                provider=candidate.model.provider,
                model=candidate.model.id,
            ):
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
) -> ModelRoute:
    retry_policy = ModelRetryPolicy.from_settings(routing_settings.get("retry"))
    fallback = routing_settings.get("fallback") if isinstance(routing_settings.get("fallback"), dict) else {}
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
    return ModelRoute(candidates, retry_policy=retry_policy, fallback_enabled=fallback_enabled)
