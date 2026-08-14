"""Normalized model usage, cost estimation, and session aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from spice.llm.models import Model

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int | None = None
    cache_metrics_available: bool = False
    provider_cost_usd: str | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name)))
        total = self.total_tokens
        object.__setattr__(
            self,
            "total_tokens",
            _non_negative_int(total) if total is not None else self.input_tokens + self.output_tokens,
        )
        if self.cache_read_tokens + self.cache_write_tokens > self.input_tokens:
            object.__setattr__(self, "cache_read_tokens", 0)
            object.__setattr__(self, "cache_write_tokens", 0)
            object.__setattr__(self, "cache_metrics_available", False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TokenUsage":
        data = data or {}
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            total_tokens=data.get("total_tokens"),
            cache_metrics_available=bool(data.get("cache_metrics_available", False)),
            provider_cost_usd=_decimal_string(data.get("provider_cost_usd")),
        )


@dataclass(frozen=True)
class ModelUsageRecord:
    provider: str
    model: str
    tokens: TokenUsage
    duration_ms: int
    model_calls: int = 1
    estimated_cost_usd: str | None = None
    cost_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.tokens.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "model_calls": self.model_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_source": self.cost_source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ModelUsageRecord":
        data = data or {}
        return cls(
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            tokens=TokenUsage.from_dict(data),
            duration_ms=_non_negative_int(data.get("duration_ms")),
            model_calls=max(_non_negative_int(data.get("model_calls", 1)), 1),
            estimated_cost_usd=_decimal_string(data.get("estimated_cost_usd")),
            cost_source=str(data.get("cost_source") or "") or None,
        )


@dataclass(frozen=True)
class SessionUsageSummary:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    estimated_cost_usd: str | None = None
    priced_calls: int = 0
    unpriced_calls: int = 0
    cache_observed_input_tokens: int = 0
    cache_metrics_calls: int = 0
    cache_unavailable_calls: int = 0

    @property
    def cache_hit_rate(self) -> float | None:
        if self.cache_observed_input_tokens <= 0:
            return None
        return self.cache_read_tokens / self.cache_observed_input_tokens


def make_usage_record(model: Model, tokens: TokenUsage | None, *, duration_ms: int) -> ModelUsageRecord:
    tokens = tokens or TokenUsage()
    cost, source = estimate_cost(tokens, model)
    return ModelUsageRecord(
        provider=model.provider,
        model=model.id,
        tokens=tokens,
        duration_ms=max(int(duration_ms), 0),
        estimated_cost_usd=cost,
        cost_source=source,
    )


def estimate_cost(tokens: TokenUsage, model: Model) -> tuple[str | None, str | None]:
    if provider_cost := _decimal_string(tokens.provider_cost_usd):
        return provider_cost, "provider"
    pricing = model.pricing
    if pricing is None:
        return None, None
    if tokens.cache_read_tokens and pricing.cache_read_per_million_usd is None:
        return None, None
    if tokens.cache_write_tokens and pricing.cache_write_per_million_usd is None:
        return None, None
    try:
        normal_input = max(tokens.input_tokens - tokens.cache_read_tokens - tokens.cache_write_tokens, 0)
        amount = Decimal(normal_input) * Decimal(pricing.input_per_million_usd)
        amount += Decimal(tokens.output_tokens) * Decimal(pricing.output_per_million_usd)
        if tokens.cache_read_tokens:
            amount += Decimal(tokens.cache_read_tokens) * Decimal(pricing.cache_read_per_million_usd or "0")
        if tokens.cache_write_tokens:
            amount += Decimal(tokens.cache_write_tokens) * Decimal(pricing.cache_write_per_million_usd or "0")
    except (InvalidOperation, ValueError):
        return None, None
    return _format_decimal(amount / MILLION), "model_pricing"


def aggregate_usage_records(records: Iterable[ModelUsageRecord]) -> SessionUsageSummary:
    rows = list(records)
    known_cost = Decimal(0)
    priced_calls = 0
    cache_observed_input = 0
    cache_metrics_calls = 0
    for record in rows:
        if record.estimated_cost_usd is not None:
            try:
                known_cost += Decimal(record.estimated_cost_usd)
                priced_calls += record.model_calls
            except InvalidOperation:
                pass
        if record.tokens.cache_metrics_available:
            cache_observed_input += record.tokens.input_tokens
            cache_metrics_calls += record.model_calls
    calls = sum(record.model_calls for record in rows)
    return SessionUsageSummary(
        model_calls=calls,
        input_tokens=sum(record.tokens.input_tokens for record in rows),
        output_tokens=sum(record.tokens.output_tokens for record in rows),
        cache_read_tokens=sum(record.tokens.cache_read_tokens for record in rows),
        cache_write_tokens=sum(record.tokens.cache_write_tokens for record in rows),
        total_tokens=sum(int(record.tokens.total_tokens or 0) for record in rows),
        duration_ms=sum(record.duration_ms for record in rows),
        estimated_cost_usd=_format_decimal(known_cost) if priced_calls else None,
        priced_calls=priced_calls,
        unpriced_calls=max(calls - priced_calls, 0),
        cache_observed_input_tokens=cache_observed_input,
        cache_metrics_calls=cache_metrics_calls,
        cache_unavailable_calls=max(calls - cache_metrics_calls, 0),
    )


def aggregate_session_usage(entries: Iterable[Any]) -> SessionUsageSummary:
    records: list[ModelUsageRecord] = []
    for entry in entries:
        if getattr(entry, "type", None) != "message":
            continue
        data = getattr(entry, "data", {}) or {}
        message = data.get("message") if isinstance(data, Mapping) else None
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        metadata = message.get("metadata")
        usage = metadata.get("usage") if isinstance(metadata, Mapping) else None
        if isinstance(usage, Mapping):
            records.append(ModelUsageRecord.from_dict(usage))
    return aggregate_usage_records(records)


def normalize_openai_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    input_tokens = _field(usage, "input_tokens", "prompt_tokens")
    output_tokens = _field(usage, "output_tokens", "completion_tokens")
    total_tokens = _optional_field(usage, "total_tokens")
    details = _raw_field(usage, "input_tokens_details", "prompt_tokens_details")
    cached_tokens = _field(details, "cached_tokens") if details is not None else _field(usage, "prompt_cache_hit_tokens")
    cache_miss = _optional_field(usage, "prompt_cache_miss_tokens")
    cache_available = details is not None or _has_field(usage, "prompt_cache_hit_tokens") or cache_miss is not None
    if cache_miss is not None and input_tokens == 0:
        input_tokens = cached_tokens + cache_miss
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cached_tokens,
        total_tokens=total_tokens,
        cache_metrics_available=cache_available,
        provider_cost_usd=_raw_field(usage, "cost", "cost_usd"),
    )


def normalize_anthropic_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    normal_input = _field(usage, "input_tokens")
    cache_read = _field(usage, "cache_read_input_tokens")
    cache_write = _field(usage, "cache_creation_input_tokens")
    output = _field(usage, "output_tokens")
    cache_available = _has_field(usage, "cache_read_input_tokens") or _has_field(usage, "cache_creation_input_tokens")
    total_input = normal_input + cache_read + cache_write
    return TokenUsage(
        input_tokens=total_input,
        output_tokens=output,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        total_tokens=total_input + output,
        cache_metrics_available=cache_available,
    )


def normalize_gemini_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    input_tokens = _field(usage, "prompt_token_count")
    output_tokens = _field(usage, "candidates_token_count")
    cached_tokens = _field(usage, "cached_content_token_count")
    total_tokens = _optional_field(usage, "total_token_count")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cached_tokens,
        total_tokens=total_tokens,
        cache_metrics_available=_has_field(usage, "cached_content_token_count"),
    )


def _raw_field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _has_field(value: Any, name: str) -> bool:
    return (isinstance(value, Mapping) and name in value) or hasattr(value, name)


def _field(value: Any, *names: str) -> int:
    return _non_negative_int(_raw_field(value, *names))


def _optional_field(value: Any, *names: str) -> int | None:
    raw = _raw_field(value, *names)
    return None if raw is None else _non_negative_int(raw)


def _non_negative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _decimal_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        return _format_decimal(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return None


def _format_decimal(value: Decimal) -> str:
    text = format(value.quantize(Decimal("0.000000000001")), "f").rstrip("0").rstrip(".")
    return text or "0"
