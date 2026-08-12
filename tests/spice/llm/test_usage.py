from __future__ import annotations

from spice.agent.sessions import SessionEntry
from spice.llm.models import Model, ModelPricing
from spice.llm.usage import (
    ModelUsageRecord,
    TokenUsage,
    aggregate_session_usage,
    aggregate_usage_records,
    make_usage_record,
    normalize_anthropic_usage,
    normalize_gemini_usage,
    normalize_openai_usage,
)


def test_provider_usage_is_normalized() -> None:
    openai = normalize_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 70,
            "prompt_cache_miss_tokens": 30,
        }
    )
    anthropic = normalize_anthropic_usage(
        {
            "input_tokens": 10,
            "cache_read_input_tokens": 60,
            "cache_creation_input_tokens": 30,
            "output_tokens": 5,
        }
    )
    gemini = normalize_gemini_usage(
        {
            "prompt_token_count": 80,
            "candidates_token_count": 12,
            "cached_content_token_count": 50,
            "total_token_count": 92,
        }
    )

    assert openai == TokenUsage(100, 20, 70, total_tokens=120, cache_metrics_available=True)
    assert anthropic == TokenUsage(100, 5, 60, 30, 105, True)
    assert gemini == TokenUsage(80, 12, 50, total_tokens=92, cache_metrics_available=True)


def test_invalid_cache_breakdown_is_discarded() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=2, cache_read_tokens=20, cache_metrics_available=True)

    assert usage.cache_read_tokens == 0
    assert usage.cache_metrics_available is False


def test_model_pricing_uses_normal_and_cached_input_rates() -> None:
    model = Model(
        id="priced",
        provider="test",
        pricing=ModelPricing("1", "2", cache_read_per_million_usd="0.1", cache_write_per_million_usd="1.25"),
    )
    record = make_usage_record(
        model,
        TokenUsage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=400_000, cache_write_tokens=100_000),
        duration_ms=25,
    )

    assert record.estimated_cost_usd == "0.865"
    assert record.cost_source == "model_pricing"


def test_aggregate_reports_partial_cost_and_cache_coverage() -> None:
    records = [
        ModelUsageRecord("a", "one", TokenUsage(100, 10, 60, cache_metrics_available=True), 10, estimated_cost_usd="0.01"),
        ModelUsageRecord("b", "two", TokenUsage(20, 5), 20),
    ]

    summary = aggregate_usage_records(records)

    assert summary.model_calls == 2
    assert summary.input_tokens == 120
    assert summary.estimated_cost_usd == "0.01"
    assert summary.priced_calls == 1
    assert summary.unpriced_calls == 1
    assert summary.cache_hit_rate == 0.6
    assert summary.cache_unavailable_calls == 1


def test_session_usage_counts_all_branches() -> None:
    def assistant(entry_id: str, parent_id: str, cost: str) -> SessionEntry:
        usage = ModelUsageRecord(
            "test", "model", TokenUsage(input_tokens=10, output_tokens=2), 5, estimated_cost_usd=cost
        )
        return SessionEntry(
            id=entry_id,
            type="message",
            timestamp="",
            parent_id=parent_id,
            data={"message": {"role": "assistant", "content": "ok", "metadata": {"usage": usage.to_dict()}}},
        )

    entries = [
        SessionEntry("user", "message", "", None, {"message": {"role": "user", "content": "hi"}}),
        assistant("old-branch", "user", "0.01"),
        assistant("active-branch", "user", "0.02"),
        SessionEntry("leaf", "leaf", "", None, {"targetId": "active-branch"}),
    ]

    summary = aggregate_session_usage(entries)

    assert summary.model_calls == 2
    assert summary.input_tokens == 20
    assert summary.estimated_cost_usd == "0.03"
