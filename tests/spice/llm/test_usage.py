from __future__ import annotations

import asyncio

import pytest

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
    track_model_stream,
    usage_scope,
    usage_token_text,
)
from spice.llm.types import Done, StreamError, TextDelta


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


def test_missing_usage_is_not_priced_as_zero():
    model = Model(id="priced", provider="test", pricing=ModelPricing("1", "2"))
    record = make_usage_record(model, None, duration_ms=1)
    assert record.estimated_cost_usd is None
    assert not record.usage_available
    assert ModelUsageRecord.from_dict(record.to_dict()) == record
    summary = aggregate_usage_records([record])
    assert summary.usage_unavailable_calls == 1
    assert summary.unpriced_calls == 1
    assert "未知" in usage_token_text(summary, summary.total_tokens)
    known = make_usage_record(model, TokenUsage(10, 5), duration_ms=1)
    summary = aggregate_usage_records([record, known])
    assert usage_token_text(summary, summary.total_tokens).startswith(">= 15")


def test_usage_ledger_deduplicates_message_and_custom_record():
    record = make_usage_record(Model(id="test", provider="test"), TokenUsage(10, 2), duration_ms=1)
    entries = [
        SessionEntry("ledger", "custom", "", None, {"customType": "model_usage", "usage": record.to_dict()}),
        SessionEntry("message", "message", "", "ledger", {
            "message": {"role": "assistant", "metadata": {"usage": record.to_dict()}}}),
    ]
    for ordered in (entries, list(reversed(entries))):
        assert aggregate_session_usage(ordered).model_calls == 1


@pytest.mark.parametrize("ending", ["done", "missing_usage", "error", "exception", "incomplete", "cancelled"])
def test_request_tracker_records_each_outcome_once(ending):
    recorded = []
    model = Model(id="test", provider="test")

    async def stream():
        yield TextDelta("partial")
        if ending == "done":
            yield Done("stop", TokenUsage(10, 3))
        elif ending == "missing_usage":
            yield Done("stop")
        elif ending == "error":
            yield StreamError("failed")
        elif ending == "exception":
            raise RuntimeError("failed")
        elif ending == "cancelled":
            raise asyncio.CancelledError()

    async def run():
        with usage_scope(recorded.append):
            try:
                return [event async for event in track_model_stream(model, stream, source="compaction")]
            except (RuntimeError, asyncio.CancelledError):
                return []

    events = asyncio.run(run())
    assert len(recorded) == 1
    assert recorded[0].source == "compaction"
    assert recorded[0].usage_available == (ending == "done")
    if ending in {"done", "missing_usage"}:
        assert events[-1].usage_record.call_id == recorded[0].call_id
    if ending == "cancelled":
        assert recorded[0].status == "cancelled"


@pytest.mark.parametrize("finish", [
    "incomplete", "length", "max_tokens", "max_output_tokens", "content_filter", "error",
])
def test_incomplete_terminal_response_keeps_usage_without_claiming_completion(finish):
    recorded = []

    async def stream():
        yield Done(finish, TokenUsage(100, 16000))

    async def run():
        with usage_scope(recorded.append):
            return [event async for event in track_model_stream(
                Model(id="test", provider="test"), stream,
            )]

    events = asyncio.run(run())
    assert len(recorded) == 1
    assert recorded[0].status == "incomplete"
    assert recorded[0].usage_available
    assert recorded[0].tokens.output_tokens == 16000
    assert events[-1].usage_record.call_id == recorded[0].call_id


def test_usage_observer_isolates_concurrent_sessions_and_inherits_child_tasks():
    first, second = [], []
    model = Model(id="test", provider="test")

    async def stream():
        await asyncio.sleep(0)
        yield Done("stop", TokenUsage(1, 1))

    async def call():
        return [event async for event in track_model_stream(model, stream, source="subagent")]

    async def session(sink):
        with usage_scope(sink.append):
            await asyncio.gather(call(), call())

    async def run():
        await asyncio.gather(session(first), session(second))
        await call()

    asyncio.run(run())
    assert len(first) == len(second) == 2
    assert len({r.call_id for r in first + second}) == 4
