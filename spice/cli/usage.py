"""Shared CLI/TUI rendering for session model usage."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from rich.table import Table

from spice.llm.usage import SessionUsageSummary, aggregate_session_usage


def session_usage_summary(agent_session: Any) -> SessionUsageSummary:
    if agent_session is None or agent_session.session is None:
        return SessionUsageSummary()
    return aggregate_session_usage(agent_session.session_store.entries(agent_session.session_id))


def build_usage_table(summary: SessionUsageSummary) -> Table:
    table = Table("Metric", "Value", title="Session usage")
    table.add_row("Model calls", f"{summary.model_calls:,}")
    table.add_row("Input tokens", f"{summary.input_tokens:,}")
    table.add_row("Cache hit", _cache_hit_text(summary))
    table.add_row("Cache read", f"{summary.cache_read_tokens:,}")
    table.add_row("Cache write", f"{summary.cache_write_tokens:,}")
    table.add_row("Output tokens", f"{summary.output_tokens:,}")
    table.add_row("Total tokens", f"{summary.total_tokens:,}")
    table.add_row("Model time", _duration_text(summary.duration_ms))
    table.add_row("Estimated cost", _cost_text(summary))
    return table


def _cache_hit_text(summary: SessionUsageSummary) -> str:
    rate = summary.cache_hit_rate
    if rate is None:
        return "unavailable"
    text = f"{rate * 100:.1f}%"
    if summary.cache_unavailable_calls:
        text += f" (observed {summary.cache_metrics_calls}/{summary.model_calls} calls)"
    return text


def _cost_text(summary: SessionUsageSummary) -> str:
    if summary.estimated_cost_usd is None:
        return "unavailable"
    amount = _money(summary.estimated_cost_usd)
    if summary.unpriced_calls:
        return f">= {amount} ({summary.unpriced_calls} calls unavailable)"
    return amount


def _money(value: str) -> str:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return "unavailable"
    if amount == 0:
        return "$0.00"
    if amount < Decimal("0.01"):
        return f"${amount:.6f}"
    return f"${amount:.4f}".rstrip("0").rstrip(".")


def _duration_text(milliseconds: int) -> str:
    if milliseconds < 1000:
        return f"{milliseconds}ms"
    seconds = milliseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
