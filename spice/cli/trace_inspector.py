"""Read-only Rich inspector for Spice runtime trace snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spice.tools.base import truncate_head_tail

SUPPORTED_TRACE_FORMATS = {"spice-trace-1"}
DEFAULT_PREVIEW_CHARS = 2_000


class TraceInspectionError(ValueError):
    """A user-actionable trace input error."""


@dataclass(frozen=True)
class TraceStep:
    index: int
    assistant: dict[str, Any]
    following_events: list[dict[str, Any]]


def validate_trace(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TraceInspectionError("Trace root must be a JSON object.")
    trace_format = data.get("format")
    if trace_format not in SUPPORTED_TRACE_FORMATS:
        raise TraceInspectionError(f"Unsupported trace format: {trace_format or '<missing>'}.")
    for field, expected in (("messages", list), ("events", list), ("summary", dict)):
        if not isinstance(data.get(field), expected):
            raise TraceInspectionError(f"Trace field '{field}' must be {expected.__name__}.")
    return data


def build_trace_steps(data: dict[str, Any]) -> list[TraceStep]:
    steps: list[TraceStep] = []
    current: TraceStep | None = None
    for raw_event in data.get("events", []):
        if not isinstance(raw_event, dict):
            continue
        if raw_event.get("type") == "assistant_message":
            current = TraceStep(len(steps) + 1, raw_event, [])
            steps.append(current)
        elif current is not None:
            current.following_events.append(raw_event)
    return steps


def render_trace(
    data: dict[str, Any],
    console: Console,
    *,
    step: int | None = None,
    show_events: bool = False,
    full: bool = False,
) -> None:
    data = validate_trace(data)
    steps = build_trace_steps(data)
    if step is not None and not 1 <= step <= len(steps):
        raise TraceInspectionError(f"Step must be between 1 and {len(steps)}.")

    _render_overview(data, console, len(steps))
    selected = [steps[step - 1]] if step is not None else steps
    if not selected:
        console.print("[yellow]No assistant steps recorded.[/yellow]")
    for item in selected:
        _render_step(item, console, full=full)
    if show_events:
        _render_events(data["events"], console)


def _render_overview(data: dict[str, Any], console: Console, steps: int) -> None:
    raw_model = data.get("model")
    model = raw_model if isinstance(raw_model, dict) else {}
    summary = data["summary"]
    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
    table = Table("Field", "Value", title="Spice trace")
    table.add_row("Format", str(data.get("format") or ""))
    table.add_row("Spice version", str(data.get("spice_version") or "unavailable"))
    table.add_row("Status", str(summary.get("status") or "unknown"))
    table.add_row("Session", str(data.get("session_id") or "(none)"))
    table.add_row("Model", f"{model.get('provider', '')}/{model.get('id', '')}".strip("/"))
    table.add_row("Steps", str(steps))
    table.add_row("Model calls", str(usage.get("model_calls", 0)))
    table.add_row("Tokens", str(usage.get("total_tokens", 0)))
    table.add_row("Estimated cost", _cost_text(usage))
    table.add_row("Stop reason", str(summary.get("stop_reason") or "(none)"))
    table.add_row("Updated", str(data.get("updated_at") or ""))
    console.print(table)


def _render_step(step: TraceStep, console: Console, *, full: bool) -> None:
    assistant = step.assistant
    text = str(assistant.get("text") or "(no assistant text)")
    console.print(Panel(text if full else truncate_head_tail(text, DEFAULT_PREVIEW_CHARS), title=f"Step {step.index} · assistant"))
    tool_calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
    if tool_calls:
        calls = Table("Tool", "ID", "Arguments", title=f"Step {step.index} · tool calls")
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            args = str(call.get("arguments") or {})
            calls.add_row(str(call.get("name") or ""), str(call.get("id") or ""), args if full else truncate_head_tail(args, 500))
        console.print(calls)
    for event in step.following_events:
        event_type = event.get("type")
        if event_type == "tool_end":
            raw_result = event.get("result")
            result = raw_result if isinstance(raw_result, dict) else {}
            content = str(result.get("content") or "(no output)")
            title = f"{event.get('tool_name', 'tool')} · {'error' if result.get('is_error') else 'result'}"
            console.print(Panel(content if full else truncate_head_tail(content, DEFAULT_PREVIEW_CHARS), title=title))
        elif event_type in {"model_retry", "model_fallback", "agent_error", "turn_end"}:
            console.print(f"[dim]{event_type}:[/dim] {_event_detail(event)}")


def _render_events(events: list[Any], console: Console) -> None:
    table = Table("#", "Time", "Event", "Detail", title="Event timeline")
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        table.add_row(str(index), str(event.get("timestamp") or ""), str(event.get("type") or "unknown"), _event_detail(event))
    console.print(table)


def _event_detail(event: dict[str, Any]) -> str:
    event_type = event.get("type")
    if event_type in {"tool_start", "tool_end"}:
        return f"{event.get('tool_name', '')} {event.get('tool_call_id', '')}".strip()
    if event_type == "model_retry":
        return f"{event.get('provider', '')}/{event.get('model', '')} attempt {event.get('next_attempt', '?')}"
    if event_type == "model_fallback":
        return f"{event.get('from_model', '')} -> {event.get('to_model', '')}"
    return str(event.get("error") or event.get("stop_reason") or event.get("session_id") or "")


def _cost_text(usage: dict[str, Any]) -> str:
    value = usage.get("estimated_cost_usd")
    if value is None:
        return "unavailable"
    prefix = ">= " if usage.get("unpriced_calls") else ""
    return f"{prefix}${value}"

