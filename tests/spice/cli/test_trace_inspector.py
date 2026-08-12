from __future__ import annotations

import json

from rich.console import Console
from typer.testing import CliRunner

import spice.cli.main as cli_main
from spice.cli.trace_inspector import TraceInspectionError, build_trace_steps, render_trace, validate_trace


def _trace() -> dict:
    return {
        "format": "spice-trace-1",
        "spice_version": "0.2.0",
        "session_id": "session-1",
        "updated_at": "2026-07-01T00:00:00Z",
        "model": {"provider": "openai", "id": "gpt-test"},
        "messages": [],
        "events": [
            {
                "type": "assistant_message",
                "text": "inspect files",
                "tool_calls": [{"id": "tc1", "name": "read_file", "arguments": {"path": "README.md"}}],
            },
            {"type": "tool_start", "tool_call_id": "tc1", "tool_name": "read_file"},
            {
                "type": "tool_end",
                "tool_call_id": "tc1",
                "tool_name": "read_file",
                "result": {"content": "contents", "is_error": False},
            },
            {"type": "assistant_message", "text": "done", "tool_calls": []},
            {"type": "turn_end", "stop_reason": "stop"},
        ],
        "summary": {
            "status": "completed",
            "stop_reason": "stop",
            "usage": {"model_calls": 2, "total_tokens": 123, "estimated_cost_usd": "0.01", "unpriced_calls": 0},
        },
    }


def test_trace_steps_group_following_tool_events() -> None:
    steps = build_trace_steps(validate_trace(_trace()))

    assert len(steps) == 2
    assert steps[0].following_events[-1]["type"] == "tool_end"
    assert steps[1].following_events[-1]["type"] == "turn_end"


def test_render_trace_can_select_one_step() -> None:
    console = Console(record=True, width=120)

    render_trace(_trace(), console, step=2, show_events=True)
    output = console.export_text()

    assert "Spice trace" in output
    assert "Step 2" in output
    assert "Step 1" not in output
    assert "Event timeline" in output


def test_validate_trace_rejects_unknown_format() -> None:
    data = _trace()
    data["format"] = "future-trace"

    try:
        validate_trace(data)
    except TraceInspectionError as exc:
        assert "Unsupported trace format" in str(exc)
    else:
        raise AssertionError("expected trace validation error")


def test_trace_inspect_cli_reads_trace_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "run.trace.json"
    path.write_text(json.dumps(_trace()), encoding="utf-8")
    monkeypatch.setattr(cli_main, "set_process_title", lambda: None)

    result = CliRunner().invoke(cli_main.app, ["trace", "inspect", str(path), "--step", "1"])

    assert result.exit_code == 0
    assert "Step 1" in result.output


def test_trace_inspect_cli_reports_invalid_json(tmp_path, monkeypatch) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(cli_main, "set_process_title", lambda: None)

    result = CliRunner().invoke(cli_main.app, ["trace", "inspect", str(path)])

    assert result.exit_code == 1
    assert "Error:" in result.output
