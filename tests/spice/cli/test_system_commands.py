from __future__ import annotations

import json

from typer.testing import CliRunner

import spice.cli.main as cli_main
import spice.cli.system_commands as system_commands
from spice.diagnostics import DoctorCheck, DoctorReport
from spice.llm.config import SpiceConfig


def _prepare_cli(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(cli_main, "set_process_title", lambda: None)
    monkeypatch.setattr(
        system_commands,
        "load_config",
        lambda: SpiceConfig(memory_enabled=False, subagents_enabled=False),
    )


def test_top_level_help_lists_doctor_and_tools(monkeypatch) -> None:
    _prepare_cli(monkeypatch)

    result = CliRunner().invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "tools" in result.output


def test_tools_json_and_show_use_shared_catalog(monkeypatch) -> None:
    _prepare_cli(monkeypatch)

    listing = CliRunner().invoke(cli_main.app, ["tools", "--json"])
    detail = CliRunner().invoke(cli_main.app, ["tools", "show", "bash", "--json"])

    assert listing.exit_code == 0
    payload = json.loads(listing.output)
    assert any(
        item["name"] == "bash" and item["confirmation"] == "required"
        for item in payload
    )
    assert detail.exit_code == 0
    assert json.loads(detail.output)["parameters"]["type"] == "object"


def test_doctor_json_returns_nonzero_for_errors(monkeypatch) -> None:
    _prepare_cli(monkeypatch)
    report = DoctorReport(
        (
            DoctorCheck("Runtime", "pass", "ready"),
            DoctorCheck("Model", "error", "API key missing"),
        )
    )
    monkeypatch.setattr(system_commands, "build_doctor_report", lambda **kwargs: report)

    result = CliRunner().invoke(cli_main.app, ["doctor", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert payload["errors"] == 1
