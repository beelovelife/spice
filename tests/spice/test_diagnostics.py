from __future__ import annotations

from spice import diagnostics
from spice.llm.config import SpiceConfig


def test_doctor_report_is_ready_with_valid_local_configuration(
    tmp_path, monkeypatch
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        diagnostics, "get_api_key", lambda provider, env_names=None: "configured"
    )
    monkeypatch.setattr(diagnostics, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        diagnostics, "log_path", lambda: tmp_path / "logs" / "spice.log"
    )

    report = diagnostics.build_doctor_report(
        cwd=tmp_path,
        config=SpiceConfig(),
        settings_path=settings,
    )

    assert report.ready is True
    assert report.error_count == 0
    assert {check.name for check in report.checks} >= {
        "Runtime",
        "TUI",
        "Config",
        "Model",
        "Storage",
        "Tools",
    }


def test_doctor_report_marks_missing_model_key_as_error(tmp_path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        diagnostics, "get_api_key", lambda provider, env_names=None: None
    )
    monkeypatch.setattr(diagnostics, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(diagnostics, "log_path", lambda: tmp_path / "spice.log")

    report = diagnostics.build_doctor_report(
        cwd=tmp_path,
        config=SpiceConfig(),
        settings_path=settings,
    )

    model_check = next(check for check in report.checks if check.name == "Model")
    assert model_check.status == "error"
    assert "API key missing" in model_check.summary
    assert report.ready is False


def test_doctor_report_marks_missing_textual_as_error(tmp_path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        diagnostics, "get_api_key", lambda provider, env_names=None: "configured"
    )
    monkeypatch.setattr(diagnostics, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(diagnostics, "log_path", lambda: tmp_path / "spice.log")
    monkeypatch.setattr(diagnostics.util, "find_spec", lambda name: None)

    report = diagnostics.build_doctor_report(
        cwd=tmp_path,
        config=SpiceConfig(),
        settings_path=settings,
    )

    tui_check = next(check for check in report.checks if check.name == "TUI")
    assert tui_check.status == "error"
    assert "Textual is not installed" in tui_check.summary
    assert report.ready is False


def test_doctor_report_marks_invalid_settings_json_as_error(
    tmp_path, monkeypatch
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        diagnostics, "get_api_key", lambda provider, env_names=None: "configured"
    )
    monkeypatch.setattr(diagnostics, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(diagnostics, "log_path", lambda: tmp_path / "spice.log")

    report = diagnostics.build_doctor_report(
        cwd=tmp_path,
        config=SpiceConfig(),
        settings_path=settings,
    )

    config_check = next(check for check in report.checks if check.name == "Config")
    assert config_check.status == "error"
    assert "invalid JSON" in config_check.summary
