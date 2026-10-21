from __future__ import annotations

import subprocess
import sys

import pytest

import spice.tui as tui


def test_package_import_does_not_load_a_frontend() -> None:
    code = """
import sys
import spice.tui

assert 'spice.tui.app' not in sys.modules
assert 'spice.tui.textual_app' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_textual_dependency_has_repair_instructions(monkeypatch) -> None:
    missing = ModuleNotFoundError("No module named 'textual'", name="textual")

    def fail_import(name: str):
        raise missing

    monkeypatch.setattr(tui.importlib, "import_module", fail_import)

    with pytest.raises(tui.TuiDependencyError, match="uv tool upgrade"):
        tui.run_textual_tui()
