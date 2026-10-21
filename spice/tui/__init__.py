"""Lazy entry points for the Textual TUI and temporary legacy fallback."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


class TuiDependencyError(RuntimeError):
    """Raised when the selected TUI frontend is not completely installed."""


def _textual_module() -> Any:
    try:
        return importlib.import_module("spice.tui.textual_app")
    except ModuleNotFoundError as exc:
        if exc.name != "textual" and not (exc.name or "").startswith("textual."):
            raise
        raise TuiDependencyError(
            "Textual is not installed in this Spice environment. "
            "For a uv tool install, run `uv tool upgrade --reinstall spice`; "
            "for the project environment, run `uv sync`."
        ) from exc


def run_textual_tui(**kwargs: Any) -> None:
    """Load and run the Textual frontend only when it is selected."""
    _textual_module().run_textual_tui(**kwargs)


def run_legacy_tui(**kwargs: Any) -> None:
    """Load and run the prompt_toolkit fallback only when it is selected."""
    module = importlib.import_module("spice.tui.app")
    module.run_tui(**kwargs)


def run_tui(
    *,
    provider: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    continue_session: bool = False,
    trace: bool = False,
    trace_file: Path | None = None,
    legacy: bool = False,
) -> None:
    runner = run_legacy_tui if legacy else run_textual_tui
    runner(
        provider=provider,
        model=model,
        session_id=session_id,
        continue_session=continue_session,
        trace=trace,
        trace_file=trace_file,
    )


def __getattr__(name: str) -> Any:
    """Preserve the public class aliases without eagerly loading either UI."""
    if name in {"SpiceTUI", "SpiceTextualApp"}:
        return _textual_module().SpiceTextualApp
    if name == "LegacySpiceTUI":
        return importlib.import_module("spice.tui.app").SpiceTUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LegacySpiceTUI",
    "SpiceTUI",
    "SpiceTextualApp",
    "TuiDependencyError",
    "run_legacy_tui",
    "run_textual_tui",
    "run_tui",
]
