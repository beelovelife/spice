"""Logging setup for Spice."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from spice.llm.config import CONFIG_DIR

LOG_DIR = CONFIG_DIR / "logs"
LOG_PATH = LOG_DIR / "spice.log"
DEBUG_LOG_PATH = CONFIG_DIR / "spice.debug.log"

_CONFIGURED = False
_LOG_PATH: Path | None = None


def configure_logging(*, debug: bool = False, log_path: Path | None = None) -> Path:
    """Configure process-wide file logging once and return the active log path."""
    global _CONFIGURED, _LOG_PATH
    path = log_path or (DEBUG_LOG_PATH if debug else LOG_PATH)
    if _CONFIGURED:
        return _LOG_PATH or path

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger("spice")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _LOG_PATH = path
    _CONFIGURED = True
    return path


def get_logger(name: str) -> logging.Logger:
    if name == "spice" or name.startswith("spice."):
        return logging.getLogger(name)
    return logging.getLogger(f"spice.{name}")


def log_path() -> Path:
    return _LOG_PATH or LOG_PATH
