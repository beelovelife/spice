from __future__ import annotations

import os
from pathlib import Path

from spice.llm.config import CONFIG_DIR, SpiceConfig
from spice.storage.factory import sqlite_path


def test_suite_uses_dedicated_config_directory() -> None:
    assert CONFIG_DIR == Path(os.environ["SPICE_CONFIG_DIR"])
    assert CONFIG_DIR != Path.home() / ".spice"


def test_default_sqlite_path_follows_config_directory() -> None:
    assert (
        sqlite_path(SpiceConfig(storage={"backend": "sqlite"}))
        == CONFIG_DIR / "spice.db"
    )
