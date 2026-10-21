"""Global test isolation from the user's real Spice configuration and state."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TEST_CONFIG_DIR = Path(tempfile.mkdtemp(prefix="spice-tests-"))

# This module is loaded before test modules import spice.llm.config, whose paths
# are intentionally process-level constants. Always override a caller's real
# config directory so tests cannot read or mutate ~/.spice.
os.environ["SPICE_CONFIG_DIR"] = str(_TEST_CONFIG_DIR)


def pytest_sessionfinish() -> None:
    shutil.rmtree(_TEST_CONFIG_DIR, ignore_errors=True)
