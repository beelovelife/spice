"""Trust decisions for project-provided stdio MCP commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from spice.llm.config import CONFIG_DIR
from spice.mcp.config import McpServerConfig

TRUST_PATH = CONFIG_DIR / "mcp-trust.json"


class McpTrustStore:
    def __init__(self, path: Path = TRUST_PATH) -> None:
        self.path = path

    def is_trusted(self, workspace: Path, server: McpServerConfig) -> bool:
        return self._read().get(self._key(workspace, server)) == server.trust_digest()

    def trust(self, workspace: Path, server: McpServerConfig) -> None:
        data = self._read()
        data[self._key(workspace, server)] = server.trust_digest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        if os.name != "nt":
            os.chmod(self.path, 0o600)

    def _key(self, workspace: Path, server: McpServerConfig) -> str:
        return f"{workspace.resolve()}::{server.name}"

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
