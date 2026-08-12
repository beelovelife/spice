from __future__ import annotations

import asyncio
from types import SimpleNamespace


from spice.mcp.config import McpConfigLoadResult, McpServerConfig
from spice.mcp.manager import McpManager


def test_server_failure_is_isolated_and_tools_are_adapted(tmp_path, monkeypatch) -> None:
    configs = McpConfigLoadResult(
        {
            "good": McpServerConfig("good", "http", url="https://good.example/mcp"),
            "bad": McpServerConfig("bad", "http", url="https://bad.example/mcp"),
        }
    )
    monkeypatch.setattr("spice.mcp.manager.load_mcp_config", lambda _cwd: configs)

    async def connect(connection):
        if connection.config.name == "bad":
            raise RuntimeError("offline")
        return [
            SimpleNamespace(
                name="lookup",
                description="Lookup",
                inputSchema={"type": "object", "properties": {}},
                annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
            )
        ]

    monkeypatch.setattr("spice.mcp.manager.McpConnection.connect", connect)
    manager = McpManager(cwd=tmp_path)

    asyncio.run(manager.ensure_connected())

    assert manager.statuses["good"].state == "connected"
    assert manager.statuses["bad"].state == "failed"
    assert [tool.name for tool in manager.tools()] == ["mcp__good__lookup"]
    assert [tool.name for tool in manager.read_only_tools()] == ["mcp__good__lookup"]
