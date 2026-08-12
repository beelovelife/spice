from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from spice.mcp.config import McpServerConfig
from spice.mcp.connection import McpConnection


def test_stdio_connection_discovers_calls_and_closes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("spice.mcp.connection.CONFIG_DIR", tmp_path / "config")
    fixture = Path(__file__).with_name("fixture_server.py")
    connection = McpConnection(
        McpServerConfig(
            name="fixture",
            transport="stdio",
            command=sys.executable,
            args=(str(fixture),),
            connect_timeout=10,
            tool_timeout=10,
        ),
        cwd=tmp_path,
    )

    async def exercise() -> None:
        tools = await connection.connect()
        assert [tool.name for tool in tools] == ["echo"]
        result = await connection.call_tool("echo", {"value": "hello"})
        assert result.content[0].text == "hello"
        await connection.close()
        assert connection.session is None

    asyncio.run(exercise())


def test_stdio_connection_can_close_from_session_owner_task(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("spice.mcp.connection.CONFIG_DIR", tmp_path / "config")
    fixture = Path(__file__).with_name("fixture_server.py")
    connection = McpConnection(
        McpServerConfig("fixture", "stdio", command=sys.executable, args=(str(fixture),)),
        cwd=tmp_path,
    )

    async def exercise() -> None:
        await asyncio.create_task(connection.connect())
        await connection.close()

    asyncio.run(exercise())


def test_failed_tool_call_is_not_replayed_after_connection_failure(tmp_path, monkeypatch) -> None:
    connection = McpConnection(McpServerConfig("fixture", "stdio", command="fixture"), cwd=tmp_path)
    request_count = 0
    cleanup_count = 0

    async def connect():
        return []

    async def request(_operation, _payload):
        nonlocal request_count
        request_count += 1
        raise RuntimeError("connection dropped after dispatch")

    async def wait_for_owner_exit():
        nonlocal cleanup_count
        cleanup_count += 1

    monkeypatch.setattr(connection, "connect", connect)
    monkeypatch.setattr(connection, "_request", request)
    monkeypatch.setattr(connection, "_wait_for_owner_exit", wait_for_owner_exit)

    async def exercise() -> None:
        try:
            await connection.call_tool("write", {"value": "once"})
        except RuntimeError as exc:
            assert "connection dropped" in str(exc)
        else:
            raise AssertionError("Expected the failed MCP call to be reported")

    asyncio.run(exercise())
    assert request_count == 1
    assert cleanup_count == 1
