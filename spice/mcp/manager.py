"""MCP server orchestration and Spice tool registration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from spice.mcp.adapter import adapt_mcp_tool
from spice.mcp.config import McpConfigLoadResult, McpServerConfig, load_mcp_config
from spice.mcp.connection import McpConnection, sanitize_mcp_error
from spice.mcp.trust import McpTrustStore
from spice.tools.base import ConfirmFn, Tool

McpStatusName = Literal["configured", "connecting", "connected", "degraded", "failed", "disabled"]


@dataclass
class McpServerStatus:
    name: str
    state: McpStatusName
    transport: str
    source: str
    tool_count: int = 0
    error: str = ""


class McpManager:
    def __init__(self, *, cwd: Path, confirm: ConfirmFn | None = None) -> None:
        self.cwd = cwd.resolve()
        self.confirm = confirm
        self.trust_store = McpTrustStore()
        self.config_result: McpConfigLoadResult = load_mcp_config(self.cwd)
        self.connections: dict[str, McpConnection] = {}
        self.statuses: dict[str, McpServerStatus] = {}
        self._tools: list[Tool] = []
        self._read_only_names: set[str] = set()
        self._initialized = False
        self.revision = 0
        self._lock = asyncio.Lock()
        self._reset_statuses()

    @property
    def config_errors(self) -> tuple[str, ...]:
        return self.config_result.errors

    async def ensure_connected(self, server_names: set[str] | None = None) -> None:
        async with self._lock:
            if self._initialized:
                await self._refresh_stale()
                return
            self._initialized = True
            configs = [
                config
                for config in self.config_result.servers.values()
                if config.enabled and (server_names is None or config.name in server_names)
            ]
            await asyncio.gather(*(self._connect_one(config) for config in configs))
            self._rebuild_tools()

    async def reload(self) -> None:
        async with self._lock:
            await self._close_all()
            self.config_result = load_mcp_config(self.cwd)
            self._initialized = False
            self._tools = []
            self._read_only_names.clear()
            self._reset_statuses()
        await self.ensure_connected()

    def tools(self) -> list[Tool]:
        return list(self._tools)

    def read_only_tools(self) -> list[Tool]:
        return [tool for tool in self._tools if tool.name in self._read_only_names]

    def status(self) -> list[McpServerStatus]:
        return [self.statuses[name] for name in sorted(self.statuses)]

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        connection = self.connections.get(server_name)
        if connection is None:
            raise RuntimeError(f"MCP server {server_name!r} is not connected")
        return await connection.call_tool(tool_name, arguments)

    async def close(self) -> None:
        async with self._lock:
            await self._close_all()
            self._initialized = False

    async def _connect_one(self, config: McpServerConfig) -> None:
        status = self.statuses[config.name]
        if config.source == "project" and config.transport == "stdio":
            if not await self._ensure_trusted(config):
                status.state = "failed"
                status.error = "Project MCP stdio command was not trusted."
                return
        status.state = "connecting"
        connection = McpConnection(config, cwd=self.cwd)
        try:
            tools = await connection.connect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status.state = "failed"
            status.error = sanitize_mcp_error(exc)
            await connection.close()
            return
        self.connections[config.name] = connection
        allowed = [tool for tool in tools if config.tool_filter.allows(str(tool.name))]
        connection.tools = allowed
        status.state = "connected"
        status.tool_count = len(allowed)

    async def _ensure_trusted(self, config: McpServerConfig) -> bool:
        if self.trust_store.is_trusted(self.cwd, config):
            return True
        if self.confirm is None:
            return False
        allowed = await self.confirm(
            "mcp_server",
            {
                "server": config.name,
                "command": config.command,
                "args": list(config.args),
                "source": str(config.source_path or ""),
            },
        )
        if allowed:
            self.trust_store.trust(self.cwd, config)
        return allowed

    async def _refresh_stale(self) -> None:
        changed = False
        for name, connection in list(self.connections.items()):
            if not connection.stale:
                continue
            try:
                tools = await connection.refresh_tools()
                connection.tools = [tool for tool in tools if connection.config.tool_filter.allows(str(tool.name))]
                self.statuses[name].tool_count = len(connection.tools)
                self.statuses[name].state = "connected"
                self.statuses[name].error = ""
            except Exception as exc:
                self.statuses[name].state = "degraded"
                self.statuses[name].error = sanitize_mcp_error(exc)
            changed = True
        if changed:
            self._rebuild_tools()

    def _rebuild_tools(self) -> None:
        tools: list[Tool] = []
        read_only: set[str] = set()
        names: set[str] = set()
        for server_name in sorted(self.connections):
            connection = self.connections[server_name]
            for remote in connection.tools:
                adapted, is_read_only = adapt_mcp_tool(connection.config, remote, self.call_tool)
                if adapted.name in names:
                    status = self.statuses[server_name]
                    status.state = "degraded"
                    status.error = f"MCP tool name collision: {adapted.name}"
                    continue
                names.add(adapted.name)
                tools.append(adapted)
                if is_read_only:
                    read_only.add(adapted.name)
        self._tools = tools
        self._read_only_names = read_only
        self.revision += 1

    async def _close_all(self) -> None:
        await asyncio.gather(*(connection.close() for connection in self.connections.values()), return_exceptions=True)
        self.connections.clear()

    def _reset_statuses(self) -> None:
        self.statuses = {
            name: McpServerStatus(
                name=name,
                state="configured" if config.enabled else "disabled",
                transport=config.transport,
                source=config.source,
            )
            for name, config in self.config_result.servers.items()
        }
