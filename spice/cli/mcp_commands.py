"""MCP management commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from spice.cli.render import CliRenderer
from spice.llm.config import SETTINGS_PATH, _read_json, _write_private_json
from spice.mcp.config import load_mcp_config
from spice.mcp.manager import McpManager
from spice.mcp.trust import McpTrustStore


def register_mcp_commands(*, app: typer.Typer, console: Console) -> None:
    @app.command("list")
    def list_servers() -> None:
        """List configured MCP servers without connecting."""
        result = load_mcp_config(Path.cwd())
        table = Table("Server", "Enabled", "Transport", "Source", "Endpoint")
        for name, server in sorted(result.servers.items()):
            endpoint = (
                " ".join([server.command or "", *server.args]).strip()
                if server.transport == "stdio"
                else server.url or ""
            )
            table.add_row(
                name,
                str(server.enabled),
                server.transport,
                server.source,
                endpoint,
            )
        console.print(table if result.servers else "No MCP servers configured.")
        for error in result.errors:
            console.print(f"[yellow]Config warning:[/yellow] {error}")

    @app.command("add")
    def add_server(
        name: str,
        command: Annotated[
            str | None, typer.Option("--command", help="Stdio executable.")
        ] = None,
        url: Annotated[
            str | None, typer.Option("--url", help="Streamable HTTP endpoint.")
        ] = None,
        args: Annotated[
            list[str] | None,
            typer.Option("--arg", help="Stdio argument; repeat for multiple values."),
        ] = None,
    ) -> None:
        """Add or replace a global MCP server."""
        if bool(command) == bool(url):
            raise typer.BadParameter("Provide exactly one of --command or --url.")
        data = _read_json(SETTINGS_PATH)
        servers = data.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            raise typer.BadParameter("settings.json mcpServers must be an object")
        server: dict[str, object] = {"enabled": True}
        if command:
            server.update({"command": command, "args": list(args or [])})
        else:
            server["url"] = url
        servers[name] = server
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_private_json(SETTINGS_PATH, data)
        console.print(f"Added MCP server {name}.")

    @app.command("remove")
    def remove_server(name: str) -> None:
        """Remove a global MCP server."""
        data = _read_json(SETTINGS_PATH)
        servers = data.get("mcpServers")
        if not isinstance(servers, dict) or name not in servers:
            raise typer.BadParameter(f"Global MCP server not found: {name}")
        del servers[name]
        _write_private_json(SETTINGS_PATH, data)
        console.print(f"Removed MCP server {name}.")

    def set_enabled(name: str, enabled: bool) -> None:
        data = _read_json(SETTINGS_PATH)
        servers = data.get("mcpServers")
        if not isinstance(servers, dict) or not isinstance(servers.get(name), dict):
            raise typer.BadParameter(f"Global MCP server not found: {name}")
        servers[name]["enabled"] = enabled
        _write_private_json(SETTINGS_PATH, data)
        console.print(f"{'Enabled' if enabled else 'Disabled'} MCP server {name}.")

    @app.command("enable")
    def enable_server(name: str) -> None:
        """Enable a global MCP server."""
        set_enabled(name, True)

    @app.command("disable")
    def disable_server(name: str) -> None:
        """Disable a global MCP server."""
        set_enabled(name, False)

    @app.command("trust")
    def trust_server(name: str) -> None:
        """Trust the current project's configured stdio command."""
        result = load_mcp_config(Path.cwd())
        server = result.servers.get(name)
        if server is None or server.source != "project" or server.transport != "stdio":
            raise typer.BadParameter("Trust applies only to project stdio MCP servers.")
        McpTrustStore().trust(Path.cwd(), server)
        console.print(f"Trusted project MCP server {name} for this configuration.")

    async def inspect(name: str, *, show_tools: bool) -> None:
        renderer = CliRenderer(console)
        manager = McpManager(cwd=Path.cwd(), confirm=renderer.confirm)
        try:
            await manager.ensure_connected({name})
            status = next(
                (item for item in manager.status() if item.name == name), None
            )
            if status is None:
                raise typer.BadParameter(f"MCP server not found: {name}")
            if status.state not in {"connected", "degraded"}:
                console.print(f"[red]{name}: {status.state}[/red] {status.error}")
                raise typer.Exit(1)
            console.print(
                f"[green]{name}: connected[/green] ({status.tool_count} tools)"
            )
            if show_tools:
                table = Table("Tool", "Confirmation", "Description")
                prefix = f"mcp__{name}__"
                for tool in manager.tools():
                    if tool.name.startswith(prefix):
                        table.add_row(
                            tool.name,
                            str(tool.requires_confirmation),
                            tool.description,
                        )
                console.print(table)
        finally:
            await manager.close()

    @app.command("test")
    def test_server(name: str) -> None:
        """Connect to and test an MCP server."""
        asyncio.run(inspect(name, show_tools=False))

    @app.command("tools")
    def server_tools(name: str) -> None:
        """Connect and list tools discovered from an MCP server."""
        asyncio.run(inspect(name, show_tools=True))
