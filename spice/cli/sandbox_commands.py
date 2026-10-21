"""Sandbox lifecycle and execution commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from spice.llm.config import load_config
from spice.sandbox.factory import create_environment, create_workspace_policy


def register_sandbox_commands(*, app: typer.Typer, console: Console) -> None:
    async def print_status() -> None:
        config = load_config()
        workspace = create_workspace_policy(config.sandbox, cwd=Path.cwd())
        environment = create_environment(config.sandbox, cwd=Path.cwd())
        settings = config.sandbox if isinstance(config.sandbox, dict) else {}
        mode = str(settings.get("mode") or "workspace")
        table = Table("Field", "Value")
        table.add_row("Mode", mode)
        table.add_row(
            "Environment",
            getattr(environment, "name", type(environment).__name__),
        )
        table.add_row("Workspace", str(workspace.root))
        table.add_row("Restrict workspace", str(workspace.restrict))
        from spice.sandbox.docker import DockerEnvironment

        if isinstance(environment, DockerEnvironment):
            try:
                status = await environment.status()
            except Exception as exc:
                table.add_row("Backend status", f"error: {exc}")
            else:
                for key, value in status.items():
                    table.add_row(str(key), str(value))
        console.print(table)

    @app.command("status")
    def status() -> None:
        """Show sandbox status for the current workspace."""
        asyncio.run(print_status())

    @app.command("init")
    def init() -> None:
        """Create or start the configured sandbox for the current workspace."""

        async def run() -> None:
            environment = create_environment(load_config().sandbox, cwd=Path.cwd())
            try:
                await environment.ensure_ready()
            except Exception as exc:
                console.print(f"[bold red]Sandbox init failed:[/bold red] {exc}")
                raise typer.Exit(1) from exc
            console.print("[green]Sandbox ready.[/green]")
            await print_status()

        asyncio.run(run())

    @app.command("exec")
    def execute(
        command: Annotated[
            str,
            typer.Argument(help="Command to run inside the configured sandbox."),
        ],
    ) -> None:
        """Run one command through the configured sandbox."""

        async def run() -> None:
            config = load_config()
            workspace = create_workspace_policy(config.sandbox, cwd=Path.cwd())
            environment = create_environment(config.sandbox, cwd=Path.cwd())
            try:
                cwd = workspace.resolve_exec_cwd(".")
                result = await environment.run(command, cwd=cwd, timeout=600)
            except Exception as exc:
                console.print(f"[bold red]Sandbox exec failed:[/bold red] {exc}")
                raise typer.Exit(1) from exc
            if result.output:
                console.print(result.output.rstrip())
            if result.exit_code:
                raise typer.Exit(result.exit_code)

        asyncio.run(run())

    @app.command("stop")
    def stop() -> None:
        """Stop the current workspace sandbox when supported by the backend."""

        async def run() -> None:
            environment = create_environment(load_config().sandbox, cwd=Path.cwd())
            stop_method = getattr(environment, "stop", None)
            if stop_method is None:
                console.print(
                    "[yellow]Sandbox environment does not support stop:[/yellow] "
                    f"{getattr(environment, 'name', type(environment).__name__)}"
                )
                return
            try:
                await stop_method()
            except Exception as exc:
                console.print(f"[bold red]Sandbox stop failed:[/bold red] {exc}")
                raise typer.Exit(1) from exc
            console.print("[green]Sandbox stopped.[/green]")

        asyncio.run(run())
