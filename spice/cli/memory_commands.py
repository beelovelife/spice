"""Long-term memory management commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from spice.agent.memory import MemoryDistiller, MemoryStore
from spice.llm.config import get_api_key, load_config, save_config
from spice.llm.model_registry import ModelRegistry, find_initial_model
from spice.llm.types import ModelRequestOptions
from spice.storage.factory import create_memory_store


def register_memory_commands(*, app: typer.Typer, console: Console) -> None:
    @app.command("status")
    def status() -> None:
        """Show memory distillation status."""
        config = load_config()
        _print_memory_status(
            create_memory_store(config, workspace=Path.cwd()),
            enabled=config.memory_enabled,
            console=console,
        )

    @app.command("enable")
    def enable() -> None:
        """Enable long-term memory."""
        config = load_config()
        config.memory_enabled = True
        save_config(config)
        console.print("[green]Long-term memory enabled.[/green]")

    @app.command("disable")
    def disable() -> None:
        """Disable long-term memory."""
        config = load_config()
        config.memory_enabled = False
        save_config(config)
        console.print("[yellow]Long-term memory disabled.[/yellow]")

    @app.command("distill")
    def distill(
        scope: Annotated[
            str,
            typer.Option(
                "--scope",
                help="Target scope: project, global (USER.md + MEMORY.md), or all.",
            ),
        ] = "all",
    ) -> None:
        """Distill this workspace's history summaries into long-term memory."""
        config = load_config()
        store = create_memory_store(config, workspace=Path.cwd())
        if not config.memory_enabled:
            console.print(
                "[yellow]Long-term memory is disabled. Run `spice memory enable` first.[/yellow]"
            )
            raise typer.Exit(1)
        allowed_targets = {
            "project": {"project"},
            "global": {"user", "memory"},
            "all": {"user", "memory", "project"},
        }.get(scope.lower())
        if allowed_targets is None:
            console.print(
                "[bold red]Error:[/bold red] --scope must be project, global, or all."
            )
            raise typer.Exit(2)
        resolved = find_initial_model(ModelRegistry(), config)
        if not resolved.model:
            console.print(
                f"[bold red]Error:[/bold red] {resolved.message or 'No model configured.'}"
            )
            raise typer.Exit(1)
        model = resolved.model
        options = ModelRequestOptions(
            api_key=get_api_key(model.provider, env_names=model.api_key_envs),
            temperature=config.temperature,
            max_tokens=model.output_tokens,
            base_url=config.base_url or model.base_url,
        )
        try:
            result = asyncio.run(
                MemoryDistiller(
                    store,
                    model=model,
                    options=options,
                    allowed_targets=allowed_targets,
                ).run()
            )
        except Exception as exc:
            console.print(f"[bold red]Memory distillation failed:[/bold red] {exc}")
            raise typer.Exit(1) from exc
        if result.get("success") is False:
            console.print(
                f"[bold red]Memory distillation failed:[/bold red] "
                f"{result.get('message', 'Plan could not be fully applied.')}"
            )
            raise typer.Exit(1)
        if result.get("processed", 0) == 0:
            console.print(result.get("message", "No unprocessed memory history."))
            return
        console.print(
            f"[green]Memory distillation processed[/green] {result.get('processed')} entries "
            f"(cursor {result.get('from_cursor')}..{result.get('to_cursor')})."
        )
        console.print(
            f"adds={result.get('adds', 0)} replacements={result.get('replacements', 0)} "
            f"removals={result.get('removals', 0)} skipped={result.get('skipped', 0)}"
        )
        raw_cleanup = result.get("cleanup")
        cleanup = raw_cleanup if isinstance(raw_cleanup, dict) else {}
        if cleanup.get("removed"):
            console.print(f"history cleanup removed {cleanup.get('removed')} entries.")


def _print_memory_status(
    store: MemoryStore,
    *,
    enabled: bool,
    console: Console,
) -> None:
    status = store.status()
    table = Table("Metric", "Value")
    table.add_row("enabled", str(enabled).lower())
    table.add_row(
        "history entries",
        f"{status['history_count']}/{status['history_limit']}",
    )
    table.add_row("processed cursor", str(status["processed_cursor"]))
    table.add_row("unprocessed entries", str(status["unprocessed_count"]))
    table.add_row("next distill batch", str(status["next_distill_batch"]))
    table.add_row("USER.md usage", str(status["user_usage"]))
    table.add_row("MEMORY.md usage", str(status["memory_usage"]))
    table.add_row("project MEMORY.md usage", str(status.get("project_usage") or "n/a"))
    table.add_row("workspace", str(status.get("workspace") or "n/a"))
    console.print(table)
