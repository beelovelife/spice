"""Top-level doctor/tools commands and skill inspection commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spice.diagnostics import DoctorReport, build_doctor_report
from spice.llm.config import load_config
from spice.skills.loader import load_skills, read_skill_content
from spice.tools.catalog import build_tool_catalog, find_tool_entry, tool_summary


def register_system_commands(
    *,
    app: typer.Typer,
    tools_app: typer.Typer,
    skills_app: typer.Typer,
    console: Console,
) -> None:
    """Bind system-inspection commands to the shared Typer applications."""

    @app.command()
    def doctor(
        json_output: Annotated[
            bool,
            typer.Option("--json", help="Emit a machine-readable diagnostic report."),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-v", help="Show diagnostic paths and details."),
        ] = False,
    ) -> None:
        """Check local Spice configuration and runtime health without network requests."""
        report = build_doctor_report(cwd=Path.cwd())
        if json_output:
            typer.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            _render_doctor_report(report, console=console, verbose=verbose)
        if not report.ready:
            raise typer.Exit(1)

    @tools_app.callback()
    def tools(
        ctx: typer.Context,
        json_output: Annotated[
            bool, typer.Option("--json", help="Emit the tool catalog as JSON.")
        ] = False,
        available_only: Annotated[
            bool,
            typer.Option(
                "--available-only",
                help="Only show tools that are currently available.",
            ),
        ] = False,
    ) -> None:
        """List built-in tools under the current feature configuration."""
        if ctx.invoked_subcommand is not None:
            return
        entries = build_tool_catalog(load_config())
        if available_only:
            entries = tuple(entry for entry in entries if entry.status == "enabled")
        if json_output:
            typer.echo(
                json.dumps(
                    [entry.to_dict() for entry in entries],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        table = Table("", "Tool", "Toolset", "Risk", "Confirm", "Description")
        status_labels = {
            "enabled": "[green]✓[/green]",
            "disabled": "[dim]○[/dim]",
            "unavailable": "[yellow]![/yellow]",
        }
        for entry in entries:
            description = tool_summary(entry.description)
            if entry.note:
                description = f"{description} [dim]({entry.note})[/dim]"
            table.add_row(
                status_labels[entry.status],
                entry.name,
                entry.toolset,
                entry.risk,
                entry.confirmation,
                description,
            )
        console.print(table)
        enabled = sum(entry.status == "enabled" for entry in entries)
        unavailable = sum(entry.status == "unavailable" for entry in entries)
        disabled = sum(entry.status == "disabled" for entry in entries)
        console.print(
            f"{enabled} enabled · {unavailable} unavailable · {disabled} disabled"
        )

    @tools_app.command("show")
    def tools_show(
        name: Annotated[str, typer.Argument(help="Built-in tool name.")],
        json_output: Annotated[
            bool, typer.Option("--json", help="Emit tool details as JSON.")
        ] = False,
    ) -> None:
        """Show one tool's status and JSON Schema."""
        entry = find_tool_entry(build_tool_catalog(load_config()), name)
        if entry is None:
            console.print(f"[bold red]Error:[/bold red] Unknown tool: {name}")
            raise typer.Exit(1)
        if json_output:
            typer.echo(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2))
            return
        table = Table(show_header=False, box=None)
        for label, value in (
            ("Tool", entry.name),
            ("Status", entry.status),
            ("Toolset", entry.toolset),
            ("Risk", entry.risk),
            ("Confirmation", entry.confirmation),
            ("Execution", entry.execution),
            ("Timeout", entry.timeout),
            ("Description", entry.description),
            ("Note", entry.note),
        ):
            if value:
                table.add_row(f"[bold]{label}[/bold]", value)
        console.print(table)
        console.print(
            Panel(
                json.dumps(entry.parameters, ensure_ascii=False, indent=2),
                title="Parameters · JSON Schema",
            )
        )

    @skills_app.command("list")
    def skills_list() -> None:
        """List available skills."""
        result = load_skills(cwd=Path.cwd())
        if not result.skills:
            console.print("No skills found.")
            return
        table = Table("Name", "Source", "Priority", "Triggers", "Always", "Description")
        for skill in result.skills:
            table.add_row(
                skill.name,
                skill.source,
                str(skill.priority),
                ", ".join(skill.triggers),
                "yes" if skill.always else "",
                skill.description,
            )
        console.print(table)

    @skills_app.command("view")
    def skills_view(
        name: Annotated[str, typer.Argument(help="Skill name.")],
        file_path: Annotated[
            str | None,
            typer.Option(
                "--file",
                "-f",
                help="Linked file path inside the skill directory.",
            ),
        ] = None,
    ) -> None:
        """Show a skill SKILL.md or linked file."""
        try:
            console.print(read_skill_content(name, file_path, cwd=Path.cwd()))
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from exc

    @skills_app.command("doctor")
    def skills_doctor() -> None:
        """Show skill loading diagnostics."""
        result = load_skills(cwd=Path.cwd())
        if not result.diagnostics:
            console.print("[green]No skill diagnostics.[/green]")
            return
        table = Table("Type", "Name", "Path", "Message")
        for diagnostic in result.diagnostics:
            table.add_row(
                diagnostic.type,
                diagnostic.name or "",
                str(diagnostic.path),
                diagnostic.message,
            )
        console.print(table)


def _render_doctor_report(
    report: DoctorReport,
    *,
    console: Console,
    verbose: bool,
) -> None:
    table = Table("", "Check", "Result", title="Spice Doctor")
    status_labels = {
        "pass": "[green]✓[/green]",
        "warning": "[yellow]![/yellow]",
        "error": "[red]✗[/red]",
    }
    for check in report.checks:
        summary = check.summary
        if verbose and check.details:
            summary = f"{summary}\n[dim]{' · '.join(check.details)}[/dim]"
        table.add_row(status_labels[check.status], check.name, summary)
    console.print(table)
    if report.ready and report.warning_count:
        console.print(f"[yellow]Ready with {report.warning_count} warning(s).[/yellow]")
    elif report.ready:
        console.print("[green]Ready.[/green]")
    else:
        console.print(
            f"[red]Not ready: {report.error_count} error(s), "
            f"{report.warning_count} warning(s).[/red]"
        )
