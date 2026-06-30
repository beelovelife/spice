"""Typer CLI for Spice."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from spice.version import __version__
from spice.agent.agent_session import AgentSession
from spice.agent.debug_trace import trace_path
from spice.agent.logging_config import configure_logging, log_path
from spice.agent.memory import MemoryDistiller, MemoryStore
from spice.agent.sessions import SessionInfo, SessionStore
from spice.agent.tool_results import tool_display_text
from spice.agent.trace import DEFAULT_TRACE_PATH, RunTraceWriter
from spice.cli.process import set_process_title
from spice.cli.render import CliRenderer
from spice.cli.run_interactive import render_prompt, run_conversation
from spice.llm.config import SECRETS_PATH, SETTINGS_PATH, get_api_key, load_config, save_config, save_secret
from spice.llm.model_registry import ModelRegistry, find_initial_model
from spice.llm.types import ModelRequestOptions
from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.skills.loader import load_skills, read_skill_file
from spice.storage.factory import create_memory_store, create_session_store, storage_backend
from spice.storage.sqlite import init_sqlite_database
from spice.tui import run_tui

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help="Spice command line agent.",
)
config_app = typer.Typer(help="Manage Spice configuration.")
sessions_app = typer.Typer(help="Manage Spice sessions.", invoke_without_command=True)
skills_app = typer.Typer(help="Inspect Spice skills.")
memory_app = typer.Typer(help="Manage Spice memory.")
sandbox_app = typer.Typer(help="Manage Spice sandbox execution.")
storage_app = typer.Typer(help="Manage Spice application-state storage.")
app.add_typer(config_app, name="config")
app.add_typer(sessions_app, name="sessions")
app.add_typer(skills_app, name="skills")
app.add_typer(memory_app, name="memory")
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(storage_app, name="storage")

console = Console()


def _create_session_store(*, cwd: Path | None):
    config = load_config()
    if storage_backend(config) == "sqlite":
        return create_session_store(config, cwd=cwd)
    return SessionStore(cwd=cwd)


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", "-v", help="Show Spice version.")] = False,
    markdown: Annotated[bool, typer.Option("--markdown/--no-markdown", help="Render Markdown-friendly streaming output.")] = True,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging.")] = False,
) -> None:
    """Run Spice. Without a subcommand, enter the interactive terminal."""
    configure_logging(debug=debug)
    set_process_title()
    if version:
        console.print(f"spice {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        asyncio.run(run_conversation(console, markdown=markdown))


@app.command()
def chat(
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Provider name.")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model id.")] = None,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Continue a session by id.")] = None,
    continue_session: Annotated[bool, typer.Option("--continue", "-c", help="Continue the latest session for this cwd.")] = False,
    markdown: Annotated[bool, typer.Option("--markdown/--no-markdown", help="Render Markdown-friendly streaming output.")] = True,
) -> None:
    """Enter the interactive terminal."""
    asyncio.run(
        run_conversation(
            console,
            provider=provider,
            model=model,
            session_id=session_id,
            continue_session=continue_session,
            markdown=markdown,
        )
    )


@app.command()
def tui(
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Provider name.")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model id.")] = None,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Continue a session by id.")] = None,
    continue_session: Annotated[bool, typer.Option("--continue", "-c", help="Continue the latest session for this cwd.")] = False,
) -> None:
    """Enter the full-screen TUI with a fixed bottom input."""
    run_tui(
        provider=provider,
        model=model,
        session_id=session_id,
        continue_session=continue_session,
    )


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="Prompt to run once.")],
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Provider name.")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model id.")] = None,
    session_id: Annotated[str | None, typer.Option("--session", "-s", help="Continue a session by id.")] = None,
    continue_session: Annotated[bool, typer.Option("--continue", "-c", help="Continue the latest session for this cwd.")] = False,
    markdown: Annotated[bool, typer.Option("--markdown/--no-markdown", help="Render Markdown-friendly streaming output.")] = True,
    trace: Annotated[bool, typer.Option("--trace/--no-trace", help="Write a runtime trace JSON file.")] = False,
    trace_file: Annotated[Path | None, typer.Option("--trace-file", help="Runtime trace output path. Implies --trace.")] = None,
) -> None:
    """Run a single prompt."""
    renderer = CliRenderer(console, markdown=markdown)
    try:
        agent_session = AgentSession(
            cwd=Path.cwd(),
            provider=provider,
            model_id=model,
            confirm=renderer.confirm,
            session_id=session_id,
            continue_latest=continue_session,
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    trace_writer = None
    if trace or trace_file is not None:
        trace_path = trace_file or DEFAULT_TRACE_PATH
        trace_writer = RunTraceWriter(trace_path, agent_session)
        agent_session.subscribe(trace_writer.record)
        console.print(f"[dim]Trace: {trace_path}[/dim]")
    console.print(f"[dim]Session: {agent_session.session_label}[/dim]")
    asyncio.run(render_prompt(agent_session, renderer, prompt))


@app.command()
def models() -> None:
    """Show available models and current configuration."""
    config = load_config()
    table = Table("Provider", "Model", "Current", "API key")
    registry = ModelRegistry()
    for model in registry.all():
        current = "yes" if model.provider == config.provider and model.id == config.model else ""
        key = "yes" if get_api_key(model.provider, env_names=model.api_key_envs) else ""
        table.add_row(model.provider, model.id, current, key)
    console.print(table)


@sandbox_app.command("status")
def sandbox_status() -> None:
    """Show sandbox status for the current workspace."""
    asyncio.run(_print_sandbox_status())


@sandbox_app.command("init")
def sandbox_init() -> None:
    """Create or start the configured sandbox for the current workspace."""
    asyncio.run(_sandbox_init())


@sandbox_app.command("exec")
def sandbox_exec(command: Annotated[str, typer.Argument(help="Command to run inside the configured sandbox.")]) -> None:
    """Run one command through the configured sandbox."""
    asyncio.run(_sandbox_exec(command))


@sandbox_app.command("stop")
def sandbox_stop() -> None:
    """Stop the current workspace sandbox when supported by the backend."""
    asyncio.run(_sandbox_stop())


@memory_app.command("status")
def memory_status() -> None:
    """Show memory distillation status."""
    config = load_config()
    _print_memory_status(create_memory_store(config), enabled=config.memory_enabled)


@memory_app.command("enable")
def memory_enable() -> None:
    """Enable long-term memory."""
    config = load_config()
    config.memory_enabled = True
    save_config(config)
    console.print("[green]Long-term memory enabled.[/green]")


@memory_app.command("disable")
def memory_disable() -> None:
    """Disable long-term memory."""
    config = load_config()
    config.memory_enabled = False
    save_config(config)
    console.print("[yellow]Long-term memory disabled.[/yellow]")


@memory_app.command("distill")
def memory_distill() -> None:
    """Distill history summaries into long-term memory."""
    config = load_config()
    store = create_memory_store(config)
    if not config.memory_enabled:
        console.print("[yellow]Long-term memory is disabled. Run `spice memory enable` first.[/yellow]")
        raise typer.Exit(1)
    registry = ModelRegistry()
    resolved = find_initial_model(registry, config)
    if not resolved.model:
        console.print(f"[bold red]Error:[/bold red] {resolved.message or 'No model configured.'}")
        raise typer.Exit(1)
    model = resolved.model
    options = ModelRequestOptions(
        api_key=get_api_key(model.provider, env_names=model.api_key_envs),
        temperature=config.temperature,
        max_tokens=model.output_tokens,
        base_url=config.base_url or model.base_url,
    )
    try:
        result = asyncio.run(MemoryDistiller(store, model=model, options=options).run())
    except Exception as exc:
        console.print(f"[bold red]Memory distillation failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    if result.get("success") is False:
        console.print(f"[bold red]Memory distillation failed:[/bold red] {result.get('message', 'Plan could not be fully applied.')}")
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
    cleanup = result.get("cleanup") if isinstance(result.get("cleanup"), dict) else {}
    if cleanup.get("removed"):
        console.print(f"history cleanup removed {cleanup.get('removed')} entries.")


def _print_memory_status(store: MemoryStore, *, enabled: bool) -> None:
    status = store.status()
    table = Table("Metric", "Value")
    table.add_row("enabled", str(enabled).lower())
    table.add_row("history entries", f"{status['history_count']}/{status['history_limit']}")
    table.add_row("processed cursor", str(status["processed_cursor"]))
    table.add_row("unprocessed entries", str(status["unprocessed_count"]))
    table.add_row("next distill batch", str(status["next_distill_batch"]))
    table.add_row("USER.md usage", str(status["user_usage"]))
    table.add_row("MEMORY.md usage", str(status["memory_usage"]))
    console.print(table)


@app.command()
def logs(
    tail: Annotated[int, typer.Option("--tail", "-n", help="Number of log lines to show.")] = 80,
    path_only: Annotated[bool, typer.Option("--path", help="Only print the log file path.")] = False,
) -> None:
    """Show the Spice runtime log path and recent lines."""
    path = log_path()
    if path_only:
        console.print(str(path))
        return
    console.print(f"[dim]Log file:[/dim] {path}")
    if not path.exists():
        console.print("[yellow]No log file yet.[/yellow]")
        return
    lines: deque[str] = deque(maxlen=max(tail, 0))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line.rstrip("\n"))
    except OSError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


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
    file_path: Annotated[str | None, typer.Option("--file", "-f", help="Linked file path inside the skill directory.")] = None,
) -> None:
    """Show a skill SKILL.md or linked file."""
    try:
        console.print(read_skill_file(name, file_path, cwd=Path.cwd()))
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


@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="Session id to resume.")],
    provider: Annotated[str | None, typer.Option("--provider", "-p", help="Provider name.")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model id.")] = None,
) -> None:
    """Resume an interactive session by id."""
    asyncio.run(run_conversation(console, provider=provider, model=model, session_id=session_id))


@sessions_app.callback()
def sessions(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of sessions to show.")] = 20,
    all_cwd: Annotated[bool, typer.Option("--all", help="Show sessions from every cwd.")] = False,
) -> None:
    """List sessions."""
    if ctx.invoked_subcommand is not None:
        return
    store = _create_session_store(cwd=None if all_cwd else Path.cwd())
    rows = store.list(limit=limit, cwd=None if all_cwd else Path.cwd())
    if not rows:
        console.print("No sessions found.")
        return
    table = Table("ID", "Leaf", "Updated", "Model", "Messages", "Preview", "CWD")
    for row in rows:
        table.add_row(
            row.id,
            row.leaf_id or "",
            row.updated_at,
            f"{row.provider}/{row.model}",
            str(row.message_count),
            row.preview,
            row.cwd,
        )
    console.print(table)


@sessions_app.command("show")
def sessions_show(
    session_id: Annotated[str, typer.Argument(help="Session id to show.")],
    tree: Annotated[bool, typer.Option("--tree", help="Show all entries as a tree.")] = False,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw JSONL entries.")] = False,
) -> None:
    """Show history for a session."""
    store = _create_session_store(cwd=Path.cwd())
    try:
        info = store.resolve(session_id, cwd=Path.cwd())
        if raw:
            _print_raw_session(store, info.id)
            return
        if tree:
            _print_session_tree(store, info.id)
            return
        context = store.build_context(info.id)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    _print_session_header(info, context.leaf_id)
    _print_messages(context.messages)


@sessions_app.command("rewind")
def sessions_rewind(
    session_id: Annotated[str, typer.Argument(help="Session id to rewind.")],
    entry_id: Annotated[str, typer.Argument(help="Entry id to make active.")],
) -> None:
    """Move a session's active leaf to an earlier entry."""
    store = _create_session_store(cwd=Path.cwd())
    try:
        info = store.resolve(session_id, cwd=Path.cwd())
        marker_id = store.set_leaf(info.id, entry_id)
        updated = store.info(info.id)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Rewound session:[/green] {updated.id} -> {updated.leaf_id} [dim](marker {marker_id})[/dim]")


@sessions_app.command("stats")
def sessions_stats(
    all_cwd: Annotated[bool, typer.Option("--all", help="Count sessions from every cwd.")] = False,
) -> None:
    """Show session counts."""
    store = _create_session_store(cwd=None if all_cwd else Path.cwd())
    rows = _session_rows(store, all_cwd=all_cwd)
    empty_count = sum(1 for row in rows if not row.leaf_id)
    table = Table("Metric", "Value")
    table.add_row("scope", "all workspaces" if all_cwd else str(Path.cwd()))
    table.add_row("sessions", str(len(rows)))
    table.add_row("empty sessions", str(empty_count))
    updated_values = [row.updated_at for row in rows if row.updated_at]
    if updated_values:
        table.add_row("oldest updated", min(updated_values))
        table.add_row("newest updated", max(updated_values))
    table.add_row("storage", str(store.base_root))
    console.print(table)


@sessions_app.command("workspaces")
def sessions_workspaces() -> None:
    """List workspaces that have sessions."""
    store = _create_session_store(cwd=None)
    rows = _session_rows(store, all_cwd=True)
    if not rows:
        console.print("No sessions found.")
        return
    stats: dict[str, dict[str, str | int]] = {}
    for row in rows:
        cwd = row.cwd or "(unknown)"
        item = stats.setdefault(cwd, {"sessions": 0, "empty": 0, "latest": ""})
        item["sessions"] = int(item["sessions"]) + 1
        if not row.leaf_id:
            item["empty"] = int(item["empty"]) + 1
        if row.updated_at > str(item["latest"]):
            item["latest"] = row.updated_at
    table = Table("CWD", "Sessions", "Empty", "Latest")
    for cwd, item in sorted(stats.items(), key=lambda pair: str(pair[1]["latest"]), reverse=True):
        table.add_row(cwd, str(item["sessions"]), str(item["empty"]), str(item["latest"]))
    console.print(table)


@sessions_app.command("delete")
def sessions_delete(
    session_id: Annotated[str, typer.Argument(help="Session id to delete.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Delete without prompting.")] = False,
) -> None:
    """Delete one session."""
    store = _create_session_store(cwd=Path.cwd())
    try:
        info = store.resolve(session_id, cwd=Path.cwd())
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    if not yes and not typer.confirm(f"Delete session {info.id}? This cannot be undone.", default=False):
        console.print("Cancelled.")
        return
    try:
        store.delete(info.id)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Deleted session:[/green] {info.id}")


@sessions_app.command("prune")
def sessions_prune(
    all_cwd: Annotated[bool, typer.Option("--all", help="Prune sessions from every cwd.")] = False,
    keep_recent: Annotated[int | None, typer.Option("--keep-recent", help="Keep the newest N sessions and prune the rest.")] = None,
    before: Annotated[str | None, typer.Option("--before", help="Prune sessions updated before this date/time.")] = None,
    start: Annotated[str | None, typer.Option("--from", help="Prune sessions updated at or after this date/time.")] = None,
    end: Annotated[str | None, typer.Option("--to", help="Prune sessions updated before or at this date/time.")] = None,
    all_sessions: Annotated[bool, typer.Option("--all-sessions", help="Prune every session in scope.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Actually delete the selected sessions.")] = False,
) -> None:
    """Preview or delete sessions by retention rule."""
    selected_rules = sum(
        [
            keep_recent is not None,
            before is not None,
            start is not None or end is not None,
            all_sessions,
        ]
    )
    if selected_rules != 1:
        raise typer.BadParameter("Choose exactly one rule: --keep-recent, --before, --from/--to, or --all-sessions.")
    if keep_recent is not None and keep_recent < 0:
        raise typer.BadParameter("--keep-recent must be >= 0.")

    store = _create_session_store(cwd=None if all_cwd else Path.cwd())
    rows = _session_rows(store, all_cwd=all_cwd)
    candidates = _prune_candidates(rows, keep_recent=keep_recent, before=before, start=start, end=end, all_sessions=all_sessions)
    _print_prune_preview(candidates, all_cwd=all_cwd)
    if not candidates:
        return
    if not yes:
        console.print("Dry run only. Use --yes to delete these sessions.")
        return
    if not typer.confirm(f"Delete {len(candidates)} sessions? This cannot be undone.", default=False):
        console.print("Cancelled.")
        return
    deleted = 0
    for info in candidates:
        try:
            store.delete(info.id)
        except ValueError as exc:
            console.print(f"[yellow]Skipped {info.id}:[/yellow] {exc}")
            continue
        deleted += 1
    console.print(f"[green]Deleted sessions:[/green] {deleted}")


@storage_app.command("init")
def storage_init(
    backend: Annotated[
        str | None,
        typer.Argument(help="Storage backend to initialize without changing settings.json. Defaults to settings.json."),
    ] = None,
) -> None:
    """Initialize application-state storage without changing settings."""
    config = load_config()
    selected_backend = (backend or storage_backend(config)).strip().lower()
    if selected_backend in {"postgres", "pgsql", "pqsql"}:
        console.print("[yellow]PostgreSQL storage is not implemented yet.[/yellow]")
        raise typer.Exit(1)
    if selected_backend == "file":
        console.print("[green]File storage needs no database initialization.[/green]")
        return
    if selected_backend != "sqlite":
        console.print(f"[bold red]Error:[/bold red] unsupported storage backend: {selected_backend}")
        raise typer.Exit(1)
    sqlite_path = Path(str(config.storage.get("sqlitePath") or "~/.spice/spice.db")).expanduser()
    init_sqlite_database(sqlite_path)
    console.print(f"[green]Initialized SQLite storage:[/green] {sqlite_path}")


def _session_rows(store: SessionStore, *, all_cwd: bool) -> list[SessionInfo]:
    return store.list(limit=1_000_000, cwd=None if all_cwd else Path.cwd(), include_empty=True)


def _prune_candidates(
    rows: list[SessionInfo],
    *,
    keep_recent: int | None,
    before: str | None,
    start: str | None,
    end: str | None,
    all_sessions: bool,
) -> list[SessionInfo]:
    if all_sessions:
        return rows
    if keep_recent is not None:
        return rows[keep_recent:]
    if before is not None:
        cutoff = _parse_session_datetime(before, end_of_day=False)
        return [row for row in rows if _session_updated_at(row) < cutoff]
    start_dt = _parse_session_datetime(start, end_of_day=False) if start is not None else None
    end_dt = _parse_session_datetime(end, end_of_day=True) if end is not None else None
    return [
        row
        for row in rows
        if (start_dt is None or _session_updated_at(row) >= start_dt)
        and (end_dt is None or _session_updated_at(row) <= end_dt)
    ]


def _parse_session_datetime(value: str, *, end_of_day: bool) -> datetime:
    text = value.strip()
    if not text:
        raise typer.BadParameter("Date/time cannot be empty.")
    try:
        if "T" not in text and len(text) == 10:
            parsed_date = datetime.fromisoformat(text).date()
            parsed = datetime.combine(parsed_date, time.max if end_of_day else time.min)
        else:
            parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date/time: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _session_updated_at(info: SessionInfo) -> datetime:
    if not info.updated_at:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(info.updated_at)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _print_prune_preview(candidates: list[SessionInfo], *, all_cwd: bool) -> None:
    scope = "all workspaces" if all_cwd else str(Path.cwd())
    console.print(f"[bold]Scope:[/bold] {scope}")
    console.print(f"[bold]Matched sessions:[/bold] {len(candidates)}")
    if not candidates:
        return
    table = Table("ID", "Updated", "Model", "Messages", "Preview", "CWD")
    for row in candidates[:50]:
        table.add_row(
            row.id,
            row.updated_at,
            f"{row.provider}/{row.model}",
            str(row.message_count),
            row.preview,
            row.cwd,
        )
    console.print(table)
    if len(candidates) > 50:
        console.print(f"[dim]...and {len(candidates) - 50} more.[/dim]")


def _print_session_header(info, leaf_id: str | None) -> None:
    console.print(
        Panel(
            f"[bold]ID:[/bold] {info.id}\n"
            f"[bold]Model:[/bold] {info.provider}/{info.model}\n"
            f"[bold]CWD:[/bold] {info.cwd}\n"
            f"[bold]Messages:[/bold] {info.message_count}\n"
            f"[bold]Leaf:[/bold] {leaf_id or ''}",
            title="Session",
        )
    )


def _print_messages(messages) -> None:
    for message in messages:
        title = message.role
        if message.name:
            title += f":{message.name}"
        body = message.content or ""
        if message.role == "tool":
            body = tool_display_text(message) or body
        if message.tool_calls:
            calls = ", ".join(f"{call.name}({call.id})" for call in message.tool_calls)
            body = f"{body}\n\nTool calls: {calls}".strip()
        console.print(Panel(body or "[dim]<empty>[/dim]", title=title))


def _print_session_tree(store: SessionStore, session_id: str) -> None:
    info = store.info(session_id)
    entries = store.entries(session_id)
    active_path_ids = {entry.id for entry in store.path_entries(session_id)} if info.leaf_id else set()
    table = Table("Entry", "Parent", "Type", "Active", "Preview")
    for entry in entries:
        active = "yes" if entry.id == info.leaf_id else ("path" if entry.id in active_path_ids else "")
        table.add_row(
            entry.id,
            entry.parent_id or "",
            entry.type,
            active,
            _entry_preview(entry.data),
        )
    console.print(table)


def _print_raw_session(store: SessionStore, session_id: str) -> None:
    path = store.path_for(session_id)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            console.print(line)
        else:
            console.print(json.dumps(parsed, ensure_ascii=False, indent=2))


def _entry_preview(entry: dict) -> str:
    if entry.get("type") == "message" and isinstance(entry.get("message"), dict):
        message = entry["message"]
        return f"{message.get('role', '')}: {str(message.get('content', '')).strip()[:80]}"
    if entry.get("type") == "model_change":
        return f"{entry.get('provider', '')}/{entry.get('model', '')}"
    if entry.get("type") == "compaction":
        return str(entry.get("summary", "")).strip()[:80]
    if entry.get("type") == "leaf":
        return f"target={entry.get('leaf_id', '')}"
    return ""


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    console.print(load_config())
    console.print(f"settings: {SETTINGS_PATH}")
    console.print(f"secrets: {SECRETS_PATH}")
    console.print(f"debug trace: {trace_path()}")


@config_app.command("path")
def config_path() -> None:
    """Show configuration paths."""
    console.print(str(SETTINGS_PATH))
    console.print(str(SECRETS_PATH))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set model, tool runtime, retry/fallback, storage, and diagnostic settings."""
    if key == "api-key":
        provider = load_config().provider
        save_secret(provider, value)
        console.print(f"Saved API key for {provider}.")
        return
    config = load_config()
    if key in {"default-model", "default_model", "defaultModel"}:
        model = ModelRegistry().find(None, value)
        if not model:
            raise typer.BadParameter(f"Unknown model profile: {value}")
        config.default_model = model.profile_key or model.id
        config.provider = model.provider
        config.model = model.id
        config.protocol = model.protocol
        config.base_url = model.base_url
        if model.temperature is not None:
            config.temperature = model.temperature
    elif key in {"debug.trace", "debug_trace"}:
        config.debug_trace = _parse_bool(value)
    elif key in {"memory.enabled", "memory_enabled"}:
        config.memory_enabled = _parse_bool(value)
    elif key in {"logging.retention_days", "logging_retention_days"}:
        try:
            retention_days = int(value)
        except ValueError as exc:
            raise typer.BadParameter("logging.retention_days must be an integer >= 1") from exc
        if retention_days < 1:
            raise typer.BadParameter("logging.retention_days must be an integer >= 1")
        config.logging_retention_days = retention_days
    elif key in {"storage.backend", "storage_backend"}:
        backend = value.strip().lower()
        if backend not in {"file", "sqlite"}:
            raise typer.BadParameter("storage.backend must be one of: file, sqlite")
        config.storage["backend"] = backend
    elif key in {"storage.sqlitePath", "storage.sqlite_path", "storage_sqlite_path"}:
        sqlite_path = value.strip()
        if not sqlite_path:
            raise typer.BadParameter("storage.sqlitePath cannot be empty")
        config.storage["sqlitePath"] = sqlite_path
    elif key in {"tools.max_concurrency", "tools_max_concurrency"}:
        try:
            config.tools["max_concurrency"] = min(max(int(value), 1), 16)
        except ValueError as exc:
            raise typer.BadParameter("tools.max_concurrency must be an integer from 1 to 16") from exc
    elif key in {"tools.default_timeout_seconds", "tools_default_timeout_seconds"}:
        try:
            config.tools["default_timeout_seconds"] = min(max(float(value), 1.0), 3600.0)
        except ValueError as exc:
            raise typer.BadParameter("tools.default_timeout_seconds must be a number from 1 to 3600") from exc
    elif key in {"modelRouting.retry.enabled", "model_routing.retry.enabled"}:
        config.model_routing["retry"]["enabled"] = _parse_bool(value)
    elif key in {"modelRouting.retry.maxAttempts", "model_routing.retry.max_attempts"}:
        try:
            config.model_routing["retry"]["maxAttempts"] = min(max(int(value), 1), 5)
        except ValueError as exc:
            raise typer.BadParameter("modelRouting.retry.maxAttempts must be an integer from 1 to 5") from exc
    elif key in {"modelRouting.fallback.enabled", "model_routing.fallback.enabled"}:
        config.model_routing["fallback"]["enabled"] = _parse_bool(value)
    elif key in {"modelRouting.fallback.profiles", "model_routing.fallback.profiles"}:
        profiles = [item.strip() for item in value.split(",") if item.strip()]
        if len(profiles) > 3:
            raise typer.BadParameter("modelRouting.fallback.profiles accepts at most 3 comma-separated profiles")
        if len(set(profiles)) != len(profiles):
            raise typer.BadParameter("modelRouting.fallback.profiles cannot contain duplicates")
        registry = ModelRegistry()
        unknown = [profile for profile in profiles if registry.find(None, profile) is None]
        if unknown:
            raise typer.BadParameter(f"Unknown fallback model profiles: {', '.join(unknown)}")
        primary = registry.find(config.provider, config.model)
        repeated_primary = [profile for profile in profiles if registry.find(None, profile) == primary]
        if repeated_primary:
            raise typer.BadParameter("The primary model cannot also be a fallback profile")
        config.model_routing["fallback"]["profiles"] = profiles
    else:
        raise typer.BadParameter(
            "Supported keys: default-model, api-key, debug.trace, memory.enabled, logging.retention_days, "
            "storage.*, tools.*, modelRouting.retry.*, modelRouting.fallback.*"
        )
    save_config(config)
    console.print(f"Updated {key}.")


@config_app.command("get")
def config_get(key: str) -> None:
    """Get a configuration value."""
    config = load_config()
    if key in {"debug.trace", "debug_trace"}:
        console.print(config.debug_trace)
        return
    if key in {"memory.enabled", "memory_enabled"}:
        console.print(config.memory_enabled)
        return
    if key in {"logging.retention_days", "logging_retention_days"}:
        console.print(config.logging_retention_days)
        return
    if key in {"storage.backend", "storage_backend"}:
        console.print(config.storage.get("backend", "file"))
        return
    if key in {"storage.sqlitePath", "storage.sqlite_path", "storage_sqlite_path"}:
        console.print(config.storage.get("sqlitePath", "~/.spice/spice.db"))
        return
    if key in {"tools.max_concurrency", "tools_max_concurrency"}:
        console.print(config.tools.get("max_concurrency", 4))
        return
    if key in {"tools.default_timeout_seconds", "tools_default_timeout_seconds"}:
        console.print(config.tools.get("default_timeout_seconds", 120))
        return
    if key in {"modelRouting.retry.enabled", "model_routing.retry.enabled"}:
        console.print(config.model_routing["retry"].get("enabled", True))
        return
    if key in {"modelRouting.retry.maxAttempts", "model_routing.retry.max_attempts"}:
        console.print(config.model_routing["retry"].get("maxAttempts", 3))
        return
    if key in {"modelRouting.fallback.enabled", "model_routing.fallback.enabled"}:
        console.print(config.model_routing["fallback"].get("enabled", False))
        return
    if key in {"modelRouting.fallback.profiles", "model_routing.fallback.profiles"}:
        console.print(",".join(config.model_routing["fallback"].get("profiles", [])))
        return
    if key in {"default-model", "defaultModel"}:
        console.print(config.default_model)
        return
    if not hasattr(config, key):
        raise typer.BadParameter(f"Unknown config key: {key}")
    console.print(getattr(config, key))


async def _print_sandbox_status() -> None:
    config = load_config()
    workspace = create_workspace_policy(config.sandbox, cwd=Path.cwd())
    environment = create_environment(config.sandbox, cwd=Path.cwd())
    settings = config.sandbox if isinstance(config.sandbox, dict) else {}
    mode = str(settings.get("mode") or "workspace")
    table = Table("Field", "Value")
    table.add_row("Mode", mode)
    table.add_row("Environment", getattr(environment, "name", type(environment).__name__))
    table.add_row("Workspace", str(workspace.root))
    table.add_row("Restrict workspace", str(workspace.restrict))
    if hasattr(environment, "status"):
        try:
            status = await environment.status()
        except Exception as exc:
            table.add_row("Backend status", f"error: {exc}")
        else:
            for key, value in status.items():
                table.add_row(str(key), str(value))
    console.print(table)


async def _sandbox_init() -> None:
    environment = create_environment(load_config().sandbox, cwd=Path.cwd())
    try:
        await environment.ensure_ready()
    except Exception as exc:
        console.print(f"[bold red]Sandbox init failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print("[green]Sandbox ready.[/green]")
    await _print_sandbox_status()


async def _sandbox_exec(command: str) -> None:
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


async def _sandbox_stop() -> None:
    environment = create_environment(load_config().sandbox, cwd=Path.cwd())
    stop = getattr(environment, "stop", None)
    if stop is None:
        console.print(f"[yellow]Sandbox environment does not support stop:[/yellow] {getattr(environment, 'name', type(environment).__name__)}")
        return
    try:
        await stop()
    except Exception as exc:
        console.print(f"[bold red]Sandbox stop failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print("[green]Sandbox stopped.[/green]")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise typer.BadParameter("Boolean value must be one of: true/false, yes/no, on/off, 1/0")


if __name__ == "__main__":
    app()
