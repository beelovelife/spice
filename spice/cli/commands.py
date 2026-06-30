"""Interactive slash command registry for Spice."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.mouse_events import MouseEventType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spice.agent.agent_session import AgentSession
from spice.agent.sessions import SessionStore
from spice.agent.tool_results import tool_display_text
from spice.extensions.manager import ExtensionManager
from spice.llm.config import get_api_key, load_config, save_config
from spice.llm.models import Model
from spice.llm.model_registry import ModelRegistry
from spice.skills.loader import load_skills, read_skill_file
from spice.storage.factory import create_session_store, storage_backend
from spice.tools.tool_registry import READ_ONLY_TOOLS, TOOLSETS, create_all_tools

MODEL_LABEL_STYLE = "#f4d99d"
MODEL_DETAIL_STYLE = "#e5e7eb"
MODEL_MUTED_STYLE = "#8f99b7"
MODEL_SELECT_BG = "bg:#3f4a68"
MODEL_ARROW_STYLE = "#f0abfc bold"
MODEL_SELECT_ARROW_STYLE = f"{MODEL_SELECT_BG} #f0abfc bold"
MODEL_SELECT_LABEL_STYLE = f"{MODEL_SELECT_BG} #ffe6ad bold"
MODEL_SELECT_DETAIL_STYLE = f"{MODEL_SELECT_BG} #ffffff"
MODEL_SELECT_MUTED_STYLE = f"{MODEL_SELECT_BG} #b6bfd8"
SESSION_PICKER_LIMIT = 30
CHOICE_PICKER_MAX_HEIGHT = 22
CHOICE_PICKER_HEADER_LINES = 3
CHOICE_PICKER_SCROLL_STEP = 5

if TYPE_CHECKING:
    from spice.cli.render import CliRenderer


def _create_session_store(*, cwd: Path | None):
    config = load_config()
    if storage_backend(config) == "sqlite":
        return create_session_store(config, cwd=cwd)
    return SessionStore(cwd=cwd)


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    usage: str

    @property
    def trigger(self) -> str:
        return f"/{self.name}"


@dataclass
class SlashCommandResult:
    session: AgentSession | None = None
    exit_requested: bool = False
    clear_requested: bool = False
    handled: bool = True
    prompt: str | None = None


@dataclass
class InteractiveCommandContext:
    console: Console
    input_session: PromptSession[str]
    renderer: "CliRenderer"
    agent_session: AgentSession
    cwd: Path


class SlashCommandCompleter(Completer):
    def __init__(self, commands: list[tuple[str, str, str]]) -> None:
        self.commands = commands

    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("/"):
            return
        lower_word = word.lower()
        for trigger, usage, description in self.commands:
            if trigger.lower().startswith(lower_word):
                args = usage.removeprefix(trigger).strip()
                display_trigger = usage if usage.startswith(f"{trigger}:") else trigger
                if display_trigger != trigger:
                    args = ""
                display = [
                    ("class:completion.command", display_trigger),
                ]
                if args:
                    display.append(("class:completion.args", f"  {args}"))
                yield Completion(
                    trigger,
                    start_position=-len(word),
                    display=display,
                    display_meta=[("class:completion.description", description)],
                )


class SlashCommandRegistry:
    def __init__(self, extensions: ExtensionManager | None = None) -> None:
        self.extensions = extensions
        self.commands = [
            SlashCommand("models", "Choose the current provider/model.", "/models"),
            SlashCommand("sessions", "List recent sessions for this workspace.", "/sessions"),
            SlashCommand("resume", "Choose and resume a previous session.", "/resume"),
            SlashCommand("clear", "Clear the visible conversation without deleting the session.", "/clear"),
            SlashCommand("reset", "Clear all messages from the current session after confirmation.", "/reset"),
            SlashCommand("delete", "Delete a session after confirmation.", "/delete [session-id|current]"),
            SlashCommand("history", "Show current session history.", "/history [--tree|--raw]"),
            SlashCommand("rewind", "Move the current session leaf to an entry.", "/rewind <entry-id>"),
            SlashCommand("tools", "Show built-in tools and toolsets.", "/tools"),
            SlashCommand("settings", "Show current interactive settings.", "/settings"),
            SlashCommand("subagent", "Control subagent tools for this session.", "/subagent [on|off|status]"),
            SlashCommand("compact", "Compact the current session context.", "/compact [status|focus]"),
            SlashCommand("plan", "Switch to read-only plan mode or plan a task.", "/plan [task|execute|cancel]"),
            SlashCommand("task", "Start, inspect, cancel, or complete a sustained task.", "/task <objective|status|cancel|complete>"),
            SlashCommand("goal", "Start, inspect, cancel, or complete a sustained goal.", "/goal <objective|status|cancel|complete>"),
            SlashCommand("skills", "List installed skills.", "/skills"),
            SlashCommand("skill", "Show a skill by name.", "/skill:<name>"),
            SlashCommand("help", "Show slash commands.", "/help"),
            SlashCommand("quit", "Exit the interactive session.", "/quit"),
        ]

    def completer(self) -> Completer:
        completions = [
            (command.trigger, command.usage, command.description)
            for command in self.commands
        ]
        completions.append(("/exit", "/exit", "Exit the interactive session."))
        if self.extensions:
            completions.extend(
                (f"/{name}", f"/{name}", command.description or "Extension command.")
                for name, command in sorted(self.extensions.commands().items())
            )
        return SlashCommandCompleter(completions)

    async def execute(self, raw_message: str, context: InteractiveCommandContext) -> SlashCommandResult:
        message = raw_message.strip()
        if not message.startswith("/"):
            return SlashCommandResult(handled=False)

        if message in {"/exit", "/quit"}:
            return SlashCommandResult(exit_requested=True)
        if message == "/help":
            self.print_help(context.console)
            return SlashCommandResult()
        if message == "/models":
            return await self._choose_model(context)
        if message == "/sessions":
            return await self._show_sessions(context)
        if message == "/resume":
            return await self._resume_session(context)
        if message == "/clear":
            return SlashCommandResult(clear_requested=True)
        if message == "/reset":
            return await self._reset_session(context)
        if message == "/delete" or message.startswith("/delete "):
            return await self._delete_session(context, message.removeprefix("/delete").strip())
        if message == "/history" or message.startswith("/history "):
            self._print_history(context, message.removeprefix("/history").strip())
            return SlashCommandResult()
        if message == "/rewind" or message.startswith("/rewind "):
            await self._rewind_session(context, message.removeprefix("/rewind").strip())
            return SlashCommandResult(session=context.agent_session)
        if message == "/tools":
            self._print_tools(context.console)
            return SlashCommandResult()
        if message == "/settings":
            self._print_settings(context)
            return SlashCommandResult()
        if message == "/subagent" or message.startswith("/subagent "):
            self._handle_subagent(context, message.removeprefix("/subagent").strip())
            return SlashCommandResult(session=context.agent_session)
        if message == "/compact" or message.startswith("/compact "):
            return await self._compact_session(context, message.removeprefix("/compact").strip())
        if message == "/plan" or message.startswith("/plan "):
            return self._handle_plan(context, message.removeprefix("/plan").strip())
        if message == "/task" or message.startswith("/task "):
            return self._handle_sustained_goal(context, message.removeprefix("/task").strip(), command="task")
        if message == "/goal" or message.startswith("/goal "):
            return self._handle_sustained_goal(context, message.removeprefix("/goal").strip(), command="goal")
        if message == "/skills":
            self._print_skills(context)
            return SlashCommandResult()
        if message.startswith("/skill:"):
            self._print_skill(context, message.removeprefix("/skill:").strip())
            return SlashCommandResult()
        if self.extensions:
            command_name, _, args = message.removeprefix("/").partition(" ")
            if command_name in self.extensions.commands():
                result = await self.extensions.handle_command(command_name, args.strip(), context)
                if result is not None:
                    context.console.print(result)
                return SlashCommandResult()

        context.console.print(f"[red]Unknown command:[/red] {message}")
        context.console.print("Run [bold]/help[/bold] to see available commands.")
        return SlashCommandResult()

    def print_help(self, console: Console) -> None:
        table = Table("Command", "Description", "Usage")
        for command in self.commands:
            table.add_row(command.trigger, command.description, command.usage)
        if self.extensions:
            for name, command in sorted(self.extensions.commands().items()):
                table.add_row(f"/{name}", command.description or "Extension command.", f"/{name}")
        console.print(table)

    def _handle_plan(self, context: InteractiveCommandContext, args: str) -> SlashCommandResult:
        if args == "cancel":
            context.agent_session.cancel_plan()
            context.console.print("[green]Switched to edit mode.[/green] Plan cleared.")
            return SlashCommandResult()
        if args == "execute":
            prompt = context.agent_session.approve_plan("manual")
            context.console.print("[green]Switched to edit mode.[/green] Executing the approved plan.")
            return SlashCommandResult(prompt=prompt)
        context.agent_session.start_plan(args)
        if args:
            context.console.print("[cyan]Plan mode on.[/cyan] Read-only tools are active.")
            return SlashCommandResult(prompt=args)
        context.console.print("[cyan]Plan mode on.[/cyan] Send the task you want to plan. Shift+Tab toggles back to edit mode.")
        return SlashCommandResult()

    def _handle_sustained_goal(self, context: InteractiveCommandContext, args: str, *, command: str) -> SlashCommandResult:
        objective = args.strip()
        if not objective:
            context.console.print(f"[red]Usage:[/red] /{command} <objective|status|cancel|complete>")
            return SlashCommandResult()
        action, _, rest = objective.partition(" ")
        action = action.lower()
        if action == "status":
            self._print_long_task_status(context, command=command)
            return SlashCommandResult(session=context.agent_session)
        if action == "cancel":
            note = rest.strip()
            try:
                context.agent_session.cancel_long_task(note=note)
            except ValueError as exc:
                context.console.print(f"[red]{exc}[/red]")
                return SlashCommandResult(session=context.agent_session)
            context.console.print(f"[yellow]Sustained {command} cancelled.[/yellow]")
            return SlashCommandResult(session=context.agent_session)
        if action == "complete":
            force, note = _parse_force_note(rest)
            try:
                context.agent_session.complete_long_task(note=note, force=force)
            except ValueError as exc:
                context.console.print(f"[red]{exc}[/red]")
                return SlashCommandResult(session=context.agent_session)
            context.console.print(f"[green]Sustained {command} completed.[/green]")
            return SlashCommandResult(session=context.agent_session)
        context.agent_session.set_interaction_mode("edit")
        if hasattr(context.agent_session, "start_long_task"):
            context.agent_session.start_long_task(objective)
        prompt = _sustained_goal_prompt(objective, command=command)
        context.console.print(f"[cyan]Sustained {command} started.[/cyan]")
        return SlashCommandResult(prompt=prompt)

    def _print_long_task_status(self, context: InteractiveCommandContext, *, command: str) -> None:
        state = context.agent_session.long_task_status()
        if not state.objective:
            context.console.print(f"No sustained {command} is active.")
            return
        table = Table("Field", "Value")
        table.add_row("task id", state.task_id or "legacy")
        table.add_row("status", state.status)
        table.add_row("objective", state.objective)
        table.add_row("continuations", f"{state.continuation_rounds}/{state.max_continuation_rounds}")
        table.add_row("remaining", str(state.remaining_continuations))
        table.add_row("needs attention", str(state.needs_user_attention).lower())
        table.add_row("completion candidate", str(state.completion_candidate).lower())
        if state.last_stop_reason:
            table.add_row("last stop", state.last_stop_reason)
        context.console.print(table)

    def _handle_subagent(self, context: InteractiveCommandContext, args: str) -> None:
        action = (args or "status").strip().lower()
        if action in {"on", "enable", "enabled"}:
            context.agent_session.set_subagents_enabled(True)
            context.console.print("[green]Subagents enabled for this session.[/green]")
            self._print_subagent_status(context)
            return
        if action in {"off", "disable", "disabled"}:
            context.agent_session.set_subagents_enabled(False)
            context.console.print("[yellow]Subagents disabled for this session.[/yellow]")
            self._print_subagent_status(context)
            return
        if action in {"", "status"}:
            self._print_subagent_status(context)
            return
        context.console.print("[red]Usage:[/red] /subagent [on|off|status]")

    def _print_subagent_status(self, context: InteractiveCommandContext) -> None:
        enabled = bool(getattr(context.agent_session, "subagents_enabled", False))
        plan_mode = bool(getattr(context.agent_session.plan_state, "is_plan_mode", False))
        active_tools = context.agent_session.get_active_tools()
        tool_available = "spawn_subagents" in active_tools
        manager = getattr(context.agent_session, "subagent_manager", None)
        max_concurrent = getattr(manager, "max_concurrent", "n/a")
        table = Table("Setting", "Value")
        table.add_row("session enabled", str(enabled).lower())
        table.add_row("max concurrent", str(max_concurrent))
        table.add_row("mode", "plan" if plan_mode else "edit")
        table.add_row("tool available", str(tool_available).lower())
        context.console.print(table)

    async def _choose_model(self, context: InteractiveCommandContext) -> SlashCommandResult:
        registry = ModelRegistry()
        config = load_config()
        models = sorted(registry.all(), key=lambda item: (item.provider, item.id))
        choices = [
            (
                f"{model.provider}/{model.id}",
                _model_label(model),
                _model_description(model, current=model.provider == context.agent_session.model.provider and model.id == context.agent_session.model.id),
            )
            for model in models
        ]
        selected = await _select_from_choices(
            context,
            title="Models",
            columns=("Model", "Details"),
            choices=choices,
            current_value=f"{context.agent_session.model.provider}/{context.agent_session.model.id}",
            interactive=True,
        )
        if not selected:
            return SlashCommandResult()

        provider, model_id = selected.split("/", 1)
        model = registry.find(provider, model_id)
        if not model:
            context.console.print(f"[red]Model not found:[/red] {selected}")
            return SlashCommandResult()

        context.agent_session.set_model(model)
        config.default_model = model.profile_key or model.id
        config.provider = model.provider
        config.model = model.id
        config.protocol = model.protocol
        config.base_url = model.base_url
        if model.temperature is not None:
            config.temperature = model.temperature
        save_config(config)
        context.console.print(f"[green]Set model to[/green] [bold]{model.id}[/bold] [green]and saved as your default for new sessions[/green]")
        if not get_api_key(model.provider, env_names=model.api_key_envs):
            context.console.print(f"[yellow]No API key found for {model.provider}.[/yellow]")
        return SlashCommandResult()

    async def _show_sessions(self, context: InteractiveCommandContext) -> SlashCommandResult:
        store = _create_session_store(cwd=context.cwd)
        rows = store.list(limit=SESSION_PICKER_LIMIT, cwd=context.cwd, include_empty=True)
        if not rows:
            context.console.print("No sessions found for this workspace.")
            return SlashCommandResult()
        selected = await _select_from_choices(
            context,
            title="Sessions",
            columns=("Session", "Details"),
            choices=[
                (
                    row.id,
                    row.id,
                    f"{row.updated_at}  {row.provider}/{row.model}  {row.message_count} messages  {row.preview}",
                )
                for row in rows
            ],
            interactive=True,
            current_value=context.agent_session.session_id if getattr(context.agent_session, "session", None) else None,
            show_current_mark=True,
        )
        if not selected:
            return SlashCommandResult()
        try:
            new_session = AgentSession(
                cwd=context.cwd,
                provider=context.agent_session.model.provider,
                model_id=context.agent_session.model.id,
                confirm=context.renderer.confirm,
                session_id=selected,
            )
        except (RuntimeError, ValueError) as exc:
            context.console.print(f"[bold red]Error:[/bold red] {exc}")
            return SlashCommandResult()
        context.console.print(f"[green]Resumed session:[/green] {new_session.session_id}")
        history_messages = [m for m in new_session.messages if m.role != "system"]
        _print_session_history(context.console, history_messages)
        return SlashCommandResult(session=new_session)

    async def _reset_session(self, context: InteractiveCommandContext) -> SlashCommandResult:
        if context.agent_session.session is None:
            context.console.print("No session has been created yet.")
            return SlashCommandResult(clear_requested=True)
        selected = await _select_from_choices(
            context,
            title="Reset",
            columns=("Choice", "Details"),
            choices=[
                ("yes", "Yes", "Clear all messages but keep the current session id."),
                ("no", "No", "Keep the current session."),
            ],
            current_value="yes",
            interactive=True,
        )
        if selected != "yes":
            context.console.print("[dim]Reset cancelled.[/dim]")
            return SlashCommandResult()

        try:
            context.agent_session.reset()
        except (RuntimeError, ValueError) as exc:
            context.console.print(f"[bold red]Error:[/bold red] {exc}")
            return SlashCommandResult()

        context.console.print(f"[green]Reset session:[/green] {context.agent_session.session_id}")
        return SlashCommandResult(session=context.agent_session, clear_requested=True)

    async def _delete_session(self, context: InteractiveCommandContext, target: str) -> SlashCommandResult:
        store = context.agent_session.session_store
        current_session_id = context.agent_session.session_id if context.agent_session.session else None
        selected_id = target
        if target == "current":
            if not current_session_id:
                context.console.print("No current session has been created yet.")
                return SlashCommandResult()
            selected_id = current_session_id
        elif not target:
            rows = store.list(limit=SESSION_PICKER_LIMIT, cwd=context.cwd, include_empty=True)
            if not rows:
                context.console.print("No sessions found for this workspace.")
                return SlashCommandResult()
            selected_id = await _select_from_choices(
                context,
                title="Sessions to delete",
                columns=("Session", "Details"),
                choices=[
                    (
                        row.id,
                        row.id,
                        f"{row.updated_at}  {row.provider}/{row.model}  {row.message_count} messages  {row.preview}",
                    )
                    for row in rows
                ],
                current_value=current_session_id,
                interactive=True,
            ) or ""
            if not selected_id:
                return SlashCommandResult()
        else:
            try:
                selected_id = store.resolve(target, cwd=context.cwd).id
            except ValueError as exc:
                context.console.print(f"[bold red]Error:[/bold red] {exc}")
                return SlashCommandResult()

        confirmed = await self._confirm_delete_session(context, selected_id, current=selected_id == current_session_id)
        if not confirmed:
            context.console.print("[dim]Delete cancelled.[/dim]")
            return SlashCommandResult()

        try:
            store.delete(selected_id)
        except ValueError as exc:
            context.console.print(f"[bold red]Error:[/bold red] {exc}")
            return SlashCommandResult()

        context.console.print(f"[green]Deleted session:[/green] {selected_id}")
        if selected_id != current_session_id:
            return SlashCommandResult()

        new_session = AgentSession(
            cwd=context.cwd,
            provider=context.agent_session.model.provider,
            model_id=context.agent_session.model.id,
            confirm=context.renderer.confirm,
            session_store=store,
            extension_manager=context.agent_session.extensions,
        )
        context.console.print("[green]Started a fresh session.[/green]")
        return SlashCommandResult(session=new_session, clear_requested=True)

    async def _confirm_delete_session(self, context: InteractiveCommandContext, session_id: str, *, current: bool) -> bool:
        detail = "Delete this current session and all of its content. A fresh session will start." if current else "Delete this session and all of its content."
        selected = await _select_from_choices(
            context,
            title=f"Delete {session_id}",
            columns=("Choice", "Details"),
            choices=[
                ("yes", "Yes", detail),
                ("no", "No", "Keep the session."),
            ],
            current_value="yes",
            interactive=True,
        )
        return selected == "yes"

    async def _resume_session(self, context: InteractiveCommandContext) -> SlashCommandResult:
        store = _create_session_store(cwd=context.cwd)
        rows = store.list(limit=SESSION_PICKER_LIMIT, cwd=context.cwd, include_empty=True)
        if not rows:
            context.console.print("No sessions found for this workspace.")
            return SlashCommandResult()

        selected = await _select_from_choices(
            context,
            title="Sessions",
            columns=("Session", "Details"),
            choices=[
                (
                    row.id,
                    row.id,
                    f"{row.updated_at}  {row.provider}/{row.model}  {row.message_count} messages  {row.preview}",
                )
                for row in rows
            ],
            interactive=True,
            current_value=context.agent_session.session_id if getattr(context.agent_session, "session", None) else None,
            show_current_mark=True,
        )
        if not selected:
            return SlashCommandResult()
        try:
            new_session = AgentSession(
                cwd=context.cwd,
                provider=context.agent_session.model.provider,
                model_id=context.agent_session.model.id,
                confirm=context.renderer.confirm,
                session_id=selected,
            )
        except (RuntimeError, ValueError) as exc:
            context.console.print(f"[bold red]Error:[/bold red] {exc}")
            return SlashCommandResult()
        context.console.print(f"[green]Resumed session:[/green] {new_session.session_id}")
        # Display session history messages
        history_messages = [m for m in new_session.messages if m.role != "system"]
        _print_session_history(context.console, history_messages)
        return SlashCommandResult(session=new_session)

    def _print_history(self, context: InteractiveCommandContext, args: str) -> None:
        if context.agent_session.session is None:
            context.console.print("No session has been created yet.")
            return
        store = context.agent_session.session_store
        session_id = context.agent_session.session_id
        try:
            if args == "--raw":
                path = store.path_for(session_id)
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        context.console.print(line)
                    else:
                        context.console.print(json.dumps(parsed, ensure_ascii=False, indent=2))
                return
            if args == "--tree":
                self._print_session_tree(context, store, session_id)
                return
            if args:
                context.console.print("Usage: /history [--tree|--raw]")
                return
            info = store.info(session_id)
            active_context = store.build_context(session_id)
        except ValueError as exc:
            context.console.print(f"[bold red]Error:[/bold red] {exc}")
            return
        context.console.print(
            Panel(
                f"[bold]ID:[/bold] {info.id}\n"
                f"[bold]Model:[/bold] {info.provider}/{info.model}\n"
                f"[bold]CWD:[/bold] {info.cwd}\n"
                f"[bold]Messages:[/bold] {info.message_count}\n"
                f"[bold]Leaf:[/bold] {active_context.leaf_id or ''}",
                title="Session",
            )
        )
        for message in active_context.messages:
            title = message.role
            if message.name:
                title += f":{message.name}"
            body = message.content or ""
            if message.tool_calls:
                calls = ", ".join(f"{call.name}({call.id})" for call in message.tool_calls)
                body = f"{body}\n\nTool calls: {calls}".strip()
            context.console.print(Panel(body or "[dim]<empty>[/dim]", title=title))

    def _print_session_tree(self, context: InteractiveCommandContext, store: SessionStore, session_id: str) -> None:
        info = store.info(session_id)
        active_path_ids = {entry.id for entry in store.path_entries(session_id)} if info.leaf_id else set()
        table = Table("Entry", "Parent", "Type", "Active", "Preview")
        for entry in store.entries(session_id):
            active = "yes" if entry.id == info.leaf_id else ("path" if entry.id in active_path_ids else "")
            table.add_row(entry.id, entry.parent_id or "", entry.type, active, _entry_preview(entry.data))
        context.console.print(table)

    async def _rewind_session(self, context: InteractiveCommandContext, entry_id: str) -> None:
        if context.agent_session.session is None:
            context.console.print("No session has been created yet.")
            return
        if not entry_id:
            store = context.agent_session.session_store
            session_id = context.agent_session.session_id
            try:
                info = store.info(session_id)
                active_path_ids = {entry.id for entry in store.path_entries(session_id)} if info.leaf_id else set()
                entries = store.entries(session_id)
            except ValueError as exc:
                context.console.print(f"[bold red]Error:[/bold red] {exc}")
                return
            choices = [
                (
                    entry.id,
                    entry.id,
                    f"{entry.type}  {_entry_active_label(entry.id, info.leaf_id, active_path_ids)}  {_entry_preview(entry.data)}".strip(),
                )
                for entry in entries
                if entry.type != "leaf"
            ]
            selected = await _select_from_choices(
                context,
                title="Rewind",
                columns=("Entry", "Details"),
                choices=choices,
            )
            if not selected:
                return
            entry_id = selected
        try:
            context.agent_session.rewind(entry_id)
        except (RuntimeError, ValueError) as exc:
            context.console.print(f"[bold red]Error:[/bold red] {exc}")
            return
        context.console.print(f"[green]Rewound session:[/green] {context.agent_session.session_id} -> {entry_id}")

    def _print_tools(self, console: Console) -> None:
        config = load_config()
        memory_enabled = config.memory_enabled
        subagents_enabled = config.subagents_enabled
        registry = create_all_tools(memory_enabled=memory_enabled, subagents_enabled=subagents_enabled)
        table = Table("Tool", "Toolset", "Risk", "Execution", "Timeout", "Description")
        for toolset, names in TOOLSETS.items():
            if toolset == "memory" and not memory_enabled:
                continue
            if toolset == "subagent" and not subagents_enabled:
                continue
            for name in names:
                tool = registry.get(name)
                if not tool:
                    continue
                risk = "read-only" if name in READ_ONLY_TOOLS and not tool.requires_confirmation else "write/exec"
                timeout = f"{tool.timeout_seconds:g}s" if tool.timeout_seconds is not None else "default"
                table.add_row(name, toolset, risk, tool.concurrency, timeout, tool.description)
        console.print(table)

    def _print_settings(self, context: InteractiveCommandContext) -> None:
        config = load_config()
        table = Table("Setting", "Value")
        table.add_row("session", context.agent_session.session_label)
        table.add_row("cwd", str(context.cwd))
        table.add_row("active model", f"{context.agent_session.model.provider}/{context.agent_session.model.id}")
        table.add_row("default model", config.default_model)
        table.add_row("temperature", str(config.temperature))
        table.add_row("memory.enabled", str(config.memory_enabled).lower())
        table.add_row("subagents.enabled", str(getattr(context.agent_session, "subagents_enabled", config.subagents_enabled)).lower())
        manager = getattr(context.agent_session, "subagent_manager", None)
        table.add_row("subagents.max_concurrent", str(getattr(manager, "max_concurrent", config.max_concurrent_subagents)))
        table.add_row("logging.retention_days", str(config.logging_retention_days))
        table.add_row("debug.trace", str(config.debug_trace))
        table.add_row("tools.max_concurrency", str(config.tools.get("max_concurrency", 4)))
        table.add_row("tools.default_timeout_seconds", str(config.tools.get("default_timeout_seconds", 120)))
        table.add_row("model retry", str(config.model_routing["retry"].get("enabled", True)).lower())
        table.add_row("model retry attempts", str(config.model_routing["retry"].get("maxAttempts", 3)))
        table.add_row("model fallback", str(config.model_routing["fallback"].get("enabled", False)).lower())
        table.add_row("fallback profiles", ", ".join(config.model_routing["fallback"].get("profiles", [])) or "(none)")
        table.add_row("output_tokens", str(context.agent_session.model.output_tokens))
        plan_state = getattr(context.agent_session, "plan_state", None)
        table.add_row("mode", getattr(plan_state, "mode", "edit"))
        todo_state = getattr(context.agent_session, "todo_state", None)
        if todo_state is not None and todo_state.status_line():
            table.add_row("todo", todo_state.status_line())
        table.add_row("tools", ", ".join(context.agent_session.get_active_tools()))
        table.add_row("session save", "enabled")
        context.console.print(table)

    async def _compact_session(self, context: InteractiveCommandContext, args: str) -> SlashCommandResult:
        if context.agent_session.session is None:
            context.console.print("No session has been created yet.")
            return SlashCommandResult()
        if args in {"status", "--status"}:
            self._print_compaction_status(context)
            return SlashCommandResult()
        focus = args or None
        try:
            result = await context.agent_session.compact(focus=focus, reason="manual", force=True)
        except Exception as exc:
            context.console.print(f"[bold red]Compaction failed:[/bold red] {exc}")
            return SlashCommandResult()
        context.console.print(
            f"[green]Compacted session:[/green] ~{result.tokens_before} -> ~{result.tokens_after} tokens"
        )
        return SlashCommandResult(session=context.agent_session)

    def _print_compaction_status(self, context: InteractiveCommandContext) -> None:
        status = context.agent_session.compaction_status()
        threshold = str(status.threshold_tokens) if status.threshold_tokens is not None else "n/a"
        context.console.print(
            Panel(
                f"[bold]Estimated tokens:[/bold] {status.estimated_tokens}\n"
                f"[bold]Threshold:[/bold] {threshold}\n"
                f"[bold]Reserve:[/bold] {status.reserve_tokens}\n"
                f"[bold]Auto compact:[/bold] {'yes' if status.should_compact else 'no'}\n"
                f"[bold]Reason:[/bold] {status.reason}",
                title="/compact status",
            )
        )

    def _print_skills(self, context: InteractiveCommandContext) -> None:
        console = context.console
        result = load_skills(cwd=context.cwd)
        skills = result.skills
        if not skills:
            console.print("No skills found in ~/.spice/skills or ./.spice/skills.")
            return
        table = Table("Skill", "Source", "Description")
        for skill in skills:
            table.add_row(skill.name, skill.source, skill.description)
        console.print(table)
        for diagnostic in result.diagnostics:
            console.print(f"[yellow]{diagnostic.type}:[/yellow] {diagnostic.message}")

    def _print_skill(self, context: InteractiveCommandContext, name: str) -> None:
        console = context.console
        if not name:
            console.print("Usage: /skill:<name>")
            return
        try:
            console.print(read_skill_file(name, cwd=context.cwd))
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")


async def _select_from_choices(
    context: InteractiveCommandContext,
    *,
    title: str,
    columns: tuple[str, str],
    choices: list[tuple[str, object, object]],
    current_value: str | None = None,
    interactive: bool = False,
    show_current_mark: bool = False,
) -> str | None:
    if interactive:
        return await _select_interactive(
            title=title,
            choices=choices,
            current_value=current_value,
            show_current_mark=show_current_mark,
        )

    table = Table("#", *columns)
    for index, (_, label, description) in enumerate(choices, start=1):
        table.add_row(str(index), label, description)
    context.console.print(table)
    if not choices:
        return None

    choice_words = [str(index) for index in range(1, len(choices) + 1)]
    choice_words.extend(value for value, _, _ in choices)
    completer = WordCompleter(choice_words, ignore_case=True)
    try:
        raw = (
            await context.input_session.prompt_async(
                f"Select {title.lower()} (blank to cancel)> ",
                completer=completer,
            )
        ).strip()
    except (EOFError, KeyboardInterrupt):
        context.console.print()
        return None
    if not raw:
        context.console.print("[dim]Cancelled.[/dim]")
        return None
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(choices):
            return choices[index - 1][0]
    for value, label, _ in choices:
        if raw == value or raw == str(label):
            return value
    context.console.print(f"[red]Invalid selection:[/red] {raw}")
    return None


async def _select_interactive(
    *,
    title: str,
    choices: list[tuple[str, object, object]],
    current_value: str | None = None,
    show_current_mark: bool = False,
) -> str | None:
    if not choices:
        return None

    selected_index = 0
    if current_value:
        for index, (value, _, _) in enumerate(choices):
            if value == current_value:
                selected_index = index
                break
    picker_height = min(CHOICE_PICKER_MAX_HEIGHT, len(choices) + CHOICE_PICKER_HEADER_LINES)
    visible_choice_count = max(1, picker_height - CHOICE_PICKER_HEADER_LINES)
    top_index = min(selected_index, max(0, len(choices) - visible_choice_count))

    def _clamp_top_index() -> None:
        nonlocal top_index
        top_index = max(0, min(top_index, max(0, len(choices) - visible_choice_count)))

    def _keep_selected_visible() -> None:
        nonlocal top_index
        if selected_index < top_index:
            top_index = selected_index
        elif selected_index >= top_index + visible_choice_count:
            top_index = selected_index - visible_choice_count + 1
        _clamp_top_index()

    def _scroll(delta: int) -> None:
        nonlocal selected_index, top_index
        top_index += delta
        _clamp_top_index()
        if selected_index < top_index:
            selected_index = top_index
        elif selected_index >= top_index + visible_choice_count:
            selected_index = top_index + visible_choice_count - 1

    key_bindings = KeyBindings()

    @key_bindings.add("up")
    def _move_up(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(choices)
        _keep_selected_visible()
        event.app.invalidate()

    @key_bindings.add("down")
    def _move_down(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(choices)
        _keep_selected_visible()
        event.app.invalidate()

    @key_bindings.add("enter")
    def _confirm(event) -> None:
        event.app.exit(result=choices[selected_index][0])

    @key_bindings.add("escape")
    @key_bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    def _mouse_handler(index: int):
        def _handle(mouse_event):
            nonlocal selected_index
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                _scroll(-CHOICE_PICKER_SCROLL_STEP)
                get_app().invalidate()
            elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                _scroll(CHOICE_PICKER_SCROLL_STEP)
                get_app().invalidate()
            elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                selected_index = index
                _keep_selected_visible()
                get_app().exit(result=choices[selected_index][0])
            return None

        return _handle

    def _fragments():
        label_width = max(len(_plain_text(label)) for _, label, _ in choices)
        number_width = len(str(len(choices)))
        gutter = " "
        fragments = [
            ("", "\n"),
            ("", "  "),
            ("#8f99b7", f"Select {title.lower()} "),
            ("#6b7280", "(up/down, enter, esc)\n\n"),
        ]
        visible_choices = choices[top_index : top_index + visible_choice_count]
        for relative_index, (value, label, description) in enumerate(visible_choices):
            index = top_index + relative_index
            mouse_handler = _mouse_handler(index)
            selected = index == selected_index
            current = value == current_value
            arrow = "❯" if selected else " "
            number = f"{index + 1:>{number_width}}. "
            plain_label = _plain_text(label)
            label_padding = " " * (label_width - len(plain_label) + 2)
            current_mark = "√ " if show_current_mark and current else ""
            arrow_style = MODEL_SELECT_ARROW_STYLE if selected else MODEL_ARROW_STYLE
            label_style = MODEL_SELECT_LABEL_STYLE if selected else MODEL_LABEL_STYLE
            detail_style = MODEL_SELECT_DETAIL_STYLE if selected else MODEL_DETAIL_STYLE
            muted_style = MODEL_SELECT_MUTED_STYLE if selected else MODEL_MUTED_STYLE
            fragments.append(("", gutter, mouse_handler))
            fragments.append((arrow_style, arrow, mouse_handler))
            fragments.append((muted_style, " ", mouse_handler))
            fragments.append((muted_style, number, mouse_handler))
            if current_mark:
                fragments.append((f"{detail_style} bold", current_mark, mouse_handler))
            fragments.append((label_style, plain_label, mouse_handler))
            fragments.append((muted_style, label_padding, mouse_handler))
            details = _plain_text(description)
            if current:
                if details == "current":
                    details = ""
                elif details.startswith("current · "):
                    details = details.removeprefix("current · ")
            if details:
                fragments.append((muted_style, "  ", mouse_handler))
                fragments.append((detail_style, details, mouse_handler))
            fragments.append(("", "\n"))
        return fragments

    def _cursor_position() -> Point:
        return Point(x=0, y=CHOICE_PICKER_HEADER_LINES + selected_index - top_index)

    class ChoicePickerControl(FormattedTextControl):
        def mouse_handler(self, mouse_event):
            nonlocal selected_index
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                _scroll(-CHOICE_PICKER_SCROLL_STEP)
                get_app().invalidate()
                return None
            if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                _scroll(CHOICE_PICKER_SCROLL_STEP)
                get_app().invalidate()
                return None
            return super().mouse_handler(mouse_event)

    class ChoicePickerScrollbarMargin(Margin):
        def get_width(self, get_ui_content) -> int:
            return 1

        def create_margin(self, window_render_info, width: int, height: int):
            if len(choices) <= visible_choice_count:
                return [("", " \n") for _ in range(height)]
            scrollbar_height = max(1, int(height * visible_choice_count / len(choices)))
            max_scroll = max(1, len(choices) - visible_choice_count)
            scrollbar_top = int((height - scrollbar_height) * top_index / max_scroll)
            rows = []
            for row in range(height):
                if scrollbar_top <= row < scrollbar_top + scrollbar_height:
                    rows.append(("class:scrollbar.button", " "))
                else:
                    rows.append(("class:scrollbar.background", " "))
                rows.append(("", "\n"))
            return rows

    class ChoicePickerWindow(Window):
        def write_to_screen(self, screen, mouse_handlers, write_position, parent_style, erase_bg, z_index):
            super().write_to_screen(screen, mouse_handlers, write_position, parent_style, erase_bg, z_index)

            def _handle(mouse_event):
                nonlocal selected_index
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    _scroll(-CHOICE_PICKER_SCROLL_STEP)
                    get_app().invalidate()
                    return None
                if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    _scroll(CHOICE_PICKER_SCROLL_STEP)
                    get_app().invalidate()
                    return None
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    row = mouse_event.position.y - write_position.ypos - CHOICE_PICKER_HEADER_LINES
                    if 0 <= row < visible_choice_count:
                        index = top_index + row
                        if index < len(choices):
                            selected_index = index
                            get_app().exit(result=choices[selected_index][0])
                            return None
                return None

            mouse_handlers.set_mouse_handler_for_range(
                x_min=write_position.xpos,
                x_max=write_position.xpos + write_position.width,
                y_min=write_position.ypos,
                y_max=write_position.ypos + write_position.height,
                handler=_handle,
            )

    control = ChoicePickerControl(
        _fragments,
        focusable=True,
        get_cursor_position=_cursor_position,
    )
    picker_window = ChoicePickerWindow(
        content=control,
        always_hide_cursor=True,
        height=Dimension(min=min(picker_height, 8), preferred=picker_height, max=picker_height),
        right_margins=[ChoicePickerScrollbarMargin()],
    )
    application = Application(
        layout=Layout(HSplit([picker_window])),
        key_bindings=key_bindings,
        full_screen=False,
        mouse_support=True,
    )
    return await application.run_async()


def _plain_text(value: object) -> str:
    if isinstance(value, Text):
        return value.plain
    return str(value)


def _model_label(model: Model) -> Text:
    return Text(f"{model.provider}/{model.id}", style=MODEL_LABEL_STYLE)


def _model_description(model: Model, *, current: bool) -> Text:
    description = Text()
    parts = []
    if current:
        parts.append("current")
    if model.context_window:
        parts.append(f"context {model.context_window}")
    if model.output_tokens:
        parts.append(f"output {model.output_tokens}")
    key = "key yes" if get_api_key(model.provider, env_names=model.api_key_envs) else "key missing"
    parts.append(key)
    for index, part in enumerate(parts):
        if index:
            description.append(" · ", style=MODEL_MUTED_STYLE)
        description.append(part, style=MODEL_DETAIL_STYLE)
    return description


def _entry_preview(entry: dict) -> str:
    if entry.get("type") == "message" and isinstance(entry.get("message"), dict):
        message = entry["message"]
        return f"{message.get('role', '')}: {str(message.get('content', '')).strip()[:80]}"
    if entry.get("type") == "model_change":
        return f"{entry.get('provider', '')}/{entry.get('modelId', '')}"
    if entry.get("type") == "compaction":
        return str(entry.get("summary", "")).strip()[:80]
    if entry.get("type") == "leaf":
        return f"target={entry.get('targetId', '')}"
    return ""


def _entry_active_label(entry_id: str, leaf_id: str | None, active_path_ids: set[str]) -> str:
    if entry_id == leaf_id:
        return "active"
    if entry_id in active_path_ids:
        return "path"
    return ""


def _print_session_history(console: Console, messages: list) -> None:
    """Render session messages like a live conversation replay."""
    from rich.markdown import Markdown
    from spice.cli.render import _build_table, _is_table_divider, _looks_like_table_line

    if not messages:
        return
    console.print()
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            content = (message.content or "").strip()
            if content:
                console.print(f"[bold green]You:[/bold green] {content}")
                console.print()
        elif message.role == "assistant":
            content = (message.content or "").strip()
            if content:
                console.print("[bold cyan]Spice:[/bold cyan] ", end="")
                # Render markdown content with table support
                _render_history_markdown(console, content)
                console.print()
            if message.tool_calls:
                for call in message.tool_calls:
                    console.print(f"  [magenta]tool: {call.name}()[/magenta]")
        elif message.role == "tool":
            # Show tool results compactly
            name = message.name or "tool"
            display = tool_display_text(message)
            if display:
                console.print(f"  [dim]{name} -> {display}[/dim]")
                continue
            content = (message.content or "").strip()
            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                console.print(f"  [dim]{name} -> {preview}[/dim]")
    console.print()


def _render_history_markdown(console: Console, text: str) -> None:
    """Render markdown text with table support for history display."""
    from rich.markdown import Markdown
    from spice.cli.render import _build_table, _is_table_divider, _looks_like_table_line

    lines = text.splitlines()
    markdown_lines: list[str] = []
    table_lines: list[str] = []

    def flush_markdown() -> None:
        if not markdown_lines:
            return
        md_text = "\n".join(markdown_lines).strip()
        markdown_lines.clear()
        if md_text:
            console.print(Markdown(md_text))

    def flush_table() -> None:
        if not table_lines:
            return
        if len(table_lines) >= 2 and _is_table_divider(table_lines[1]):
            flush_markdown()
            console.print(_build_table(table_lines))
        else:
            markdown_lines.extend(table_lines)
        table_lines.clear()

    for line in lines:
        if table_lines:
            if len(table_lines) == 1 and _is_table_divider(line):
                table_lines.append(line)
                continue
            if len(table_lines) >= 2 and _looks_like_table_line(line):
                table_lines.append(line)
                continue
            flush_table()

        if _looks_like_table_line(line):
            table_lines.append(line)
            continue
        markdown_lines.append(line)

    flush_table()
    flush_markdown()


def _sustained_goal_prompt(objective: str, *, command: str) -> str:
    label = "goal" if command == "goal" else "task"
    return f"""Start or continue this sustained {label}:

{objective}

This is long-running execution mode, not read-only planning mode.

The sustained goal has already been persisted by the /{command} command. Work toward the objective using the available tools. Break work into concrete steps when useful, keep the todo list updated for immediate execution progress, and continue until the objective is actually done or you need user input.

When the objective is fully done and verified, call complete_long_task with a concise completion note. Do not call complete_long_task merely because you produced a plan.
"""


def _parse_force_note(text: str) -> tuple[bool, str]:
    parts = text.split()
    force = False
    kept: list[str] = []
    for part in parts:
        if part == "--force":
            force = True
        else:
            kept.append(part)
    return force, " ".join(kept)
