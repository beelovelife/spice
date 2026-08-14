"""CLI adapter for the shared interactive slash command layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
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
from spice.extensions.manager import ExtensionManager
from spice.interactive.commands import SlashCommand as CoreSlashCommand
from spice.interactive.commands import SlashCommandRegistry as CoreSlashCommandRegistry
from spice.interactive.confirm import ConfirmPolicy
from spice.interactive.sessions import entry_active_label, entry_preview
from spice.interactive.types import (
    ChoiceRequest,
    CommandContext,
    CommandResult,
    CommandView,
    ConfirmRequest,
    PanelView,
    TableView,
    TextView,
)

# Re-exports used by TUI / tests during migration.
from spice.interactive.commands import is_execute_request, parse_force_note, sustained_goal_prompt  # noqa: F401

_parse_force_note = parse_force_note

MODEL_LABEL_STYLE = "#f4d99d"
MODEL_DETAIL_STYLE = "#e5e7eb"
MODEL_MUTED_STYLE = "#8f99b7"
MODEL_SELECT_BG = "bg:#3f4a68"
MODEL_ARROW_STYLE = "#f0abfc bold"
MODEL_SELECT_ARROW_STYLE = f"{MODEL_SELECT_BG} #f0abfc bold"
MODEL_SELECT_LABEL_STYLE = f"{MODEL_SELECT_BG} #ffe6ad bold"
MODEL_SELECT_DETAIL_STYLE = f"{MODEL_SELECT_BG} #ffffff"
MODEL_SELECT_MUTED_STYLE = f"{MODEL_SELECT_BG} #b6bfd8"
CHOICE_PICKER_MAX_HEIGHT = 22
CHOICE_PICKER_HEADER_LINES = 3
CHOICE_PICKER_SCROLL_STEP = 5

if TYPE_CHECKING:
    from spice.cli.render import CliRenderer

SlashCommand = CoreSlashCommand


@dataclass
class SlashCommandResult:
    session: AgentSession | None = None
    exit_requested: bool = False
    clear_requested: bool = False
    handled: bool = True
    prompt: str | None = None
    replay_session_history: bool = False
    views: list[CommandView] = field(default_factory=list)


@dataclass
class InteractiveCommandContext:
    console: Console
    input_session: PromptSession[str] | None
    renderer: "CliRenderer"
    agent_session: AgentSession
    cwd: Path
    confirm_policy: ConfirmPolicy | None = None


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
                display: StyleAndTextTuples = [
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


class CliInteractivePort:
    def __init__(self, context: InteractiveCommandContext) -> None:
        self.context = context

    async def choose(self, request: ChoiceRequest) -> str | None:
        choices = [(item.id, item.label, item.detail) for item in request.items]
        return await _select_from_choices(
            self.context,
            title=request.title,
            columns=request.columns or ("Option", "Details"),
            choices=choices,
            current_value=request.current_id,
            interactive=True,
            show_current_mark=request.show_current_mark,
        )

    async def confirm(self, request: ConfirmRequest) -> str:
        choices = [(item.id, item.label, item.detail) for item in request.choices]
        selected = await _select_from_choices(
            self.context,
            title=request.question,
            columns=("Choice", "Details"),
            choices=choices,
            current_value=request.current_id,
            interactive=True,
        )
        return selected or "deny"


def paint_command_result(console: Console, result: CommandResult) -> None:
    for view in result.views:
        if isinstance(view, TextView):
            style_map = {
                "plain": None,
                "success": "green",
                "warning": "yellow",
                "error": "red",
                "dim": "dim",
            }
            style = style_map.get(view.style)
            if style:
                console.print(f"[{style}]{view.text}[/{style}]")
            else:
                console.print(view.text)
        elif isinstance(view, TableView):
            table = Table(*view.columns, title=view.title)
            for row in view.rows:
                table.add_row(*row)
            console.print(table)
        elif isinstance(view, PanelView):
            console.print(Panel(view.body, title=view.title))


class SlashCommandRegistry:
    """CLI-facing registry wrapping the shared interactive registry."""

    def __init__(self, extensions: ExtensionManager | None = None) -> None:
        self.extensions = extensions
        self._core = CoreSlashCommandRegistry(extensions)
        self.commands = self._core.list_commands()

    def completer(self) -> Completer:
        return SlashCommandCompleter(self._core.completion_items())

    def print_help(self, console: Console) -> None:
        table = Table("Command", "Description", "Usage")
        for command in self._core.list_commands():
            table.add_row(command.trigger, command.description, command.usage)
        console.print(table)

    async def execute(self, raw_message: str, context: InteractiveCommandContext) -> SlashCommandResult:
        policy = context.confirm_policy
        if policy is None:
            policy = getattr(context.renderer, "confirm_policy", None) or ConfirmPolicy()
            context.confirm_policy = policy

        port = CliInteractivePort(context)
        command_ctx = CommandContext(
            session=context.agent_session,
            cwd=context.cwd,
            port=port,
            raw=raw_message,
            args="",
            confirm_policy=policy,
            extras={"extension_context": context},
        )
        result = await self._core.execute(raw_message, command_ctx)

        return SlashCommandResult(
            session=result.session,
            exit_requested=result.exit_requested,
            clear_requested=result.clear_requested,
            handled=result.handled,
            prompt=result.followup_prompt,
            replay_session_history=result.replay_session_history,
            views=list(result.views),
        )


async def _select_from_choices(
    context: InteractiveCommandContext,
    *,
    title: str,
    columns: tuple[str, ...],
    choices: Sequence[tuple[str, object, object]],
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
        table.add_row(str(index), str(label), str(description))
    context.console.print(table)
    if not choices:
        return None

    if context.input_session is None:
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
    choices: Sequence[tuple[str, object, object]],
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
        fragments: StyleAndTextTuples = [
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

        def create_margin(self, window_render_info, width: int, height: int) -> StyleAndTextTuples:
            if len(choices) <= visible_choice_count:
                return [("", " \n") for _ in range(height)]
            scrollbar_height = max(1, int(height * visible_choice_count / len(choices)))
            max_scroll = max(1, len(choices) - visible_choice_count)
            scrollbar_top = int((height - scrollbar_height) * top_index / max_scroll)
            rows: StyleAndTextTuples = []
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


def _print_session_history(console: Console, messages: list) -> None:
    """Render session messages like a live conversation replay."""
    from spice.agent.tool_results import tool_display_text

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
                _render_history_markdown(console, content)
                console.print()
            if message.tool_calls:
                for call in message.tool_calls:
                    console.print(f"  [magenta]tool: {call.name}()[/magenta]")
        elif message.role == "tool":
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


# Compatibility aliases for TUI imports.
_entry_preview = entry_preview
_entry_active_label = entry_active_label
_sustained_goal_prompt = sustained_goal_prompt
