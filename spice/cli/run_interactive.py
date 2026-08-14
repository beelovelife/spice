"""Interactive conversation runner for the Spice CLI."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.filters import has_completions
from prompt_toolkit.styles import Style
from rich.console import Console

from spice.agent.agent_session import AgentSession
from spice.cli.commands import (
    InteractiveCommandContext,
    SlashCommandRegistry,
    _print_session_history,
    _select_from_choices,
    paint_command_result,
)
from spice.interactive.commands import is_execute_request
from spice.interactive.types import CommandResult
from spice.cli.completion import SpiceInputCompleter, create_completion_key_bindings
from spice.cli.render import CliRenderer
from spice.cli.terminal import enable_cursor_blink_after_render, preserve_cursor_blink
from spice.cli.welcome import print_compact_welcome, print_welcome as print_welcome


COMPLETION_MENU_RESERVED_ROWS = 16
_pt_prompt = importlib.import_module("prompt_toolkit.shortcuts.prompt")


class UpwardCompletionPromptSession(PromptSession[str]):
    """Prompt session that renders completion menus above the input line."""

    def _get_default_buffer_control_height(self):
        return _pt_prompt.Dimension()

    def _create_layout(self):
        dyncond = self._dyncond
        has_before_fragments, get_prompt_text_1, get_prompt_text_2 = (
            _pt_prompt._split_multiline_prompt(self._get_prompt)
        )
        default_buffer = self.default_buffer
        search_buffer = self.search_buffer

        @_pt_prompt.Condition
        def display_placeholder() -> bool:
            return self.placeholder is not None and self.default_buffer.text == ""

        all_input_processors = [
            _pt_prompt.HighlightIncrementalSearchProcessor(),
            _pt_prompt.HighlightSelectionProcessor(),
            _pt_prompt.ConditionalProcessor(
                _pt_prompt.AppendAutoSuggestion(),
                _pt_prompt.has_focus(default_buffer) & ~_pt_prompt.is_done,
            ),
            _pt_prompt.ConditionalProcessor(
                _pt_prompt.PasswordProcessor(), dyncond("is_password")
            ),
            _pt_prompt.DisplayMultipleCursors(),
            _pt_prompt.DynamicProcessor(
                lambda: _pt_prompt.merge_processors(self.input_processors or [])
            ),
            _pt_prompt.ConditionalProcessor(
                _pt_prompt.AfterInput(lambda: self.placeholder),
                filter=display_placeholder,
            ),
        ]

        bottom_toolbar = _pt_prompt.ConditionalContainer(
            _pt_prompt.Window(
                _pt_prompt.FormattedTextControl(
                    lambda: self.bottom_toolbar, style="class:bottom-toolbar.text"
                ),
                style="class:bottom-toolbar",
                dont_extend_height=True,
                height=_pt_prompt.Dimension(min=1),
            ),
            filter=_pt_prompt.Condition(lambda: self.bottom_toolbar is not None)
            & ~_pt_prompt.is_done
            & _pt_prompt.renderer_height_is_known,
        )

        search_toolbar = _pt_prompt.SearchToolbar(
            search_buffer, ignore_case=dyncond("search_ignore_case")
        )
        search_buffer_control = _pt_prompt.SearchBufferControl(
            buffer=search_buffer,
            input_processors=[_pt_prompt.ReverseSearchProcessor()],
            ignore_case=dyncond("search_ignore_case"),
        )
        system_toolbar = _pt_prompt.SystemToolbar(
            enable_global_bindings=dyncond("enable_system_prompt")
        )

        def get_search_buffer_control():
            if _pt_prompt.is_true(self.multiline):
                return search_toolbar.control
            return search_buffer_control

        default_buffer_control = _pt_prompt.BufferControl(
            buffer=default_buffer,
            search_buffer_control=get_search_buffer_control,
            input_processors=all_input_processors,
            include_default_input_processors=False,
            lexer=_pt_prompt.DynamicLexer(lambda: self.lexer),
            preview_search=True,
        )
        default_buffer_window = _pt_prompt.Window(
            default_buffer_control,
            height=self._get_default_buffer_control_height,
            get_line_prefix=_pt_prompt.partial(
                self._get_line_prefix, get_prompt_text_2=get_prompt_text_2
            ),
            wrap_lines=dyncond("wrap_lines"),
        )

        @_pt_prompt.Condition
        def multi_column_complete_style() -> bool:
            return self.complete_style == _pt_prompt.CompleteStyle.MULTI_COLUMN

        completion_space_filter = (
            _pt_prompt.has_focus(default_buffer) & has_completions & ~_pt_prompt.is_done
        )
        main_input_content = _pt_prompt.HSplit(
            [
                _pt_prompt.ConditionalContainer(
                    _pt_prompt.Window(
                        height=_pt_prompt.Dimension(
                            min=1,
                            preferred=COMPLETION_MENU_RESERVED_ROWS,
                            max=COMPLETION_MENU_RESERVED_ROWS,
                        )
                    ),
                    filter=completion_space_filter,
                ),
                _pt_prompt.ConditionalContainer(
                    _pt_prompt.Window(
                        _pt_prompt.FormattedTextControl(get_prompt_text_1),
                        dont_extend_height=True,
                    ),
                    _pt_prompt.Condition(has_before_fragments),
                ),
                _pt_prompt.ConditionalContainer(
                    default_buffer_window,
                    _pt_prompt.Condition(
                        lambda: (
                            _pt_prompt.get_app().layout.current_control
                            != search_buffer_control
                        )
                    ),
                ),
                _pt_prompt.ConditionalContainer(
                    _pt_prompt.Window(search_buffer_control),
                    _pt_prompt.Condition(
                        lambda: (
                            _pt_prompt.get_app().layout.current_control
                            == search_buffer_control
                        )
                    ),
                ),
            ]
        )

        main_input_container = _pt_prompt.FloatContainer(
            main_input_content,
            [
                _pt_prompt.Float(
                    xcursor=True,
                    ycursor=True,
                    transparent=True,
                    content=_pt_prompt.CompletionsMenu(
                        max_height=COMPLETION_MENU_RESERVED_ROWS,
                        scroll_offset=1,
                        extra_filter=_pt_prompt.has_focus(default_buffer)
                        & ~multi_column_complete_style,
                    ),
                ),
                _pt_prompt.Float(
                    xcursor=True,
                    ycursor=True,
                    transparent=True,
                    content=_pt_prompt.MultiColumnCompletionsMenu(
                        show_meta=True,
                        extra_filter=_pt_prompt.has_focus(default_buffer)
                        & multi_column_complete_style,
                    ),
                ),
                _pt_prompt.Float(
                    right=0,
                    top=0,
                    hide_when_covering_content=True,
                    content=_pt_prompt._RPrompt(lambda: self.rprompt),
                ),
            ],
        )

        layout = _pt_prompt.HSplit(
            [
                _pt_prompt.ConditionalContainer(
                    _pt_prompt.Frame(main_input_container),
                    filter=dyncond("show_frame"),
                    alternative_content=main_input_container,
                ),
                _pt_prompt.ConditionalContainer(
                    _pt_prompt.ValidationToolbar(), filter=~_pt_prompt.is_done
                ),
                _pt_prompt.ConditionalContainer(
                    system_toolbar,
                    dyncond("enable_system_prompt") & ~_pt_prompt.is_done,
                ),
                _pt_prompt.ConditionalContainer(
                    _pt_prompt.Window(
                        _pt_prompt.FormattedTextControl(self._get_arg_text), height=1
                    ),
                    dyncond("multiline") & _pt_prompt.has_arg,
                ),
                _pt_prompt.ConditionalContainer(
                    search_toolbar, dyncond("multiline") & ~_pt_prompt.is_done
                ),
                bottom_toolbar,
            ]
        )

        return _pt_prompt.Layout(layout, default_buffer_window)


def prompt_style() -> Style:
    return Style.from_dict(
        {
            "prompt": "#67e8f9 bold",
            "status": "bg:#111827 #cbd5e1",
            "status.plan": "bg:#111827 #5eead4 bold",
            "status.sep": "bg:#111827 #64748b",
            "completion-menu": "bg:#111827 #ffffff",
            "completion-menu.completion": "bg:#111827 #ffffff",
            "completion-menu.completion.current": "bg:#4338ca #ffffff bold",
            "completion-menu.meta": "bg:#111827 #ffffff",
            "completion-menu.meta.completion": "bg:#111827 #ffffff",
            "completion-menu.meta.completion.current": "bg:#4338ca #ffffff",
            "completion-menu.multi-column-meta": "bg:#111827 #ffffff",
            "completion.command": "#67e8f9 bold",
            "completion.args": "#a3a3a3",
            "completion.description": "#ffffff",
            "completion.directory": "#60a5fa bold",
            "completion.file": "#bbf7d0",
            "completion.description.directory": "#93c5fd",
            "completion.description.file": "#a7f3d0",
            "scrollbar.background": "bg:#111827",
            "scrollbar.button": "bg:#475569",
        }
    )


async def run_prompt(session: AgentSession, renderer: CliRenderer, prompt: str) -> None:
    refresh_task = asyncio.create_task(_refresh_waiting_indicator(renderer))
    try:
        async for event in session.prompt(prompt):
            renderer.render_event(event)
    finally:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task
        renderer.finish_response()


async def render_prompt_interruptible(
    session: AgentSession, renderer: CliRenderer, prompt: str, console: Console
) -> bool:
    """Render one prompt and turn Ctrl+C cancellation into a reusable CLI session."""
    try:
        await run_prompt(session, renderer, prompt)
        return True
    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print(
            "\n[yellow]Current model/tool execution interrupted. You can enter a new instruction.[/yellow]"
        )
        return False


async def _refresh_waiting_indicator(renderer: CliRenderer) -> None:
    while True:
        await asyncio.sleep(0.125)
        renderer.refresh_waiting_indicator()


def _bottom_toolbar(
    session: AgentSession, renderer: CliRenderer, *, show_edit_hint: bool = False
):
    mode = session.plan_state.mode
    if mode == "plan":
        return [
            ("class:status.plan", " plan mode "),
            ("class:status.sep", " | "),
            ("class:status", "shift+tab"),
        ]
    if show_edit_hint:
        return [
            ("class:status", " edit mode "),
            ("class:status.sep", " | "),
            ("class:status", "shift+tab"),
        ]
    return None


async def _handle_plan_ready(
    *,
    console: Console,
    input_session: PromptSession[str],
    renderer: CliRenderer,
    agent_session: AgentSession,
) -> None:
    if not agent_session.plan_state.is_plan_mode or not agent_session.plan_state.steps:
        return
    choice = await _select_from_choices(
        InteractiveCommandContext(
            console=console,
            input_session=input_session,
            renderer=renderer,
            agent_session=agent_session,
            cwd=Path.cwd(),
        ),
        title="Plan ready",
        columns=("Action", "Details"),
        choices=[
            ("auto", "Yes, and use auto mode", "allow file edits for this session"),
            ("refine", "Tell Spice what to change", "stay in plan mode"),
        ],
        interactive=True,
    )
    if choice == "auto":
        renderer.allow_file_edits_for_session()
        prompt = agent_session.approve_plan("auto")
        console.print(
            "[green]Switched to edit mode.[/green] Executing with file edits auto-approved for this session."
        )
        await render_prompt_interruptible(agent_session, renderer, prompt, console)


async def run_conversation(
    console: Console,
    *,
    provider: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    continue_session: bool = False,
    markdown: bool = True,
    trace: bool = False,
    trace_file: Path | None = None,
) -> None:
    if not sys.stdin.isatty():
        console.print(
            "[bold red]Error:[/bold red] interactive mode requires a terminal stdin."
        )
        console.print(
            "Use [bold]spice run <prompt>[/bold] for non-interactive execution."
        )
        return

    preserve_cursor_blink()
    print_compact_welcome(console)
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
        return
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return
    console.print(f"[dim]Session: {agent_session.session_label}[/dim]")
    trace_writer = None
    if trace:
        from spice.agent.trace import attach_trace_writer

        trace_writer, trace_path = attach_trace_writer(agent_session, path=trace_file)
        console.print(f"[dim]Trace: {trace_path}[/dim]")
    if agent_session.extensions.errors:
        for error in agent_session.extensions.errors:
            console.print(f"[yellow]Extension load failed:[/yellow] {error}")
    commands = SlashCommandRegistry(agent_session.extensions)
    key_bindings = create_completion_key_bindings()
    show_edit_hint = False

    @key_bindings.add("s-tab")
    def _toggle_plan_mode(event) -> None:
        nonlocal show_edit_hint
        mode = agent_session.toggle_interaction_mode()
        show_edit_hint = mode == "edit"
        _sync_bottom_toolbar()
        event.app.invalidate()

    input_session: PromptSession[str] = UpwardCompletionPromptSession(
        cursor=CursorShape.BLINKING_BLOCK,
        completer=SpiceInputCompleter(commands.completer(), cwd=Path.cwd()),
        complete_while_typing=True,
        key_bindings=key_bindings,
        reserve_space_for_menu=COMPLETION_MENU_RESERVED_ROWS,
        style=prompt_style(),
        mouse_support=has_completions,
    )
    prompt_app = getattr(input_session, "app", None)
    if prompt_app is not None:
        prompt_app.after_render += enable_cursor_blink_after_render

    def _sync_bottom_toolbar() -> None:
        if agent_session.plan_state.mode == "plan" or show_edit_hint:
            input_session.bottom_toolbar = lambda: _bottom_toolbar(
                agent_session,
                renderer,
                show_edit_hint=show_edit_hint,
            )
        else:
            input_session.bottom_toolbar = None

    while True:
        try:
            _sync_bottom_toolbar()
            message = (await input_session.prompt_async("spice ❯ ")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not message:
            continue
        if message in {"clear", "cls"}:
            message = "/clear"
        if agent_session.plan_state.mode != "plan":
            show_edit_hint = False
            _sync_bottom_toolbar()
        if message in {"exit", "quit"}:
            break
        if message.startswith("/"):
            result = await commands.execute(
                message,
                InteractiveCommandContext(
                    console=console,
                    input_session=input_session,
                    renderer=renderer,
                    agent_session=agent_session,
                    cwd=Path.cwd(),
                    confirm_policy=renderer.confirm_policy,
                ),
            )
            previous_session = agent_session
            new_session = result.session
            replacing_session = (
                new_session is not None and new_session is not previous_session
            )
            if new_session is not None and new_session is not previous_session:
                renderer.reset_confirm_policy()
                set_confirm = getattr(new_session, "set_confirm", None)
                if set_confirm is not None:
                    set_confirm(renderer.confirm)
                else:
                    new_session.confirm = renderer.confirm
                agent_session = new_session
                if trace_writer is not None:
                    trace_writer.bind(agent_session)
                commands = SlashCommandRegistry(agent_session.extensions)
                input_session.completer = SpiceInputCompleter(
                    commands.completer(), cwd=Path.cwd()
                )
                close = getattr(previous_session, "aclose", None)
                if close is not None:
                    await close()
            if result.clear_requested:
                console.clear()
                print_compact_welcome(console)
                console.print(f"[dim]Session: {agent_session.session_label}[/dim]")
                if agent_session.extensions.errors:
                    for error in agent_session.extensions.errors:
                        console.print(
                            f"[yellow]Extension load failed:[/yellow] {error}"
                        )
            paint_command_result(
                console,
                CommandResult(views=result.views),
            )
            if replacing_session:
                console.print(f"[dim]Session: {agent_session.session_label}[/dim]")
            if result.replay_session_history:
                history_messages = [
                    message
                    for message in agent_session.messages
                    if message.role != "system"
                ]
                _print_session_history(console, history_messages)
            if result.prompt:
                await render_prompt_interruptible(
                    agent_session, renderer, result.prompt, console
                )
                await _handle_plan_ready(
                    console=console,
                    input_session=input_session,
                    renderer=renderer,
                    agent_session=agent_session,
                )
            if result.exit_requested:
                break
            if result.handled:
                continue

            console.print(f"[red]Unknown command:[/red] {message}")
            continue

        if agent_session.plan_state.is_plan_mode and is_execute_request(message):
            prompt = agent_session.approve_plan("manual")
            console.print(
                "[green]Switched to edit mode.[/green] Executing the approved plan."
            )
            console.print()
            await render_prompt_interruptible(agent_session, renderer, prompt, console)
            continue

        console.print()
        await render_prompt_interruptible(agent_session, renderer, message, console)
        await _handle_plan_ready(
            console=console,
            input_session=input_session,
            renderer=renderer,
            agent_session=agent_session,
        )
    close = getattr(agent_session, "aclose", None)
    if close is not None:
        await close()
