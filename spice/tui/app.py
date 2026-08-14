"""Minimal full-screen TUI for Spice built on prompt_toolkit.

Layout:

    +------------------------------------------+
    |          message area (scrollable)       |
    |                                          |
    +------------------------------------------+
    |  spice ❯  fixed input box                |
    +------------------------------------------+

It reuses ``AgentSession.prompt()`` and consumes the same ``AgentEvent``
stream as the regular CLI. Rendering is intentionally plain text for the
first version; styling can be layered on later.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from rich import box
from rich.console import Console
from rich.table import Table

from spice.agent.agent_session import AgentSession
from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    RoundCompleteEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.cli.commands import SlashCommandRegistry
from spice.cli.completion import (
    SpiceInputCompleter,
    accept_completion,
    insert_text_and_maybe_complete,
    start_or_accept_completion,
)
from spice.cli.render import (
    COMPACTABLE_TOOL_NAMES,
    CompactToolGroup,
    _is_table_divider,
    _looks_like_table_line,
    _split_table_cells,
    _preview_text,
    _todo_items_from_result,
    _todo_marker,
    format_tool_end,
    format_tool_start,
)
from spice.interactive.activity import activity_parts, activity_text
from spice.interactive.commands import is_execute_request
from spice.interactive.confirm import (
    ConfirmPolicy,
    is_file_edit_tool,
)
from spice.interactive.sessions import (
    SESSION_PICKER_LIMIT,
    entry_active_label,
    entry_preview,
)
from spice.interactive.types import (
    ChoiceItem,
    ChoiceRequest,
    CommandContext,
    CommandResult,
    ConfirmRequest,
    PanelView,
    TableView,
    TextView,
)
from spice.cli.terminal import enable_cursor_blink_after_render
from spice.cli.welcome import WELCOME_QUOTES
from spice.llm.config import get_api_key, load_config, save_config
from spice.llm.models import Model
from spice.llm.model_registry import ModelRegistry
from spice.skills.loader import load_skills, read_skill_content
from spice.storage.factory import create_session_store
from spice.tools.tool_registry import READ_ONLY_TOOLS, TOOLSETS, create_all_tools


class _TuiConsole:
    def __init__(self, owner: "SpiceTUI") -> None:
        self.owner = owner

    def print(self, *objects, **kwargs) -> None:
        output = StringIO()
        console = Console(
            file=output, force_terminal=False, color_system=None, width=100
        )
        console.print(*objects, **kwargs)
        self.owner._append(output.getvalue())


class _TuiInteractivePort:
    def __init__(self, owner: "SpiceTUI") -> None:
        self.owner = owner

    async def choose(self, request: ChoiceRequest) -> str | None:
        return await self.owner._port_choose(request)

    async def confirm(self, request: ConfirmRequest) -> str:
        return await self.owner._port_confirm(request)


class _TuiToolConfirmPort(_TuiInteractivePort):
    """Render one tool preview while ConfirmPolicy serializes the modal."""

    def __init__(self, owner: "SpiceTUI", tool_name: str, args: dict) -> None:
        super().__init__(owner)
        self.tool_name = tool_name
        self.args = args

    async def confirm(self, request: ConfirmRequest) -> str:
        self.owner._ensure_newline()
        if not is_file_edit_tool(self.tool_name):
            args_preview, _ = _preview_text(str(self.args), max_chars=700, max_lines=8)
            self.owner._append(f"\nConfirm {self.tool_name}\n{args_preview}\n")
        return await self.owner._port_confirm(request)


def _compact_blank_lines(text: str, *, max_blank_lines: int = 1) -> str:
    lines = text.splitlines()
    compacted: list[str] = []
    blank_count = 0
    for raw_line in lines:
        line = raw_line.rstrip()
        blank = not line.strip()
        if blank:
            blank_count += 1
            if blank_count > max_blank_lines:
                continue
        else:
            blank_count = 0
        compacted.append(line)
    return "\n".join(compacted)


def _render_plain_markdown(text: str) -> str:
    rendered: list[str] = []
    in_code_block = False
    prev_kind: str | None = None  # "text", "list", "heading", "code"
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            prev_kind = "code"
            continue
        if in_code_block:
            rendered.append(f"    {line}" if line else "")
            prev_kind = "code"
            continue
        if not stripped:
            if prev_kind == "list":
                continue
            rendered.append("")
            prev_kind = None
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            if rendered and rendered[-1].strip():
                rendered.append("")
            rendered.append(_strip_inline_markdown(heading.group(2)))
            rendered.append("")
            prev_kind = "heading"
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            rendered.append(f" \u2022 {_strip_inline_markdown(bullet.group(1))}")
            prev_kind = "list"
            continue
        numbered = re.match(r"^(\d+[.)])\s+(.*)$", stripped)
        if numbered:
            rendered.append(
                f" {numbered.group(1)} {_strip_inline_markdown(numbered.group(2))}"
            )
            prev_kind = "list"
            continue
        if prev_kind == "list" and rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append(_strip_inline_markdown(line))
        prev_kind = "text"
    return _compact_blank_lines("\n".join(rendered))


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    return text


def _render_plain_table(lines: list[str], *, width: int) -> str:
    header = _split_table_cells(lines[0])
    rows = [_split_table_cells(line) for line in lines[2:]]
    column_count = (
        max(len(header), *(len(row) for row in rows)) if rows else len(header)
    )
    header += [""] * (column_count - len(header))
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=width)
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=False)
    for column in header:
        table.add_column(_strip_inline_markdown(column), overflow="fold")
    for row in rows:
        cells = row + [""] * (column_count - len(row))
        table.add_row(*(_strip_inline_markdown(cell) for cell in cells))
    console.print(table)
    return output.getvalue().rstrip()


class _MessageBufferControl(FormattedTextControl):
    """Read-only message control with explicit display-line wrapping."""

    def __init__(self, tui: "SpiceTUI", **kwargs):
        kwargs.pop("buffer", None)
        kwargs.pop("input_processors", None)
        kwargs.setdefault("focusable", False)
        super().__init__(
            text=tui._message_fragments,
            show_cursor=False,
            get_cursor_position=tui._message_cursor_position,
            **kwargs,
        )
        self._tui = tui

    def create_content(self, width: int, height: int | None):
        self._tui._message_display_width = width
        if isinstance(height, int) and height > 0:
            self._tui._message_display_height = height
        self._tui._sync_message_scroll()
        return super().create_content(width, height)

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._tui._scroll_active_view(-3)
            self._tui._app.invalidate()
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._tui._scroll_active_view(3)
            self._tui._app.invalidate()
            return None
        return super().mouse_handler(mouse_event)


class _InputBufferControl(BufferControl):
    """Input control that lets wheel scrolling continue to move chat history."""

    def __init__(self, tui: "SpiceTUI", **kwargs):
        super().__init__(**kwargs)
        self._tui = tui

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._tui._scroll_active_view(-3)
            self._tui._app.invalidate()
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._tui._scroll_active_view(3)
            self._tui._app.invalidate()
            return None
        return super().mouse_handler(mouse_event)


class SpiceTUI:
    """Full-screen TUI with a fixed bottom input area."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        continue_session: bool = False,
        trace: bool = False,
        trace_file: Path | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._session_id = session_id
        self._continue_session = continue_session
        self._trace = trace
        self._trace_file = trace_file
        self._trace_writer = None
        self._welcome_quote_en, self._welcome_quote_zh = random.choice(WELCOME_QUOTES)
        self._agent_session: AgentSession | None = None
        self._busy = False
        self._streaming = False
        self._pending_task: asyncio.Task | None = None
        self.confirm_policy = ConfirmPolicy()
        self._show_edit_mode_hint = False
        self._confirmation_future: asyncio.Future[str | None] | None = None
        self._confirmation_question = ""
        self._confirmation_choices: list[tuple[str, str, str]] = []
        self._confirmation_index = 0
        self._confirmation_start = 0
        self._model_choices: list[Model] = []
        self._model_choice_index = 0
        self._model_choice_start = 0
        self._session_choices: list[str] = []
        self._session_choice_rows: list[tuple[str, str, str]] = []
        self._session_choice_index = 0
        self._session_choice_start = 0
        self._session_choice_mode = "resume"
        self._rewind_choices: list[str] = []
        self._rewind_choice_rows: list[tuple[str, str]] = []
        self._rewind_choice_index = 0
        self._rewind_choice_start = 0
        self._message_follow_tail = True
        self._message_vertical_scroll = 0
        self._message_display_width: int | None = None
        self._message_display_height: int | None = None
        self._response_start = 0
        self._response_parts: list[str] = []
        self._waiting_started: float | None = None
        self._waiting_start: int | None = None
        self._waiting_label = "Waiting for model"
        self._reasoning_streaming = False
        self._activity_started: float | None = None
        self._last_stream_delta_at: float | None = None
        self._active_tool_name: str | None = None
        self._pending_tool_args: dict[str, dict] = {}
        self._compact_tool_group: CompactToolGroup | None = None
        self._todo_items: list[dict] = []
        self._todo_summary: dict[str, int] = {}

        self._message_buffer: Buffer = Buffer(read_only=True)
        self._input_buffer: Buffer = Buffer(
            complete_while_typing=True,
            multiline=False,
            accept_handler=self._on_accept,
        )

        self._app: Application = self._build_app()

    @property
    def _allow_file_edits(self) -> bool:
        return self.confirm_policy.allow_file_edits

    @_allow_file_edits.setter
    def _allow_file_edits(self, value: bool) -> None:
        self.confirm_policy.allow_file_edits = value

    @property
    def _allow_all_tools(self) -> bool:
        return self.confirm_policy.allow_all_tools

    @_allow_all_tools.setter
    def _allow_all_tools(self, value: bool) -> None:
        self.confirm_policy.allow_all_tools = value

    @property
    def _allow_read_only_bash(self) -> bool:
        return self.confirm_policy.allow_read_only_bash

    @_allow_read_only_bash.setter
    def _allow_read_only_bash(self, value: bool) -> None:
        self.confirm_policy.allow_read_only_bash = value

    # ------------------------------------------------------------------ UI

    def _build_app(self) -> Application:
        kb = KeyBindings()
        has_confirmation = Condition(lambda: self._confirmation_future is not None)
        has_model_selector = Condition(lambda: bool(self._model_choices))
        has_session_selector = Condition(lambda: bool(self._session_choices))
        has_rewind_selector = Condition(lambda: bool(self._rewind_choices))
        is_busy = Condition(lambda: self._busy)
        has_overlay = Condition(
            lambda: (
                self._confirmation_future is not None
                or bool(self._model_choices)
                or bool(self._session_choices)
                or bool(self._rewind_choices)
            )
        )

        @kb.add("c-c", filter=~has_overlay)
        @kb.add("c-d")
        def _quit(event) -> None:
            if self._pending_task and not self._pending_task.done():
                self._pending_task.cancel()
                return
            event.app.exit()

        @kb.add("escape", filter=~has_overlay & is_busy)
        def _interrupt_current(event) -> None:
            if self._pending_task and not self._pending_task.done():
                self._pending_task.cancel()
                event.app.invalidate()

        @kb.add("pageup")
        def _page_up(event) -> None:
            if self._move_model_choice(-self._selector_page_size()):
                event.app.invalidate()
                return
            if self._move_session_choice(-self._selector_page_size(row_height=2)):
                event.app.invalidate()
                return
            if self._move_rewind_choice(-self._selector_page_size()):
                event.app.invalidate()
                return
            if self._move_confirmation(-1):
                event.app.invalidate()
                return
            self._scroll_messages(-self._message_page_size())
            event.app.invalidate()

        @kb.add("pagedown")
        def _page_down(event) -> None:
            if self._move_model_choice(self._selector_page_size()):
                event.app.invalidate()
                return
            if self._move_session_choice(self._selector_page_size(row_height=2)):
                event.app.invalidate()
                return
            if self._move_rewind_choice(self._selector_page_size()):
                event.app.invalidate()
                return
            if self._move_confirmation(1):
                event.app.invalidate()
                return
            self._scroll_messages(self._message_page_size())
            event.app.invalidate()

        @kb.add("c-up")
        def _line_up(event) -> None:
            self._scroll_messages(-3)
            event.app.invalidate()

        @kb.add("c-down")
        def _line_down(event) -> None:
            self._scroll_messages(3)
            event.app.invalidate()

        @kb.add("up", filter=has_confirmation)
        def _up(event) -> None:
            if self._move_confirmation(-1):
                event.app.invalidate()

        @kb.add("down", filter=has_confirmation)
        def _down(event) -> None:
            if self._move_confirmation(1):
                event.app.invalidate()

        @kb.add("enter", filter=has_confirmation)
        def _enter(event) -> None:
            if self._accept_confirmation():
                event.app.invalidate()

        @kb.add("escape", filter=has_confirmation)
        def _escape(event) -> None:
            if self._cancel_confirmation():
                event.app.invalidate()

        @kb.add("up", filter=has_model_selector)
        def _model_up(event) -> None:
            if self._move_model_choice(-1):
                event.app.invalidate()

        @kb.add("down", filter=has_model_selector)
        def _model_down(event) -> None:
            if self._move_model_choice(1):
                event.app.invalidate()

        @kb.add("enter", filter=has_model_selector)
        def _model_enter(event) -> None:
            if self._accept_model_choice():
                event.app.invalidate()

        @kb.add("escape", filter=has_model_selector)
        def _model_escape(event) -> None:
            if self._cancel_model_choice():
                event.app.invalidate()

        @kb.add("up", filter=has_session_selector)
        def _session_up(event) -> None:
            if self._move_session_choice(-1):
                event.app.invalidate()

        @kb.add("down", filter=has_session_selector)
        def _session_down(event) -> None:
            if self._move_session_choice(1):
                event.app.invalidate()

        @kb.add("enter", filter=has_session_selector)
        def _session_enter(event) -> None:
            if self._accept_session_choice():
                event.app.invalidate()

        @kb.add("escape", filter=has_session_selector)
        def _session_escape(event) -> None:
            if self._cancel_session_choice():
                event.app.invalidate()

        @kb.add("up", filter=has_rewind_selector)
        def _rewind_up(event) -> None:
            if self._move_rewind_choice(-1):
                event.app.invalidate()

        @kb.add("down", filter=has_rewind_selector)
        def _rewind_down(event) -> None:
            if self._move_rewind_choice(1):
                event.app.invalidate()

        @kb.add("enter", filter=has_rewind_selector)
        def _rewind_enter(event) -> None:
            if self._accept_rewind_choice():
                event.app.invalidate()

        @kb.add("escape", filter=has_rewind_selector)
        def _rewind_escape(event) -> None:
            if self._cancel_rewind_choice():
                event.app.invalidate()

        @kb.add("c-c", filter=has_overlay)
        def _cancel_overlay(event) -> None:
            if (
                self._cancel_confirmation()
                or self._cancel_model_choice()
                or self._cancel_session_choice()
                or self._cancel_rewind_choice()
            ):
                event.app.invalidate()

        not_overlay = ~has_overlay

        @kb.add("up", filter=not_overlay & has_completions)
        def _complete_up(event) -> None:
            event.current_buffer.complete_previous()

        @kb.add("down", filter=not_overlay & has_completions)
        def _complete_down(event) -> None:
            event.current_buffer.complete_next()

        @kb.add("tab", filter=not_overlay)
        def _tab(event) -> None:
            start_or_accept_completion(event.current_buffer)

        @kb.add("s-tab", filter=not_overlay)
        def _shift_tab(event) -> None:
            buf = event.current_buffer
            if buf.complete_state:
                buf.complete_previous()
            else:
                self._toggle_interaction_mode()
                event.app.invalidate()

        @kb.add("escape", filter=not_overlay & has_completions & ~is_busy)
        def _cancel_completion(event) -> None:
            event.current_buffer.cancel_completion()

        @kb.add("enter", filter=not_overlay & has_completions)
        def _accept_completion(event) -> None:
            accept_completion(event.current_buffer)

        @kb.add("@", filter=not_overlay)
        def _at_completion(event) -> None:
            insert_text_and_maybe_complete(event.current_buffer, "@")

        @kb.add("/", filter=not_overlay)
        def _slash_completion(event) -> None:
            insert_text_and_maybe_complete(event.current_buffer, "/")

        @kb.add("up", filter=not_overlay & ~has_completions)
        def _arrow_up(event) -> None:
            self._scroll_messages(-3)
            event.app.invalidate()

        @kb.add("down", filter=not_overlay & ~has_completions)
        def _arrow_down(event) -> None:
            self._scroll_messages(3)
            event.app.invalidate()

        self._message_control = _MessageBufferControl(
            self,
        )
        self._message_window = Window(
            content=self._message_control,
            wrap_lines=False,
            always_hide_cursor=True,
            get_vertical_scroll=lambda _window: self._message_vertical_scroll,
            right_margins=[ScrollbarMargin(display_arrows=False)],
            style="class:message-area",
        )
        self._input_window = Window(
            content=_InputBufferControl(
                self,
                buffer=self._input_buffer,
                focus_on_click=True,
                input_processors=[BeforeInput("spice ❯ ", style="class:prompt")],
            ),
            height=3,
            wrap_lines=True,
        )
        self._todo_window = Window(
            height=8,
            content=FormattedTextControl(self._todo_fragments),
            style="class:todo",
            wrap_lines=True,
        )

        layout = Layout(
            FloatContainer(
                content=HSplit(
                    [
                        Window(
                            height=5,
                            content=FormattedTextControl(self._welcome_fragments),
                        ),
                        self._message_window,
                        ConditionalContainer(
                            content=self._todo_window,
                            filter=Condition(lambda: bool(self._todo_items)),
                        ),
                        Window(height=1, char="─", style="class:separator"),
                        self._input_window,
                        Window(
                            height=1, content=FormattedTextControl(self._mode_fragments)
                        ),
                    ]
                ),
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=8, scroll_offset=1),
                    ),
                ],
            ),
            focused_element=self._input_buffer,
        )

        style = Style.from_dict(
            {
                "separator": "#374151",
                "prompt": "#67e8f9 bold",
                "message-area scrollbar.background": "bg:#2b3347",
                "message-area scrollbar.button": "bg:#5b647c",
                "message.user-label": "#67e8f9 bold",
                "message.assistant-label": "#a7f3d0 bold",
                "message.tool": "#c4b5fd",
                "message.tool-result": "#86efac",
                "welcome.border": "#8f99b7",
                "welcome.dot": "#67e8f9 bold",
                "welcome.quote": "#f4d99d bold",
                "mode": "#8f99b7",
                "mode.plan": "#14b8a6 bold",
                "mode.edit": "#bbf7d0 bold",
                "mode.sep": "#6b7280",
                "activity": "#8f99b7",
                "activity.marker": "#67e8f9 bold",
                "todo": "#c7d2fe",
                "todo.title": "#67e8f9 bold",
                "todo.done": "#a7f3d0",
                "todo.active": "#f4d99d bold",
                "todo.pending": "#8f99b7",
                "todo.cancelled": "#fca5a5",
                "todo.border": "#374151",
                "thought-label": "#737b91 italic",
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
                "completion.directory": "#ffffff",
                "completion.file": "#ffffff",
                "completion.description.directory": "#ffffff",
                "completion.description.file": "#ffffff",
            }
        )

        application = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
            min_redraw_interval=0.01,
            cursor=CursorShape.BLINKING_BLOCK,
        )
        application.after_render += enable_cursor_blink_after_render
        return application

    # -------------------------------------------------------------- helpers

    def _truncate_cells(self, text: str, width: int) -> str:
        cells = 0
        result: list[str] = []
        for char in text:
            char_width = get_cwidth(char)
            if cells + char_width > width:
                break
            result.append(char)
            cells += char_width
        return "".join(result)

    def _welcome_line(self, text: str, width: int, style: str):
        text = self._truncate_cells(text, width)
        padding = width - get_cwidth(text)
        left = padding // 2
        right = padding - left
        return [
            ("class:welcome.border", "│ " + (" " * left)),
            (style, text),
            ("class:welcome.border", (" " * right) + " │\n"),
        ]

    def _welcome_fragments(self):
        quote_width = max(
            get_cwidth(self._welcome_quote_en), get_cwidth(self._welcome_quote_zh)
        )
        width = max(58, min(88, quote_width + 8))
        content_width = width - 4
        dash = "╌" * (width - 2)
        return [
            ("class:welcome.border", "╭" + dash + "╮\n"),
            *self._welcome_line(
                "• " + self._welcome_quote_en, content_width, "class:welcome.quote"
            ),
            *self._welcome_line(
                self._welcome_quote_zh, content_width, "class:welcome.quote"
            ),
            ("class:welcome.border", "╰" + dash + "╯\n"),
            ("", "\n"),
        ]

    def _mode_fragments(self):
        fragments = self._activity_fragments()
        if self._agent_session is None:
            return fragments
        mode = self._agent_session.plan_state.mode
        if mode == "plan":
            if fragments:
                fragments.append(("class:mode.sep", " | "))
            fragments.extend(
                [
                    ("class:mode.plan", " plan mode "),
                    ("class:mode.sep", " | "),
                    ("class:mode", "shift+tab"),
                ]
            )
            return fragments
        if self._show_edit_mode_hint:
            if fragments:
                fragments.append(("class:mode.sep", " | "))
            fragments.extend(
                [
                    ("class:mode", " edit mode "),
                    ("class:mode.sep", " | "),
                    ("class:mode", "shift+tab"),
                ]
            )
        return fragments

    def _activity_fragments(self):
        if not getattr(self, "_busy", False):
            return []
        now = time.monotonic()
        started = getattr(self, "_activity_started", None) or now
        elapsed_seconds = max(0.0, now - started)
        active_tool = getattr(self, "_active_tool_name", None)
        if active_tool:
            label = f"Running {active_tool}"
        elif getattr(self, "_reasoning_streaming", False):
            label = "Thinking"
        elif getattr(self, "_streaming", False):
            last_delta = getattr(self, "_last_stream_delta_at", None) or now
            label = "Waiting for model" if now - last_delta >= 2 else "Generating"
        else:
            label = getattr(self, "_waiting_label", "Waiting")
        marker, status = activity_parts(label, elapsed_seconds)
        return [
            ("class:activity.marker", f" {marker} "),
            ("class:activity", status),
        ]

    def _todo_fragments(self):
        if not self._todo_items:
            return []
        summary = self._todo_summary or {}
        total = int(summary.get("total") or len(self._todo_items))
        completed = int(summary.get("completed") or 0)
        in_progress = int(summary.get("in_progress") or 0)
        pending = int(summary.get("pending") or 0)
        cancelled = int(summary.get("cancelled") or 0)
        parts = [f"{completed}/{total} completed"]
        if in_progress:
            parts.append(f"{in_progress} running")
        if pending:
            parts.append(f"{pending} pending")
        if cancelled:
            parts.append(f"{cancelled} cancelled")
        fragments = [
            ("class:todo.border", "─" * 80 + "\n"),
            ("class:todo.title", "Todo "),
            ("class:todo", " · ".join(parts) + "\n"),
        ]
        for item in self._todo_items[:5]:
            status = str(item.get("status") or "pending")
            style = {
                "completed": "class:todo.done",
                "in_progress": "class:todo.active",
                "pending": "class:todo.pending",
                "cancelled": "class:todo.cancelled",
            }.get(status, "class:todo.pending")
            item_id = str(item.get("id") or "?")
            content = str(item.get("content") or "(no description)")
            fragments.extend(
                [(style, f"{_todo_marker(status)} {item_id}. {content}\n")]
            )
        remaining = len(self._todo_items) - 5
        if remaining > 0:
            fragments.append(("class:todo.pending", f"... {remaining} more\n"))
        return fragments

    def _append(self, text: str) -> None:
        """Append text to the read-only message buffer and scroll to bottom."""
        if not text:
            return
        new_text = self._message_buffer.text + text
        self._set_message_text(new_text)

    def _set_message_text(self, text: str) -> None:
        """Replace buffer text and keep the message viewport in sync."""
        self._message_buffer.set_document(
            Document(text, cursor_position=len(text)),
            bypass_readonly=True,
        )
        self._sync_message_scroll(text)

    def _message_lines(self, text: str | None = None) -> list[str]:
        value = self._message_buffer.text if text is None else text
        return value.split("\n") if value else [""]

    def _message_display_lines(self, text: str | None = None) -> list[str]:
        width = max(1, self._message_window_width() - 1)
        display_lines: list[str] = []
        for logical_line in self._message_lines(text):
            if not logical_line:
                display_lines.append("")
                continue
            current: list[str] = []
            current_width = 0
            for char in logical_line.expandtabs(4):
                char_width = max(0, get_cwidth(char))
                if current and char_width and current_width + char_width > width:
                    display_lines.append("".join(current))
                    current = []
                    current_width = 0
                current.append(char)
                current_width += char_width
            display_lines.append("".join(current))
        return display_lines

    def _message_fragments(self):
        fragments = []
        lines = self._message_display_lines()
        for index, line in enumerate(lines):
            suffix = "\n" if index < len(lines) - 1 else ""
            if line.startswith("You: "):
                fragments.extend(
                    [
                        ("class:message.user-label", "You:"),
                        ("", line.removeprefix("You:") + suffix),
                    ]
                )
            elif line.startswith("Spice: "):
                fragments.extend(
                    [
                        ("class:message.assistant-label", "Spice:"),
                        ("", line.removeprefix("Spice:") + suffix),
                    ]
                )
            elif line.strip().startswith("Thought for "):
                fragments.append(("class:thought-label", line + suffix))
            elif line.lstrip().startswith(("tool:", "❯")):
                fragments.append(("class:message.tool", line + suffix))
            elif line.startswith("  ") and " -> " in line:
                fragments.append(("class:message.tool-result", line + suffix))
            else:
                fragments.append(("", line + suffix))
        return fragments

    def _message_cursor_position(self) -> Point:
        last_row = max(0, len(self._message_display_lines()) - 1)
        if self._message_follow_tail:
            return Point(x=0, y=last_row)
        height = getattr(self, "_message_display_height", None)
        if not isinstance(height, int) or height <= 0:
            height = self._message_window_height()
        visible_bottom = self._message_vertical_scroll + max(1, height) - 1
        return Point(x=0, y=min(last_row, visible_bottom))

    def _message_bottom_scroll(self, text: str | None = None) -> int:
        lines = self._message_display_lines(text)
        return max(0, len(lines) - self._message_window_height())

    def _sync_message_scroll(self, text: str | None = None) -> None:
        bottom = self._message_bottom_scroll(text)
        if self._message_follow_tail:
            self._message_vertical_scroll = bottom
        else:
            self._message_vertical_scroll = max(
                0, min(self._message_vertical_scroll, bottom)
            )

    def _follow_message_tail(self) -> None:
        self._message_follow_tail = True
        self._sync_message_scroll()

    def _message_page_size(self) -> int:
        return max(4, self._message_window_height() - 2)

    def _selector_page_size(self, *, row_height: int = 1) -> int:
        return max(1, (self._message_window_height() - 6) // row_height)

    def _selector_slice(
        self,
        total: int,
        index: int,
        *,
        row_height: int = 1,
    ) -> tuple[int, int]:
        page_size = min(total, self._selector_page_size(row_height=row_height))
        start = min(max(0, index - page_size // 2), max(0, total - page_size))
        return start, start + page_size

    def _scroll_active_view(self, line_delta: int) -> None:
        step = -1 if line_delta < 0 else 1
        if getattr(self, "_model_choices", None) and self._move_model_choice(step):
            return
        if getattr(self, "_session_choices", None) and self._move_session_choice(step):
            return
        if getattr(self, "_rewind_choices", None) and self._move_rewind_choice(step):
            return
        if getattr(
            self, "_confirmation_future", None
        ) is not None and self._move_confirmation(step):
            return
        self._scroll_messages(line_delta)

    def _message_window_height(self) -> int:
        display_height = getattr(self, "_message_display_height", None)
        if isinstance(display_height, int) and display_height > 0:
            return display_height
        render_info = getattr(self._message_window, "render_info", None)
        height = getattr(render_info, "window_height", None)
        return height if isinstance(height, int) and height > 0 else 12

    def _message_window_width(self) -> int:
        display_width = getattr(self, "_message_display_width", None)
        if isinstance(display_width, int) and display_width > 0:
            return display_width
        render_info = getattr(self._message_window, "render_info", None)
        width = getattr(render_info, "window_width", None)
        return width if isinstance(width, int) and width > 0 else 100

    def _scroll_messages(self, line_delta: int) -> None:
        text = self._message_buffer.text
        if not text:
            return
        bottom = self._message_bottom_scroll(text)
        current_scroll = max(0, min(self._message_vertical_scroll, bottom))
        new_scroll = max(0, min(bottom, current_scroll + line_delta))
        if new_scroll == current_scroll:
            self._message_follow_tail = new_scroll >= bottom
            return
        self._message_vertical_scroll = new_scroll
        self._message_follow_tail = new_scroll >= bottom

    def _ensure_newline(self) -> None:
        if self._message_buffer.text and not self._message_buffer.text.endswith("\n"):
            self._append("\n")

    def _ensure_turn_gap(self) -> None:
        text = self._message_buffer.text
        if not text:
            return
        if text.endswith("\n\n"):
            return
        self._ensure_newline()
        self._append("\n")

    def _replace_from(self, start: int, text: str) -> None:
        current = self._message_buffer.text
        new_text = current[:start] + text
        self._set_message_text(new_text)

    def _start_waiting_indicator(self, label: str = "Waiting for model") -> None:
        self._waiting_started = time.monotonic()
        self._waiting_label = label
        self._waiting_start = len(self._message_buffer.text)
        self._append(self._waiting_indicator_text(0.0))

    def _render_waiting_indicator(self) -> None:
        if self._waiting_started is None or self._waiting_start is None:
            return
        elapsed = max(0.0, time.monotonic() - self._waiting_started)
        current = self._message_buffer.text
        self._set_message_text(
            current[: self._waiting_start] + self._waiting_indicator_text(elapsed)
        )
        if self._app is not None:
            self._app.invalidate()

    def _stop_waiting_indicator(self, *, clear: bool) -> None:
        if self._waiting_started is None:
            return
        if clear and self._waiting_start is not None:
            current = self._message_buffer.text
            self._set_message_text(current[: self._waiting_start])
        self._waiting_started = None
        self._waiting_start = None
        if self._app is not None:
            self._app.invalidate()

    def _waiting_indicator_text(self, elapsed: float) -> str:
        return f"{activity_text(self._waiting_label, elapsed)}\n"

    def _finish_reasoning_stream(self) -> None:
        if not getattr(self, "_reasoning_streaming", False):
            return
        self._ensure_newline()
        self._reasoning_streaming = False

    def _capture_rich(self, renderable) -> str:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=self._message_window_width(),
        )
        console.print(renderable)
        return output.getvalue()

    def _render_markdown_text(self, text: str) -> str:
        if not text.strip():
            return ""
        chunks: list[str] = []
        markdown_lines: list[str] = []
        table_lines: list[str] = []

        def flush_markdown() -> None:
            if not markdown_lines:
                return
            markdown_text = "\n".join(markdown_lines).strip("\n")
            markdown_lines.clear()
            if markdown_text.strip():
                chunks.append(_render_plain_markdown(markdown_text).rstrip())

        def flush_table() -> None:
            if not table_lines:
                return
            if len(table_lines) >= 2 and _is_table_divider(table_lines[1]):
                flush_markdown()
                chunks.append(
                    _render_plain_table(
                        table_lines, width=self._message_window_width()
                    ).rstrip()
                )
            else:
                markdown_lines.extend(table_lines)
            table_lines.clear()

        for line in text.splitlines():
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
        return (
            _compact_blank_lines(
                "\n\n".join(chunk for chunk in chunks if chunk), max_blank_lines=2
            ).rstrip()
            + "\n"
        )

    def _render_table_text(self, table: Table) -> str:
        return self._capture_rich(table)

    def _move_confirmation(self, step: int) -> bool:
        if self._confirmation_future is None or not self._confirmation_choices:
            return False
        self._confirmation_index = (self._confirmation_index + step) % len(
            self._confirmation_choices
        )
        self._render_confirmation()
        return True

    def _accept_confirmation(self) -> bool:
        if self._confirmation_future is None or not self._confirmation_choices:
            return False
        value = self._confirmation_choices[self._confirmation_index][0]
        future = self._confirmation_future
        self._confirmation_future = None
        self._confirmation_choices = []
        self._replace_from(self._confirmation_start, "")
        if not future.done():
            future.set_result(value)
        return True

    def _cancel_confirmation(self) -> bool:
        if self._confirmation_future is None:
            return False
        future = self._confirmation_future
        self._confirmation_future = None
        self._confirmation_choices = []
        self._replace_from(self._confirmation_start, "")
        if not future.done():
            future.set_result(None)
        return True

    def _render_confirmation(self) -> None:
        lines = ["", self._confirmation_question, ""]
        for index, (_, label, detail) in enumerate(self._confirmation_choices, start=1):
            arrow = "❯" if index - 1 == self._confirmation_index else " "
            suffix = f"  {detail}" if detail else ""
            lines.append(f"{arrow} {index}. {label}{suffix}")
        lines.append("")
        lines.append("Esc/Ctrl+C to cancel")
        self._replace_from(self._confirmation_start, "\n".join(lines) + "\n")

    async def confirm(self, tool_name: str, args: dict) -> bool:
        return await self.confirm_policy.confirm(
            tool_name,
            args,
            _TuiToolConfirmPort(self, tool_name, args),
        )

    async def _port_choose(self, request: ChoiceRequest) -> str | None:
        if not request.items:
            return None
        self._ensure_newline()
        self._confirmation_question = request.title
        self._confirmation_choices = [
            (item.id, item.label, item.detail) for item in request.items
        ]
        self._confirmation_index = 0
        if request.current_id:
            for index, (value, _, _) in enumerate(self._confirmation_choices):
                if value == request.current_id:
                    self._confirmation_index = index
                    break
        self._confirmation_start = len(self._message_buffer.text)
        self._confirmation_future = asyncio.get_running_loop().create_future()
        self._render_confirmation()
        self._app.invalidate()
        try:
            return await self._confirmation_future
        except asyncio.CancelledError:
            self._confirmation_future = None
            self._confirmation_choices = []
            raise

    async def _port_confirm(self, request: ConfirmRequest) -> str:
        selected = await self._port_choose(
            ChoiceRequest(
                title=request.question,
                items=tuple(
                    ChoiceItem(id=c.id, label=c.label, detail=c.detail)
                    for c in request.choices
                ),
                current_id=request.current_id,
            )
        )
        return selected or "deny"

    def _paint_command_result(self, result: CommandResult) -> None:
        for view in result.views:
            if isinstance(view, TextView):
                self._ensure_newline()
                self._append(view.text + "\n")
            elif isinstance(view, TableView):
                table = Table(*(view.columns), title=view.title)
                for row in view.rows:
                    table.add_row(*row)
                self._append_table(table)
            elif isinstance(view, PanelView):
                self._ensure_newline()
                self._append(f"[{view.title}]\n{view.body}\n")

    async def _select_plan_action(self) -> str | None:
        self._ensure_newline()
        self._confirmation_question = "Spice has written up a plan and is ready to execute. Would you like to proceed?"
        self._confirmation_choices = [
            ("auto", "Yes, and use auto mode", ""),
            ("refine", "Tell Spice what to change", "shift+tab to continue planning"),
        ]
        self._confirmation_index = 0
        self._confirmation_start = len(self._message_buffer.text)
        self._confirmation_future = asyncio.get_running_loop().create_future()
        self._render_confirmation()
        self._app.invalidate()

        try:
            return await self._confirmation_future
        except asyncio.CancelledError:
            self._confirmation_future = None
            self._confirmation_choices = []
            raise

    async def _confirm_reset(self) -> bool:
        self._ensure_newline()
        self._confirmation_question = (
            "Reset current session? This clears all messages but keeps the session id."
        )
        self._confirmation_choices = [
            ("yes", "Yes", "clear messages"),
            ("no", "No", "return to chat"),
        ]
        self._confirmation_index = 0
        self._confirmation_start = len(self._message_buffer.text)
        self._confirmation_future = asyncio.get_running_loop().create_future()
        self._render_confirmation()
        self._app.invalidate()

        try:
            selected = await self._confirmation_future
        except asyncio.CancelledError:
            self._confirmation_future = None
            self._confirmation_choices = []
            raise
        return selected == "yes"

    async def _confirm_delete_session(self, session_id: str, *, current: bool) -> bool:
        self._ensure_newline()
        suffix = " A fresh session will start." if current else ""
        self._confirmation_question = f"Delete session {session_id}? This deletes the session and all content.{suffix}"
        self._confirmation_choices = [
            ("yes", "Yes", "delete session"),
            ("no", "No", "return to chat"),
        ]
        self._confirmation_index = 0
        self._confirmation_start = len(self._message_buffer.text)
        self._confirmation_future = asyncio.get_running_loop().create_future()
        self._render_confirmation()
        self._app.invalidate()

        try:
            selected = await self._confirmation_future
        except asyncio.CancelledError:
            self._confirmation_future = None
            self._confirmation_choices = []
            raise
        return selected == "yes"

    def _start_model_selector(self) -> None:
        if self._agent_session is None:
            return
        registry = ModelRegistry()
        current = self._agent_session.model
        self._model_choices = sorted(
            registry.all(), key=lambda item: (item.provider, item.id)
        )
        self._model_choice_index = next(
            (
                index
                for index, model in enumerate(self._model_choices)
                if model.provider == current.provider and model.id == current.id
            ),
            0,
        )
        self._ensure_newline()
        self._model_choice_start = len(self._message_buffer.text)
        self._render_model_selector()

    def _move_model_choice(self, step: int) -> bool:
        if not self._model_choices:
            return False
        if abs(step) == 1:
            self._model_choice_index = (self._model_choice_index + step) % len(
                self._model_choices
            )
        else:
            self._model_choice_index = max(
                0,
                min(len(self._model_choices) - 1, self._model_choice_index + step),
            )
        self._render_model_selector()
        return True

    def _accept_model_choice(self) -> bool:
        if not self._model_choices or self._agent_session is None:
            return False
        model = self._model_choices[self._model_choice_index]
        self._model_choices = []
        self._replace_from(self._model_choice_start, "")
        self._set_model(model)
        return True

    def _cancel_model_choice(self) -> bool:
        if not self._model_choices:
            return False
        self._model_choices = []
        self._replace_from(self._model_choice_start, "")
        self._append("Model selection cancelled.\n")
        return True

    def _render_model_selector(self) -> None:
        if self._agent_session is None:
            return
        current = self._agent_session.model
        lines = ["", "Select model (up/down, enter, esc)", ""]
        start, end = self._selector_slice(
            len(self._model_choices), self._model_choice_index
        )
        if start:
            lines.append(f"  ↑ {start} more")
        for index in range(start, end):
            model = self._model_choices[index]
            arrow = "❯" if index == self._model_choice_index else " "
            active = (
                "*"
                if model.provider == current.provider and model.id == current.id
                else " "
            )
            key_status = (
                "key ok"
                if get_api_key(model.provider, env_names=model.api_key_envs)
                else "missing key"
            )
            provider = model.provider_name or model.provider
            lines.append(
                f"{arrow} {active} {model.provider}/{model.id}  {provider}  {key_status}"
            )
        if end < len(self._model_choices):
            lines.append(f"  ↓ {len(self._model_choices) - end} more")
        lines.append("")
        self._replace_from(self._model_choice_start, "\n".join(lines) + "\n")

    def _set_model(self, model: Model) -> None:
        if self._agent_session is None:
            return
        self._agent_session.set_model(model)
        config = load_config()
        config.default_model = model.profile_key or model.id
        config.provider = model.provider
        config.model = model.id
        config.protocol = model.protocol
        config.base_url = model.base_url
        if model.temperature is not None:
            config.temperature = model.temperature
        save_config(config)
        self._ensure_newline()
        self._append(f"Set model to {model.provider}/{model.id}\n")

    def _start_session_selector(self, *, mode: str = "resume") -> None:
        store = create_session_store(load_config(), cwd=Path.cwd())
        rows = store.list(
            limit=SESSION_PICKER_LIMIT, cwd=Path.cwd(), include_empty=True
        )
        if not rows:
            self._ensure_newline()
            self._append("No sessions found for this workspace.\n")
            return
        self._session_choice_mode = mode
        self._session_choices = [row.id for row in rows]
        self._session_choice_rows = [
            (
                row.id,
                f"{row.provider}/{row.model}",
                f"{row.message_count} messages  {row.updated_at}  {row.preview}",
            )
            for row in rows
        ]
        active_session_id = (
            self._agent_session.session.id
            if self._agent_session and self._agent_session.session
            else None
        )
        self._session_choice_index = next(
            (
                index
                for index, session_id in enumerate(self._session_choices)
                if session_id == active_session_id
            ),
            0,
        )
        self._ensure_newline()
        self._session_choice_start = len(self._message_buffer.text)
        self._render_session_selector()

    def _move_session_choice(self, step: int) -> bool:
        if not self._session_choices:
            return False
        if abs(step) == 1:
            self._session_choice_index = (self._session_choice_index + step) % len(
                self._session_choices
            )
        else:
            self._session_choice_index = max(
                0,
                min(len(self._session_choices) - 1, self._session_choice_index + step),
            )
        self._render_session_selector()
        return True

    def _accept_session_choice(self) -> bool:
        if not self._session_choices:
            return False
        session_id = self._session_choices[self._session_choice_index]
        self._session_choices = []
        self._session_choice_rows = []
        self._replace_from(self._session_choice_start, "")
        if self._session_choice_mode == "delete":
            self._session_choice_mode = "resume"
            self._pending_task = asyncio.create_task(self._delete_session(session_id))
            return True
        self._session_choice_mode = "resume"
        self._resume_session(session_id)
        return True

    def _cancel_session_choice(self) -> bool:
        if not self._session_choices:
            return False
        self._session_choices = []
        self._session_choice_rows = []
        self._session_choice_mode = "resume"
        self._replace_from(self._session_choice_start, "")
        self._append("Session selection cancelled.\n")
        return True

    def _render_session_selector(self) -> None:
        active_session_id = (
            self._agent_session.session.id
            if self._agent_session and self._agent_session.session
            else None
        )
        action = (
            "session to delete" if self._session_choice_mode == "delete" else "session"
        )
        lines = ["", f"Select {action} (up/down, enter, esc)", ""]
        start, end = self._selector_slice(
            len(self._session_choice_rows),
            self._session_choice_index,
            row_height=2,
        )
        if start:
            lines.append(f"  ↑ {start} more")
        for index in range(start, end):
            session_id, model, details = self._session_choice_rows[index]
            arrow = "❯" if index == self._session_choice_index else " "
            active = "√" if session_id == active_session_id else " "
            lines.append(f"{arrow} {active} {session_id}  {model}")
            lines.append(f"      {details}")
        if end < len(self._session_choice_rows):
            lines.append(f"  ↓ {len(self._session_choice_rows) - end} more")
        lines.append("")
        self._replace_from(self._session_choice_start, "\n".join(lines) + "\n")

    def _start_rewind_selector(self) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        store = self._agent_session.session_store
        session_id = self._agent_session.session_id
        try:
            info = store.info(session_id)
            active_path_ids = (
                {entry.id for entry in store.path_entries(session_id)}
                if info.leaf_id
                else set()
            )
            entries = [
                entry for entry in store.entries(session_id) if entry.type != "leaf"
            ]
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return
        self._rewind_choices = [entry.id for entry in entries]
        self._rewind_choice_rows = [
            (
                entry.id,
                f"{entry.type}  {entry_active_label(entry.id, info.leaf_id, active_path_ids)}  {entry_preview(entry.data)}".strip(),
            )
            for entry in entries
        ]
        if not self._rewind_choices:
            self._ensure_newline()
            self._append("No entries to rewind.\n")
            return
        self._rewind_choice_index = 0
        self._ensure_newline()
        self._rewind_choice_start = len(self._message_buffer.text)
        self._render_rewind_selector()

    def _move_rewind_choice(self, step: int) -> bool:
        if not self._rewind_choices:
            return False
        if abs(step) == 1:
            self._rewind_choice_index = (self._rewind_choice_index + step) % len(
                self._rewind_choices
            )
        else:
            self._rewind_choice_index = max(
                0,
                min(len(self._rewind_choices) - 1, self._rewind_choice_index + step),
            )
        self._render_rewind_selector()
        return True

    def _accept_rewind_choice(self) -> bool:
        if not self._rewind_choices:
            return False
        entry_id = self._rewind_choices[self._rewind_choice_index]
        self._rewind_choices = []
        self._rewind_choice_rows = []
        self._replace_from(self._rewind_choice_start, "")
        self._rewind_session(entry_id)
        return True

    def _cancel_rewind_choice(self) -> bool:
        if not self._rewind_choices:
            return False
        self._rewind_choices = []
        self._rewind_choice_rows = []
        self._replace_from(self._rewind_choice_start, "")
        self._append("Rewind cancelled.\n")
        return True

    def _render_rewind_selector(self) -> None:
        lines = ["", "Select rewind entry (up/down, enter, esc)", ""]
        start, end = self._selector_slice(
            len(self._rewind_choice_rows), self._rewind_choice_index
        )
        if start:
            lines.append(f"  ↑ {start} more")
        for index in range(start, end):
            entry_id, details = self._rewind_choice_rows[index]
            arrow = "❯" if index == self._rewind_choice_index else " "
            lines.append(f"{arrow} {entry_id}  {details}")
        if end < len(self._rewind_choice_rows):
            lines.append(f"  ↓ {len(self._rewind_choice_rows) - end} more")
        lines.append("")
        self._replace_from(self._rewind_choice_start, "\n".join(lines) + "\n")

    def _append_table(self, table: Table) -> None:
        self._ensure_newline()
        self._append(self._render_table_text(table))

    def _model_label(self) -> str:
        if self._agent_session is None:
            return ""
        model = self._agent_session.model
        return f"{model.provider}/{model.id}"

    async def _run_slash_command(self, text: str) -> None:
        try:
            await self._handle_slash_command(text)
        except Exception as exc:  # noqa: BLE001
            self._ensure_newline()
            self._append(f"Command failed: {exc}\n")
        finally:
            self._app.invalidate()

    async def _handle_slash_command(self, text: str) -> bool:
        if self._agent_session is None or not text.startswith("/"):
            return False

        from spice.interactive.commands import SlashCommandRegistry as CoreRegistry

        registry = CoreRegistry(self._agent_session.extensions)
        port = _TuiInteractivePort(self)
        ctx = CommandContext(
            session=self._agent_session,
            cwd=Path.cwd(),
            port=port,
            raw=text,
            args="",
            confirm_policy=self.confirm_policy,
            extras={
                "extension_context": SimpleNamespace(
                    console=_TuiConsole(self),
                    agent_session=self._agent_session,
                    cwd=Path.cwd(),
                ),
                "registry": registry,
            },
        )
        result = await registry.execute(text, ctx)
        if result.exit_requested:
            self._app.exit()
            return True

        previous_session = self._agent_session
        new_session = result.session
        replacing_session = (
            new_session is not None and new_session is not previous_session
        )
        if result.clear_requested or (
            replacing_session and result.replay_session_history
        ):
            self._clear_conversation()
        if new_session is not None and new_session is not previous_session:
            self.confirm_policy = ConfirmPolicy()
            set_confirm = getattr(new_session, "set_confirm", None)
            if set_confirm is not None:
                set_confirm(self.confirm)
            else:
                new_session.confirm = self.confirm
            self._agent_session = new_session
            self._bind_trace_session()
            self._refresh_completer()
            close = getattr(previous_session, "aclose", None)
            if close is not None:
                await close()

        self._paint_command_result(result)
        if replacing_session and new_session is not None:
            self._ensure_newline()
            self._append(f"Session: {new_session.session_label}\n")
            self._append(f"Model: {new_session.runtime_model_label}\n")
            if result.replay_session_history:
                self._display_session_history()
        if result.followup_prompt:
            await self._run_prompt(result.followup_prompt)
        return True

    def _show_settings(self) -> None:
        if self._agent_session is None:
            return
        config = load_config()
        table = Table("Setting", "Value")
        table.add_row("session", self._agent_session.session_label)
        table.add_row("cwd", str(Path.cwd()))
        table.add_row("model", self._model_label())
        table.add_row(
            "protocol", self._agent_session.model.protocol or "provider default"
        )
        table.add_row("default model", f"{config.provider}/{config.model}")
        table.add_row("temperature", str(config.temperature))
        table.add_row("memory.enabled", str(config.memory_enabled).lower())
        table.add_row(
            "subagents.enabled", str(self._agent_session.subagents_enabled).lower()
        )
        table.add_row(
            "subagents.max_concurrent",
            str(self._agent_session.subagent_manager.max_concurrent),
        )
        table.add_row("logging.retention_days", str(config.logging_retention_days))
        table.add_row(
            "tools.max_concurrency", str(config.tools.get("max_concurrency", 4))
        )
        table.add_row(
            "tools.default_timeout_seconds",
            str(config.tools.get("default_timeout_seconds", 120)),
        )
        table.add_row(
            "model retry",
            str(config.model_routing["retry"].get("enabled", True)).lower(),
        )
        table.add_row(
            "model retry attempts",
            str(config.model_routing["retry"].get("maxAttempts", 3)),
        )
        table.add_row(
            "model fallback",
            str(config.model_routing["fallback"].get("enabled", False)).lower(),
        )
        table.add_row(
            "fallback profiles",
            ", ".join(config.model_routing["fallback"].get("profiles", [])) or "(none)",
        )
        table.add_row("output tokens", str(self._agent_session.model.output_tokens))
        table.add_row("mode", self._agent_session.plan_state.mode)
        if self._agent_session.todo_state.status_line():
            table.add_row("todo", self._agent_session.todo_state.status_line())
        table.add_row(
            "tool approvals",
            "all tools allowed for session" if self._allow_all_tools else "ask",
        )
        table.add_row(
            "file edit approvals",
            "session allowed" if self._allow_file_edits else "ask",
        )
        self._append_table(table)

    def _show_tools(self) -> None:
        memory_enabled = load_config().memory_enabled
        subagents_enabled = bool(
            self._agent_session and self._agent_session.subagents_enabled
        )
        registry = create_all_tools(
            memory_enabled=memory_enabled, subagents_enabled=subagents_enabled
        )
        table = Table("Toolset", "Tools")
        for name, tools in TOOLSETS.items():
            if name == "memory" and not memory_enabled:
                continue
            if name == "subagent" and not subagents_enabled:
                continue
            table.add_row(name, ", ".join(tools))
        table.add_row(
            "read-only", ", ".join(name for name in READ_ONLY_TOOLS if name in registry)
        )
        table.add_row(
            "parallel",
            ", ".join(
                name
                for name, tool in registry.items()
                if tool.concurrency == "parallel"
            ),
        )
        self._append_table(table)

    def _handle_subagent_command(self, arg: str) -> None:
        if self._agent_session is None:
            return
        action = (arg or "status").strip().lower()
        if action in {"on", "enable", "enabled"}:
            self._agent_session.set_subagents_enabled(True)
            self._append("Subagents enabled for this session.\n")
            self._show_subagent_status()
            return
        if action in {"off", "disable", "disabled"}:
            self._agent_session.set_subagents_enabled(False)
            self._append("Subagents disabled for this session.\n")
            self._show_subagent_status()
            return
        if action in {"", "status"}:
            self._show_subagent_status()
            return
        self._append("Usage: /subagent [on|off|status]\n")

    def _show_subagent_status(self) -> None:
        if self._agent_session is None:
            return
        active_tools = self._agent_session.get_active_tools()
        table = Table("Setting", "Value")
        table.add_row(
            "session enabled", str(self._agent_session.subagents_enabled).lower()
        )
        table.add_row(
            "max concurrent", str(self._agent_session.subagent_manager.max_concurrent)
        )
        table.add_row("mode", self._agent_session.plan_state.mode)
        table.add_row("tool available", str("spawn_subagents" in active_tools).lower())
        self._append_table(table)

    def _show_models(self, selector: str = "") -> None:
        if self._agent_session is None:
            return
        registry = ModelRegistry()
        if selector:
            provider, _, model_id = selector.partition("/")
            if not model_id:
                model_id = provider
                provider = self._agent_session.model.provider
            model = registry.find(provider, model_id)
            if not model:
                self._ensure_newline()
                self._append(f"Model not found: {selector}\n")
                return
            self._set_model(model)
            return

        self._start_model_selector()

    def _show_sessions(self) -> None:
        self._start_session_selector()

    def _show_delete_sessions(self) -> None:
        self._start_session_selector(mode="delete")

    def _clear_conversation(self) -> None:
        self._message_follow_tail = True
        self._set_message_text("")

    async def _reset_session(self) -> None:
        if self._agent_session is None:
            return
        if self._agent_session.session is None:
            self._clear_conversation()
            self._append("No session has been created yet.\n")
            return
        confirmed = await self._confirm_reset()
        if not confirmed:
            self._ensure_newline()
            self._append("Reset cancelled.\n")
            return

        try:
            self._agent_session.reset()
        except (RuntimeError, ValueError) as exc:
            self._ensure_newline()
            self._append(f"Reset failed: {exc}\n")
            return

        self._clear_conversation()
        self._append(f"Reset session: {self._agent_session.session_label}\n")
        self._append(f"Session: {self._agent_session.session_label}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        self._refresh_completer()

    async def _delete_session(self, target: str) -> None:
        if self._agent_session is None:
            return
        current_session_id = (
            self._agent_session.session_id if self._agent_session.session else None
        )
        session_id = target
        if target == "current":
            if not current_session_id:
                self._ensure_newline()
                self._append("No current session has been created yet.\n")
                return
            session_id = current_session_id
        elif not target:
            self._show_delete_sessions()
            return
        else:
            try:
                session_id = self._agent_session.session_store.resolve(
                    target, cwd=Path.cwd()
                ).id
            except ValueError as exc:
                self._ensure_newline()
                self._append(f"Delete failed: {exc}\n")
                return

        confirmed = await self._confirm_delete_session(
            session_id, current=session_id == current_session_id
        )
        if not confirmed:
            self._ensure_newline()
            self._append("Delete cancelled.\n")
            return

        try:
            self._agent_session.session_store.delete(session_id)
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Delete failed: {exc}\n")
            return

        if session_id != current_session_id:
            self._ensure_newline()
            self._append(f"Deleted session: {session_id}\n")
            return

        current_model = self._agent_session.model
        self._agent_session = AgentSession(
            cwd=Path.cwd(),
            provider=current_model.provider,
            model_id=current_model.id,
            confirm=self.confirm,
            session_store=self._agent_session.session_store,
            extension_manager=self._agent_session.extensions,
        )
        self.confirm_policy = ConfirmPolicy()
        self._bind_trace_session()
        self._clear_conversation()
        self._append(f"Deleted session: {session_id}\n")
        self._append(f"Session: {self._agent_session.session_label}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        self._refresh_completer()

    def _resume_session(self, session_id: str) -> None:
        if not session_id:
            self._start_session_selector()
            return
        try:
            self._agent_session = AgentSession(
                cwd=Path.cwd(),
                provider=self._provider,
                model_id=self._model,
                session_id=session_id,
                confirm=self.confirm,
            )
        except (RuntimeError, ValueError) as exc:
            self._ensure_newline()
            self._append(f"Resume failed: {exc}\n")
            return
        self.confirm_policy = ConfirmPolicy()
        self._bind_trace_session()
        # A resumed session replaces the current conversation view. Resetting
        # follow-tail is essential when the previous view was scrolled up.
        self._clear_conversation()
        self._append(f"Resumed session: {self._agent_session.session_label}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        self._display_session_history()
        self._refresh_completer()

    def _display_session_history(self) -> None:
        """Render session messages like re-entering the conversation."""
        if self._agent_session is None:
            return
        messages = [m for m in self._agent_session.messages if m.role != "system"]
        if not messages:
            return
        self._append("\n")
        for message in messages:
            if message.role == "user":
                content = (message.content or "").strip()
                if content:
                    self._append(f"You: {content}\n\n")
            elif message.role == "assistant":
                content = (message.content or "").strip()
                if content:
                    self._append("Spice: ")
                    self._append(self._render_markdown_text(content))
                    self._append("\n")
                if message.tool_calls:
                    for call in message.tool_calls:
                        self._append(f"  tool: {call.name}()\n")
            elif message.role == "tool":
                content = (message.content or "").strip()
                name = message.name or "tool"
                if content:
                    preview = content[:300] + "..." if len(content) > 300 else content
                    self._append(f"  {name} -> {preview}\n")

    def _show_history(self, args: str) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        store = self._agent_session.session_store
        session_id = self._agent_session.session_id
        try:
            if args == "--raw":
                path = store.path_for(session_id)
                lines = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        lines.append(line)
                    else:
                        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
                self._ensure_newline()
                self._append("\n".join(lines) + "\n")
                return
            if args == "--tree":
                self._show_history_tree()
                return
            if args:
                self._ensure_newline()
                self._append("Usage: /history [--tree|--raw]\n")
                return
            info = store.info(session_id)
            active_context = store.build_context(session_id)
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return

        table = Table("Field", "Value")
        table.add_row("ID", info.id)
        table.add_row("Model", f"{info.provider}/{info.model}")
        table.add_row("CWD", info.cwd)
        table.add_row("Messages", str(info.message_count))
        table.add_row("Leaf", active_context.leaf_id or "")
        self._append_table(table)
        for message in active_context.messages:
            title = (
                message.role if not message.name else f"{message.role}:{message.name}"
            )
            body = message.content or "<empty>"
            if message.tool_calls:
                calls = ", ".join(
                    f"{call.name}({call.id})" for call in message.tool_calls
                )
                body = f"{body}\n\nTool calls: {calls}".strip()
            self._ensure_newline()
            self._append(f"{title}\n{body}\n")

    def _show_history_tree(self) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        store = self._agent_session.session_store
        session_id = self._agent_session.session_id
        try:
            info = store.info(session_id)
            active_path_ids = (
                {entry.id for entry in store.path_entries(session_id)}
                if info.leaf_id
                else set()
            )
            entries = store.entries(session_id)
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return
        table = Table("Entry", "Parent", "Type", "Active", "Preview")
        for entry in entries:
            active = (
                "yes"
                if entry.id == info.leaf_id
                else ("path" if entry.id in active_path_ids else "")
            )
            table.add_row(
                entry.id,
                entry.parent_id or "",
                entry.type,
                active,
                entry_preview(entry.data),
            )
        self._append_table(table)

    def _rewind_session(self, entry_id: str) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        if not entry_id:
            self._start_rewind_selector()
            return
        try:
            self._agent_session.rewind(entry_id)
        except (RuntimeError, ValueError) as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return
        self._ensure_newline()
        self._append(
            f"Rewound session: {self._agent_session.session_id} -> {entry_id}\n"
        )

    async def _show_compact(self, args: str = "") -> None:
        session = self._agent_session
        self._ensure_newline()
        if session is None or session.session is None:
            self._append("No session has been created yet.\n")
            return
        if args in {"status", "--status"}:
            status = session.compaction_status()
            threshold = (
                str(status.threshold_tokens)
                if status.threshold_tokens is not None
                else "n/a"
            )
            self._append(
                f"Compaction status: estimated={status.estimated_tokens} threshold={threshold} "
                f"reserve={status.reserve_tokens} auto={'yes' if status.should_compact else 'no'} "
                f"reason={status.reason}\n"
            )
            return
        try:
            result = await session.compact(
                focus=args or None, reason="manual", force=True
            )
        except Exception as exc:
            self._append(f"Compaction failed: {exc}\n")
            return
        self._append(
            f"Compacted session: ~{result.tokens_before} -> ~{result.tokens_after} tokens\n"
        )

    async def _handle_memory_command(self, args: str) -> None:
        session = self._agent_session
        if session is None:
            return
        words = args.split()
        action = words[0].lower() if words else "project"
        store = session.memory_store
        if action == "status":
            status = store.status()
            table = Table("Memory", "Usage")
            table.add_row("Project", status.get("project_usage") or "n/a")
            table.add_row("User", status["user_usage"])
            table.add_row("Global", status["memory_usage"])
            table.add_row("Unprocessed history", str(status["unprocessed_count"]))
            table.add_row("Workspace", status.get("workspace") or "n/a")
            self._append_table(table)
            return
        if action == "show":
            scope = words[1].lower() if len(words) > 1 else "project"
            target = {"user": "user", "global": "memory", "project": "project"}.get(
                scope
            )
            if target is None:
                self._append("Usage: /memory show [user|global|project]\n")
                return
            entries = store.read_entries(target)
            self._append(
                f"{scope} memory:\n"
                + ("\n\n".join(entries) if entries else "(empty)")
                + "\n"
            )
            return
        if not session.config.memory_enabled:
            self._append(
                "Long-term memory is disabled. Run `spice memory enable` first.\n"
            )
            return
        if action not in {"distill", "all", "project", "global"}:
            self._append(
                "Usage: /memory [project|global|all|status|show [user|global|project]]\n"
            )
            return
        distill_scope: Literal["all", "global", "project"] = (
            "project"
            if action == "project"
            else "global"
            if action == "global"
            else "all"
        )
        try:
            result = await session.distill_current_memory(scope=distill_scope)
        except Exception as exc:
            self._append(f"Memory distillation failed: {exc}\n")
            return
        if result.get("processed", 0) == 0:
            self._append(
                str(result.get("message", "No conversation content to distill.")) + "\n"
            )
        elif result.get("success"):
            self._append(
                "Memory updated: "
                f"adds={result.get('adds', 0)} replacements={result.get('replacements', 0)} "
                f"removals={result.get('removals', 0)}\n"
            )
        else:
            self._append(
                str(
                    result.get(
                        "message", "Memory review completed with skipped changes."
                    )
                )
                + "\n"
            )

    def _show_skills(self) -> None:
        result = load_skills(cwd=Path.cwd())
        if not result.skills:
            self._ensure_newline()
            self._append("No skills found.\n")
            return
        table = Table("Skill", "Description")
        for skill in result.skills:
            table.add_row(skill.name, skill.description or "")
        self._append_table(table)

    def _show_skill(self, name: str) -> None:
        if not name:
            self._ensure_newline()
            self._append("Usage: /skill:<name>\n")
            return
        try:
            content = read_skill_content(name, cwd=Path.cwd())
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"{exc}\n")
            return
        self._ensure_newline()
        self._append(f"# Skill: {name}\n")
        self._append(self._render_markdown_text(content))

    def _refresh_completer(self) -> None:
        if self._agent_session is None:
            return
        slash_completer = SlashCommandRegistry(
            self._agent_session.extensions
        ).completer()
        completer = SpiceInputCompleter(slash_completer, cwd=Path.cwd())
        self._input_buffer.completer = completer
        self._input_buffer.complete_while_typing = Condition(lambda: True)

    def _toggle_interaction_mode(self) -> None:
        if self._agent_session is None:
            return
        mode = self._agent_session.toggle_interaction_mode()
        self._show_edit_mode_hint = mode == "edit"

    async def _handle_plan_command(self, arg: str) -> None:
        if self._agent_session is None:
            return
        if arg == "cancel":
            self._agent_session.cancel_plan()
            self._show_edit_mode_hint = False
            self._ensure_newline()
            self._append("Switched to edit mode. Plan cleared.\n")
            return
        if arg == "execute":
            prompt = self._agent_session.approve_plan("manual")
            self._show_edit_mode_hint = False
            self._ensure_newline()
            self._append("Switched to edit mode. Executing the approved plan.\n")
            await self._run_prompt(prompt)
            return
        self._agent_session.start_plan(arg)
        self._ensure_newline()
        self._append("Plan mode on. Read-only tools are active.\n")
        if arg:
            await self._run_prompt(arg)

    async def _handle_sustained_goal_command(self, arg: str, *, command: str) -> None:
        if self._agent_session is None:
            return
        objective = arg.strip()
        if not objective:
            self._ensure_newline()
            self._append(f"Usage: /{command} <objective|status|cancel|complete>\n")
            return
        action, _, rest = objective.partition(" ")
        action = action.lower()
        if action == "status":
            state = self._agent_session.long_task_status()
            self._ensure_newline()
            if not state.objective:
                self._append(f"No sustained {command} is active.\n")
                return
            self._append(
                f"Sustained {command}: {state.status} {state.task_id or 'legacy'}\n"
                f"Objective: {state.objective}\n"
                f"Continuations: {state.continuation_rounds}/{state.max_continuation_rounds}, remaining {state.remaining_continuations}\n"
                f"Needs attention: {str(state.needs_user_attention).lower()}\n"
                f"Completion candidate: {str(state.completion_candidate).lower()}\n"
            )
            return
        if action == "cancel":
            try:
                self._agent_session.cancel_long_task(note=rest.strip())
            except ValueError as exc:
                self._ensure_newline()
                self._append(f"{exc}\n")
                return
            self._ensure_newline()
            self._append(f"Sustained {command} cancelled.\n")
            return
        if action == "complete":
            from spice.interactive.commands import parse_force_note

            force, note = parse_force_note(rest)
            try:
                self._agent_session.complete_long_task(note=note, force=force)
            except ValueError as exc:
                self._ensure_newline()
                self._append(f"{exc}\n")
                return
            self._ensure_newline()
            self._append(f"Sustained {command} completed.\n")
            return
        self._agent_session.set_interaction_mode("edit")
        self._agent_session.start_long_task(objective)
        self._ensure_newline()
        self._append(f"Sustained {command} started.\n")
        from spice.interactive.commands import sustained_goal_prompt

        await self._run_prompt(sustained_goal_prompt(objective, command=command))

    # --------------------------------------------------------------- input

    def _on_accept(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        if not text:
            return False  # clear input
        buffer.reset(append_to_history=True)
        self._follow_message_tail()

        if self._busy:
            self._ensure_newline()
            self._append("[busy] please wait for the current turn to finish.\n")
            return True

        if text in {"clear", "cls"}:
            text = "/clear"

        if self._agent_session and self._agent_session.plan_state.mode != "plan":
            self._show_edit_mode_hint = False

        if text in {"exit", "quit", "/quit"}:
            self._app.exit()
            return True

        if text.startswith("/"):
            self._pending_task = asyncio.create_task(self._run_slash_command(text))
            return True

        if (
            self._agent_session
            and self._agent_session.plan_state.is_plan_mode
            and is_execute_request(text)
        ):
            prompt = self._agent_session.approve_plan("manual")
            self._ensure_newline()
            self._append("Switched to edit mode. Executing the approved plan.\n")
            self._pending_task = asyncio.create_task(self._run_prompt(prompt))
            return True

        self._ensure_turn_gap()
        self._append(f"You: {text}\n\n")
        self._pending_task = asyncio.create_task(self._run_prompt(text))
        return True

    async def _run_prompt(self, message: str) -> None:
        if self._agent_session is None:
            return
        self._busy = True
        self._streaming = False
        self._activity_started = time.monotonic()
        self._last_stream_delta_at = None
        self._active_tool_name = None
        refresh_task = asyncio.create_task(self._refresh_waiting_indicator())
        try:
            async for evt in self._agent_session.prompt(message):
                self._render_event(evt)
                self._app.invalidate()
            await self._handle_plan_ready()
        except asyncio.CancelledError:
            self._stop_waiting_indicator(clear=True)
            self._append("\n[cancelled]\n")
            raise
        except Exception as exc:  # noqa: BLE001
            self._stop_waiting_indicator(clear=True)
            self._append(f"\nError: {exc}\n")
        finally:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
            self._stop_waiting_indicator(clear=True)
            self._busy = False
            self._streaming = False
            self._activity_started = None
            self._last_stream_delta_at = None
            self._active_tool_name = None
            self._ensure_newline()
            self._app.invalidate()

    async def _refresh_waiting_indicator(self) -> None:
        while True:
            await asyncio.sleep(0.125)
            self._render_waiting_indicator()
            self._app.invalidate()

    async def _handle_plan_ready(self) -> None:
        if self._agent_session is None:
            return
        if (
            not self._agent_session.plan_state.is_plan_mode
            or not self._agent_session.plan_state.steps
        ):
            return
        selected = await self._select_plan_action()
        if selected == "auto":
            self._allow_file_edits = True
            prompt = self._agent_session.approve_plan("auto")
            self._ensure_newline()
            self._append(
                "Switched to edit mode. Executing with file edits auto-approved for this session.\n"
            )
            await self._run_prompt(prompt)

    # ------------------------------------------------------------- events

    def _render_event(self, event: AgentEvent) -> None:
        if isinstance(event, (AgentStartEvent, AgentEndEvent)):
            return
        if isinstance(event, TurnStartEvent):
            self._flush_compact_tool_group()
            self._ensure_newline()
            self._start_waiting_indicator("Waiting for model")
            self._response_parts = []
            self._streaming = False
            self._last_stream_delta_at = None
            self._active_tool_name = None
        elif isinstance(event, ReasoningDeltaEvent):
            self._stop_waiting_indicator(clear=True)
            self._last_stream_delta_at = time.monotonic()
            self._active_tool_name = None
            if not self._reasoning_streaming:
                self._flush_compact_tool_group()
                self._append("Thinking: ")
                self._reasoning_streaming = True
            self._append(event.text)
        elif isinstance(event, TextDeltaEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._last_stream_delta_at = time.monotonic()
            self._active_tool_name = None
            if not self._streaming:
                self._flush_compact_tool_group()
                self._append("Spice: ")
                self._response_start = len(self._message_buffer.text)
                self._response_parts = []
                self._streaming = True
            else:
                self._flush_compact_tool_group()
            self._response_parts.append(event.text)
            self._append(event.text)
        elif isinstance(event, AssistantMessageEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            if event.tool_calls and not (event.text or "").strip():
                self._streaming = False
                self._response_parts = []
                return
            self._flush_compact_tool_group()
            text = event.text or "".join(self._response_parts)
            if not self._streaming and text.strip():
                self._append("Spice: ")
                self._response_start = len(self._message_buffer.text)
            if text.strip():
                self._replace_from(
                    self._response_start, self._render_markdown_text(text)
                )
            self._streaming = False
            self._response_parts = []
            self._ensure_newline()
        elif isinstance(event, ModelRetryEvent):
            self._stop_waiting_indicator(clear=True)
            self._last_stream_delta_at = None
            self._active_tool_name = None
            self._ensure_newline()
            self._append(
                f"Model request failed temporarily. Retrying {event.next_attempt}/{event.max_attempts} "
                f"in {event.delay_seconds:.1f}s ({event.provider}/{event.model}).\n"
            )
            self._start_waiting_indicator("Retrying model request")
        elif isinstance(event, ModelFallbackEvent):
            self._stop_waiting_indicator(clear=True)
            self._last_stream_delta_at = None
            self._active_tool_name = None
            self._ensure_newline()
            self._append(
                f"Model fallback: {event.from_provider}/{event.from_model} -> "
                f"{event.to_provider}/{event.to_model} (this turn, reason: {event.reason}).\n"
            )
            self._start_waiting_indicator("Waiting for fallback model")
        elif isinstance(event, ToolExecutionStartEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._active_tool_name = event.tool_name
            self._ensure_newline()
            args = event.args or {}
            self._pending_tool_args[event.tool_call_id] = args
            if event.tool_name == "update_todo":
                self._start_waiting_indicator(f"Running {event.tool_name}")
                return
            start = format_tool_start(event.tool_name, args)
            if event.tool_name in COMPACTABLE_TOOL_NAMES:
                self._start_compact_tool(event.tool_name, start)
                self._start_waiting_indicator(f"Running {event.tool_name}")
                return
            self._flush_compact_tool_group()
            self._append(f"{start}\n")
            self._start_waiting_indicator(f"Running {event.tool_name}")
        elif isinstance(event, ToolExecutionUpdateEvent):
            self._stop_waiting_indicator(clear=True)
            self._flush_compact_tool_group()
            self._ensure_newline()
            self._append(event.text)
        elif isinstance(event, ToolExecutionEndEvent):
            self._stop_waiting_indicator(clear=True)
            self._active_tool_name = None
            self._last_stream_delta_at = None
            self._ensure_newline()
            had_start = event.tool_call_id in self._pending_tool_args
            args = self._pending_tool_args.pop(event.tool_call_id, {})
            summary = format_tool_end(event.tool_name, args, event.result)
            if event.tool_name == "update_todo" and not event.result.is_error:
                self._todo_items, self._todo_summary = _todo_items_from_result(
                    event.result
                )
                self._app.invalidate()
                return
            if (
                had_start
                and event.tool_name in COMPACTABLE_TOOL_NAMES
                and self._compact_tool_group is not None
            ):
                self._finish_compact_tool(
                    event.tool_name, summary, is_error=event.result.is_error
                )
                return
            self._flush_compact_tool_group()
            self._append(f"  {summary}\n")
        elif isinstance(event, RoundCompleteEvent):
            self._finish_reasoning_stream()
            self._start_waiting_indicator("Waiting for model")
        elif isinstance(event, AgentErrorEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._streaming = False
            self._active_tool_name = None
            self._ensure_newline()
            self._flush_compact_tool_group()
            label = (
                "Current turn stopped"
                if event.kind == "fatal_tool"
                else "Interrupted"
                if event.kind == "user_interrupted"
                else "Error"
            )
            self._append(f"{label}: {event.message}\n")
        elif isinstance(event, TurnEndEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._streaming = False
            self._active_tool_name = None
            self._ensure_newline()
            self._flush_compact_tool_group()
            if not self._todo_has_active_work():
                self._todo_items = []
                self._todo_summary = {}
                self._app.invalidate()

    def _todo_has_active_work(self) -> bool:
        return any(
            str(item.get("status") or "pending") in {"pending", "in_progress"}
            for item in self._todo_items
        )

    def _start_compact_tool(self, tool_name: str, start: str) -> None:
        if (
            self._compact_tool_group is not None
            and self._compact_tool_group.tool_name != tool_name
        ):
            self._flush_compact_tool_group()
        if self._compact_tool_group is None:
            self._compact_tool_group = CompactToolGroup(tool_name=tool_name)
        self._compact_tool_group.count += 1
        self._compact_tool_group.last_start = start
        self._compact_tool_group.last_end = ""

    def _finish_compact_tool(
        self, tool_name: str, summary: str, *, is_error: bool
    ) -> None:
        if (
            self._compact_tool_group is None
            or self._compact_tool_group.tool_name != tool_name
        ):
            self._flush_compact_tool_group()
            self._compact_tool_group = CompactToolGroup(tool_name=tool_name, count=1)
        self._compact_tool_group.last_end = summary
        if is_error:
            self._compact_tool_group.failures += 1

    def _flush_compact_tool_group(self) -> None:
        group = self._compact_tool_group
        if group is None:
            return
        self._compact_tool_group = None
        if not group.last_start and not group.last_end:
            return
        start = group.last_start or f"❯ {group.tool_name}"
        if group.count > 1:
            start = f"{start} · {group.count} tool calls"
        summary = group.last_end or "... running"
        if group.count > 1:
            summary = f"{summary} · {group.count} {group.tool_name} calls"
        if group.failures:
            summary = f"{summary} · {group.failures} failed"
        self._append(f"{start}\n  {summary}\n")

    # ----------------------------------------------------------------- run

    async def run(self) -> None:
        try:
            self._agent_session = AgentSession(
                cwd=Path.cwd(),
                provider=self._provider,
                model_id=self._model,
                session_id=self._session_id,
                continue_latest=self._continue_session,
                confirm=self.confirm,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}")
            return

        self._refresh_completer()
        self._append(f"Session: {self._agent_session.session_label}\n")
        if self._trace:
            from spice.agent.trace import attach_trace_writer

            self._trace_writer, trace_path = attach_trace_writer(
                self._agent_session, path=self._trace_file
            )
            self._append(f"Trace: {trace_path}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        if self._agent_session.extensions.errors:
            for err in self._agent_session.extensions.errors:
                self._append(f"Extension load failed: {err}\n")
        # self._append("Type your message and press Enter. Ctrl+C to quit.\n")

        try:
            await self._app.run_async()
        finally:
            if self._agent_session is not None:
                await self._agent_session.aclose()

    def _bind_trace_session(self) -> None:
        trace_writer = getattr(self, "_trace_writer", None)
        if trace_writer is not None and self._agent_session is not None:
            trace_writer.bind(self._agent_session)


def run_tui(
    *,
    provider: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    continue_session: bool = False,
    trace: bool = False,
    trace_file: Path | None = None,
) -> None:
    """Synchronous entry point used by the Typer command."""
    from spice.cli.terminal import preserve_cursor_blink

    preserve_cursor_blink()
    tui = SpiceTUI(
        provider=provider,
        model=model,
        session_id=session_id,
        continue_session=continue_session,
        trace=trace,
        trace_file=trace_file,
    )
    asyncio.run(tui.run())
