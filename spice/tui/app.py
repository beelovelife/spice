"""Minimal full-screen TUI for Spice built on prompt_toolkit.

Layout:

    +------------------------------------------+
    |          message area (scrollable)       |
    |                                          |
    +------------------------------------------+
    |  >  fixed input box                      |
    +------------------------------------------+

It reuses ``AgentSession.prompt()`` and consumes the same ``AgentEvent``
stream as the regular CLI. Rendering is intentionally plain text for the
first version; styling can be layered on later.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cursor_shapes import CursorShape
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
    TurnFinalizationStartEvent,
    TurnStartEvent,
)
from spice.agent.turn_outcome import pause_notice
from spice.cli.completion import (
    accept_completion,
    insert_text_and_maybe_complete,
    start_or_accept_completion,
)
from spice.cli.render import (
    COMPACTABLE_TOOL_NAMES,
    CompactToolGroup,
    _todo_items_from_result,
    format_tool_end,
    format_tool_start,
)
from spice.interactive.commands import is_execute_request
from spice.interactive.confirm import (
    ConfirmPolicy,
)
from spice.cli.terminal import enable_cursor_blink_after_render
from spice.cli.welcome import choose_welcome_quote
from spice.llm.models import Model
from spice.tui.legacy_commands import LegacyCommandMixin
from spice.tui.legacy_views import LegacyViewMixin


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


class SpiceTUI(LegacyViewMixin, LegacyCommandMixin):
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
        self._welcome_quote_en, self._welcome_quote_zh = choose_welcome_quote()
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
                input_processors=[BeforeInput("> ", style="class:prompt")],
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
        self._append(f"> {text}\n\n")
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
        elif isinstance(event, TurnFinalizationStartEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._streaming = False
            self._ensure_newline()
            self._flush_compact_tool_group()
            self._start_waiting_indicator("整理进度")
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
            if event.status == "paused":
                self._append(pause_notice(event.checkpoint.round_limit if event.checkpoint else None) + "\n")
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
        self._display_session_history()
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
