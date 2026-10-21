"""View state, selectors, confirmations, and rendering for the legacy TUI."""

from __future__ import annotations

import asyncio
import time
from io import StringIO
from pathlib import Path

from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.table import Table

from spice.cli.render import _is_table_divider, _looks_like_table_line, _todo_marker
from spice.interactive.activity import activity_parts, activity_text
from spice.interactive.sessions import (
    SESSION_PICKER_LIMIT,
    entry_active_label,
    entry_preview,
)
from spice.interactive.types import (
    ChoiceItem,
    ChoiceRequest,
    CommandResult,
    ConfirmRequest,
    PanelView,
    TableView,
    TextView,
)
from spice.llm.config import get_api_key, load_config, save_config
from spice.llm.model_registry import ModelRegistry
from spice.llm.models import Model
from spice.storage.factory import create_session_store
from spice.tui.legacy_ports import TuiToolConfirmPort as _TuiToolConfirmPort
from spice.tui.legacy_render import (
    compact_blank_lines as _compact_blank_lines,
    render_plain_markdown as _render_plain_markdown,
    render_plain_table as _render_plain_table,
)


class LegacyViewMixin:
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
            if line.startswith("> "):
                fragments.extend(
                    [
                        ("class:message.user-label", ">"),
                        ("", line.removeprefix(">") + suffix),
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
