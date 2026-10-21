"""Focused Textual widgets for the Spice conversation surface."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from textual.binding import Binding
from textual.content import Content, Span
from textual.message import Message
from textual.widgets import Collapsible, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from spice.interactive.activity import activity_parts
from spice.interactive.tool_presentation import tool_presentation
from spice.tools.base import ToolResult


def _history_preview(text: str, *, limit: int = 100) -> str:
    preview = " ".join(text.split())
    return preview if len(preview) <= limit else f"{preview[: limit - 1]}…"


class Composer(TextArea):
    """A multiline composer where Enter submits and modified Enter adds a line."""

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter,alt+enter", "newline", "New line", priority=True),
        Binding("tab", "complete", "Complete", priority=True),
        Binding("up", "navigate_up", "Previous", priority=True),
        Binding("down", "navigate_down", "Next", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class CompletionRequested(Message):
        pass

    class NavigationRequested(Message):
        def __init__(self, direction: int) -> None:
            super().__init__()
            self.direction = direction

    def action_submit(self) -> None:
        value = self.text.strip()
        self.post_message(self.Submitted(value))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_complete(self) -> None:
        self.post_message(self.CompletionRequested())

    def action_navigate_up(self) -> None:
        self.post_message(self.NavigationRequested(-1))

    def action_navigate_down(self) -> None:
        self.post_message(self.NavigationRequested(1))

    def move_cursor_vertically(self, direction: int) -> None:
        if direction < 0:
            super().action_cursor_up()
        else:
            super().action_cursor_down()


class CommandSuggestions(OptionList):
    """Keyboard and mouse suggestions for slash commands or input history."""

    class Accepted(Message):
        def __init__(self, trigger: str) -> None:
            super().__init__()
            self.trigger = trigger

    class HistoryAccepted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self) -> None:
        super().__init__(id="command-suggestions", markup=False, compact=True)
        self.mode = "commands"
        self._history_values: dict[str, str] = {}

    def show_commands(self, commands: Iterable[tuple[str, str, str]]) -> None:
        self.mode = "commands"
        self._history_values.clear()
        options = [
            Option(f"{usage}  —  {description}", id=trigger)
            for trigger, usage, description in commands
        ]
        self.set_options(options)
        self.display = bool(options)
        if options:
            self.highlighted = 0

    def show_history(self, questions: Iterable[str]) -> None:
        self.mode = "history"
        values = list(questions)[-7:]
        self._history_values = {
            f"history-{index}": value for index, value in enumerate(values)
        }
        options = [
            Option(_history_preview(value), id=option_id)
            for option_id, value in self._history_values.items()
        ]
        self.set_options(options)
        self.display = bool(options)
        if options:
            self.highlighted = len(options) - 1

    def accept_highlighted(self) -> bool:
        option = self.highlighted_option
        if option is None or option.id is None:
            return False
        history_value = self._history_values.get(option.id)
        if self.mode == "history" and history_value is not None:
            self.post_message(self.HistoryAccepted(history_value))
        else:
            self.post_message(self.Accepted(option.id))
        return True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            history_value = self._history_values.get(event.option_id)
            if self.mode == "history" and history_value is not None:
                self.post_message(self.HistoryAccepted(history_value))
            else:
                self.post_message(self.Accepted(event.option_id))


def _user_prompt_content(text: str) -> Content:
    """Render a compact terminal prompt marker without reserving a role column."""
    marker = ">\N{NO-BREAK SPACE}\N{NO-BREAK SPACE}"
    return Content(marker, spans=[Span(0, 1, "bold #67e8f9")]) + Content(text)


class UserMessage(Static):
    def __init__(self, text: str) -> None:
        super().__init__(
            _user_prompt_content(text), classes="message user-message", markup=False
        )


class AssistantMessage(Markdown):
    def __init__(self) -> None:
        super().__init__("", classes="message assistant-message assistant-body")
        self.text = ""

    def append_delta(self, delta: str) -> None:
        self.text += delta
        self.update(self.text)

    def settle(self, text: str | None = None) -> None:
        if text and not self.text:
            self.text = text
        self.update(self.text)
        self.add_class("settled")


class ReasoningMessage(Collapsible):
    """Provider reasoning that auto-collapses after streaming completes."""

    def __init__(self, *, kind: str = "reasoning") -> None:
        self.text = ""
        self.kind = kind
        self.started_at = time.monotonic()
        title = "Thinking" if kind == "reasoning" else "Reasoning summary"
        super().__init__(
            Static("", classes="reasoning-body", markup=False),
            title=title,
            collapsed=False,
            classes="reasoning-message",
        )

    def append_delta(self, delta: str) -> None:
        self.text += delta
        self.query_one(".reasoning-body", Static).update(self.text)

    def settle(self) -> None:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        duration = "<1s" if elapsed < 1 else f"{round(elapsed)}s"
        noun = "Thought" if self.kind == "reasoning" else "Reasoning summary"
        self.title = f"{noun} for {duration} · click to expand"
        self.collapsed = True
        self.add_class("settled")


class ToolMessage(Collapsible):
    """One stable, expandable container for a tool call and its result."""

    def __init__(self, tool_call_id: str, tool_name: str, args: dict) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.args = args
        self.output = ""
        self.started_at = time.monotonic()
        presentation = tool_presentation(tool_name)
        self.presentation = presentation
        args_text = _format_args(args)
        super().__init__(
            Static(args_text, classes="tool-args", markup=False),
            Static("", classes="tool-output", markup=False),
            title=f"{presentation.icon} Running {presentation.label.lower()}",
            collapsed=not bool(args_text),
            classes=f"tool-message tool-{presentation.category}",
        )

    def append_update(self, text: str) -> None:
        self.output += text
        preview = _tail(self.output, max_lines=12, max_chars=1800)
        self.query_one(".tool-output", Static).update(preview)
        self.collapsed = False

    def settle(self, summary: str, result: ToolResult) -> None:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        mark = "✗" if result.is_error else "✓"
        self.title = (
            f"{self.presentation.icon} {self.presentation.label} · {mark} "
            f"{summary} · {_format_duration(elapsed)}"
        )
        if result.content and not self.output:
            self.query_one(".tool-output", Static).update(
                _tail(result.content, max_lines=12, max_chars=1800)
            )
        self.set_class(result.is_error, "failed")
        self.set_class(not result.is_error, "succeeded")
        self.collapsed = not result.is_error


class NoticeMessage(Static):
    def __init__(self, text: str, *, kind: str = "info") -> None:
        super().__init__(text, classes=f"notice notice-{kind}", markup=False)


class ActivityLane(Static):
    """Stable activity region; animation never mutates the transcript."""

    def __init__(self) -> None:
        super().__init__("", id="activity-lane", markup=False)
        self.label = ""
        self.detail = ""
        self.started_at: float | None = None

    def on_mount(self) -> None:
        self.display = False
        self.set_interval(0.125, self._tick)

    def start(self, label: str, detail: str = "") -> None:
        if label != self.label or detail != self.detail or self.started_at is None:
            self.started_at = time.monotonic()
        self.label = label
        self.detail = detail
        self.display = True
        self._tick()

    def stop(self) -> None:
        self.started_at = None
        self.label = ""
        self.detail = ""
        self.display = False
        self.update("")

    def _tick(self) -> None:
        if self.started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self.started_at)
        marker, status = activity_parts(self.label, elapsed)
        suffix = f" · {self.detail}" if self.detail else ""
        self.update(f"{marker} {status}{suffix}    Esc stop")


class QueueStrip(Static):
    def __init__(self) -> None:
        super().__init__("", id="queue-strip", markup=False)

    def show_messages(self, messages: Iterable[str]) -> None:
        values = list(messages)
        if not values:
            self.display = False
            self.update("")
            return
        self.display = True
        preview = values[0].replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:89] + "…"
        self.update(f"Queued {len(values)} · next: {preview}")


class StatusBar(Static):
    def __init__(self) -> None:
        super().__init__("", id="status-bar", markup=False)

    def set_status(
        self,
        *,
        cwd: str,
        model: str,
        protocol: str,
        permission: str,
    ) -> None:
        cwd = _compact_path(cwd)
        if self.size.width and self.size.width < 70:
            model = model.rsplit("/", 1)[-1]
            self.update(f"{model}  ·  permissions: {permission}  ·  /help")
            return
        self.update(
            f"{cwd}  ·  {model}  ·  {protocol}  ·  permissions: {permission}  ·  /help"
        )


def _format_args(args: dict) -> str:
    if not args:
        return ""
    try:
        rendered = json.dumps(args, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = str(args)
    return _tail(rendered, max_lines=10, max_chars=1400)


def _tail(value: str, *, max_lines: int, max_chars: int) -> str:
    value = value.strip()
    lines = value.splitlines()
    if len(lines) > max_lines:
        value = "\n".join(
            [*lines[: max_lines - 1], f"… {len(lines) - max_lines + 1} more lines"]
        )
    if len(value) > max_chars:
        value = value[: max_chars - 1] + "…"
    return value


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1000):d}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes}m{remaining:02d}s"


def _compact_path(value: str) -> str:
    path = Path(value)
    parts = path.parts
    if len(parts) <= 3:
        return value
    return f"…/{'/'.join(parts[-2:])}"
