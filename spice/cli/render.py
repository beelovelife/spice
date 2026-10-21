"""Render agent events in the CLI."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    TurnEndEvent,
    TurnFinalizationStartEvent,
    TurnStartEvent,
)
from spice.interactive.activity import activity_text
from spice.agent.turn_outcome import pause_notice
from spice.interactive.confirm import ConfirmPolicy, is_file_edit_tool
from spice.interactive.types import ChoiceRequest, ConfirmRequest
from spice.tools.base import ToolResult
from spice.cli.tool_render import (
    _confirmation_question,
    _format_error_summary,
    _is_file_edit_tool,
    _is_read_only_bash_command,
    _preview_text,
    _preview_tool_result,
    _select_confirmation,
    _todo_display_lines,
    _todo_items_from_result,
    _todo_marker,
    _todo_status_line,
    _tool_result_style,
    format_tool_end,
    format_tool_start,
)


_TABLE_DIVIDER_RE = re.compile(r"^\s*:?-+:?\s*$")
_ARGS_PREVIEW_CHARS = 700
_ARGS_PREVIEW_LINES = 8
_SUMMARY_INDENT = "  "
COMPACTABLE_TOOL_NAMES = {"list_dir", "read_file", "read_files", "search_files"}

# Compatibility exports for the legacy TUI and downstream integrations. New
# code should import pure tool rendering helpers from spice.cli.tool_render.
__all__ = [
    "COMPACTABLE_TOOL_NAMES",
    "CompactToolGroup",
    "CliRenderer",
    "format_tool_start",
    "format_tool_end",
    "_build_table",
    "_confirmation_question",
    "_format_error_summary",
    "_is_file_edit_tool",
    "_is_read_only_bash_command",
    "_is_table_divider",
    "_looks_like_table_line",
    "_preview_text",
    "_preview_tool_result",
    "_select_confirmation",
    "_split_table_cells",
    "_todo_display_lines",
    "_todo_items_from_result",
    "_todo_marker",
]


@dataclass
class CompactToolGroup:
    tool_name: str
    count: int = 0
    failures: int = 0
    last_start: str = ""
    last_end: str = ""
    rendered_lines: int = 0


@dataclass
class TodoStatusLine:
    text: str = ""
    rendered_lines: int = 0


class _CliToolConfirmPort:
    """Render one tool confirmation while ConfirmPolicy owns serialization."""

    def __init__(self, renderer: "CliRenderer", tool_name: str, args: dict) -> None:
        self.renderer = renderer
        self.tool_name = tool_name
        self.args = args

    async def choose(self, request: ChoiceRequest) -> str | None:
        return None

    async def confirm(self, request: ConfirmRequest) -> str:
        self.renderer._finish_stream_before_block()
        if not is_file_edit_tool(self.tool_name):
            args_preview, folded = _preview_text(
                str(self.args),
                max_chars=_ARGS_PREVIEW_CHARS,
                max_lines=_ARGS_PREVIEW_LINES,
            )
            title = f"[yellow]Confirm {self.tool_name}{' (preview)' if folded else ''}[/yellow]"
            self.renderer.console.print(
                Panel(args_preview, title=title, border_style="yellow")
            )
        return (
            await _select_confirmation(
                request.question,
                choices=request.choices,
            )
            or "deny"
        )


class CliRenderer:
    def __init__(self, console: Console, *, markdown: bool = True) -> None:
        self.console = console
        self.markdown = markdown
        self.confirm_policy = ConfirmPolicy()
        self._streaming = False
        self._line_buffer = ""
        self._plain_line_streaming = False
        self._table_buffer: list[str] = []
        self._line_open = False
        self._pending_tool_args: dict[str, dict] = {}
        self._compact_tool_group: CompactToolGroup | None = None
        self._todo_status = TodoStatusLine()
        self._waiting_started: float | None = None
        self._waiting_label = "Waiting for model"
        self._waiting_rendered_lines = 0
        self._reasoning_streaming = False
        self._reasoning_buffer = ""
        self._reasoning_started: float | None = None
        self._reasoning_live: Live | None = None
        self.last_reasoning = ""
        self.last_reasoning_seconds = 0
        self._stream_text_seen = False

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

    def allow_file_edits_for_session(self) -> None:
        self.confirm_policy.allow_file_edits = True

    def reset_confirm_policy(self) -> None:
        self.confirm_policy = ConfirmPolicy()

    async def confirm(self, tool_name: str, args: dict) -> bool:
        return await self.confirm_policy.confirm(
            tool_name,
            args,
            _CliToolConfirmPort(self, tool_name, args),
        )

    def finish_response(self) -> None:
        """Leave the terminal ready for the next prompt."""
        self._stop_waiting_indicator(clear=True)
        self._finish_reasoning_stream()
        self._flush_compact_tool_group()
        if self._streaming:
            self._finish_stream()
            self._streaming = False
        else:
            self._ensure_newline()

    def refresh_waiting_indicator(self) -> None:
        self._render_waiting_indicator()

    def render_event(self, event: AgentEvent) -> None:
        if isinstance(event, (AgentStartEvent, AgentEndEvent)):
            return
        if isinstance(event, TurnStartEvent):
            self._flush_compact_tool_group()
            self._todo_status = TodoStatusLine()
            self._reset_markdown_stream()
            self._start_waiting_indicator("Waiting for model")
        elif isinstance(event, ReasoningDeltaEvent):
            self._stop_waiting_indicator(clear=True)
            self._flush_compact_tool_group()
            self._freeze_todo_status()
            if self.console.is_terminal:
                if not self._reasoning_streaming:
                    self._begin_reasoning_stream()
                self._reasoning_buffer += event.text
                self._update_reasoning_stream()
        elif isinstance(event, TextDeltaEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._flush_compact_tool_group()
            self._freeze_todo_status()
            if not self._streaming:
                self._begin_response_stream()
            self._stream_text_seen = True
            if self.markdown:
                self._render_markdown_delta(event.text)
            else:
                self.console.print(event.text, end="")
                self._line_open = not event.text.endswith("\n")
        elif isinstance(event, AssistantMessageEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            if event.tool_calls and not (event.text or "").strip():
                self._streaming = False
                return
            self._flush_compact_tool_group()
            self._freeze_todo_status()
            if (event.text or "").strip() and not self._stream_text_seen:
                if not self._streaming:
                    self._begin_response_stream()
                if self.markdown:
                    self._render_markdown_delta(event.text)
                else:
                    self.console.print(event.text, end="")
                    self._line_open = not event.text.endswith("\n")
            if self._streaming:
                self._finish_stream()
                self._streaming = False
        elif isinstance(event, ModelRetryEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_stream_before_block()
            self.console.print(
                "[yellow]Model request failed temporarily. "
                f"Retrying {event.next_attempt}/{event.max_attempts} in {event.delay_seconds:.1f}s "
                f"({event.provider}/{event.model}).[/yellow]"
            )
            self._start_waiting_indicator("Retrying model request")
        elif isinstance(event, ModelFallbackEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_stream_before_block()
            self.console.print(
                "[yellow]Model fallback:[/yellow] "
                f"{event.from_provider}/{event.from_model} -> {event.to_provider}/{event.to_model} "
                f"[dim](this turn, reason: {event.reason})[/dim]"
            )
            self._start_waiting_indicator("Waiting for fallback model")
        elif isinstance(event, ToolExecutionStartEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._finish_stream_before_block()
            args = event.args or {}
            self._pending_tool_args[event.tool_call_id] = args
            if event.tool_name == "update_todo":
                self._start_waiting_indicator(
                    f"Running {event.tool_name}", stream_prefix=False
                )
                return
            self._freeze_todo_status()
            start = format_tool_start(event.tool_name, args)
            if event.tool_name in COMPACTABLE_TOOL_NAMES:
                self._start_compact_tool(event.tool_name, start)
                self._start_waiting_indicator(
                    f"Running {event.tool_name}", stream_prefix=False
                )
                return
            self._flush_compact_tool_group()
            self.console.print(f"[magenta]{start}[/magenta]")
            self._line_open = False
            self._start_waiting_indicator(
                f"Running {event.tool_name}", stream_prefix=False
            )
        elif isinstance(event, ToolExecutionEndEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_stream_before_block()
            had_start = event.tool_call_id in self._pending_tool_args
            args = self._pending_tool_args.pop(event.tool_call_id, {})
            summary = format_tool_end(event.tool_name, args, event.result)
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
            style = _tool_result_style(event.result)
            if event.tool_name == "update_todo" and not event.result.is_error:
                self._render_todo_status(event.result)
            else:
                self._freeze_todo_status()
                self.console.print(
                    f"{_SUMMARY_INDENT}[{style}]{escape(summary)}[/{style}]"
                )
            self._line_open = False
        elif isinstance(event, RoundCompleteEvent):
            self._finish_reasoning_stream()
            self._start_waiting_indicator("Waiting for model")
        elif isinstance(event, TurnFinalizationStartEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._finish_stream_before_block()
            self._start_waiting_indicator("整理进度")
        elif isinstance(event, AgentErrorEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._flush_compact_tool_group()
            if self._streaming:
                self._finish_stream(blank_line=False)
                self._streaming = False
            if event.kind == "fatal_tool":
                self.console.print(
                    f"[bold yellow]Current turn stopped:[/bold yellow] {event.message}"
                )
            elif event.kind == "user_interrupted":
                self.console.print(
                    f"[bold yellow]Interrupted:[/bold yellow] {event.message}"
                )
            else:
                self.console.print(f"[bold red]Error:[/bold red] {event.message}")
            self._line_open = False
        elif isinstance(event, TurnEndEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_reasoning_stream()
            self._flush_compact_tool_group()
            if self._streaming:
                self._finish_stream()
                self._streaming = False
            self._todo_status = TodoStatusLine()
            if event.status == "paused":
                notice = pause_notice(event.checkpoint.round_limit if event.checkpoint else None)
                self.console.print(f"[yellow]{escape(notice)}[/yellow]")

    def _begin_response_stream(self) -> None:
        self._streaming = True

    def _start_waiting_indicator(
        self, label: str, *, stream_prefix: bool = True
    ) -> None:
        self._waiting_started = time.monotonic()
        self._waiting_label = label
        self._stream_text_seen = False
        if self.console.is_terminal:
            self._render_waiting_indicator()
        elif stream_prefix:
            self._begin_response_stream()

    def _render_waiting_indicator(self) -> None:
        if self._waiting_started is None or not self.console.is_terminal:
            return
        if self._waiting_rendered_lines:
            self._clear_rendered_tool_lines(self._waiting_rendered_lines)
        elapsed = max(0.0, time.monotonic() - self._waiting_started)
        status = activity_text(self._waiting_label, elapsed)
        self.console.print(f"[italic yellow]{status}[/italic yellow]")
        self._waiting_rendered_lines = 1
        self._line_open = False

    def _stop_waiting_indicator(self, *, clear: bool) -> None:
        if self._waiting_started is None:
            return
        if clear and self._waiting_rendered_lines:
            self._clear_rendered_tool_lines(self._waiting_rendered_lines)
        self._waiting_started = None
        self._waiting_rendered_lines = 0

    def _finish_reasoning_stream(self) -> None:
        if not self._reasoning_streaming:
            return
        elapsed = max(
            0.0, time.monotonic() - (self._reasoning_started or time.monotonic())
        )
        live = self._reasoning_live
        if live is not None:
            live.stop()
        self.last_reasoning = self._reasoning_buffer.strip()
        self.last_reasoning_seconds = int(round(elapsed))
        if self.last_reasoning:
            duration = "<1s" if elapsed < 1 else f"{self.last_reasoning_seconds}s"
            self.console.print(
                f"[dim]▶ Thought for {duration} · type /thinking to view[/dim]"
            )
        self._reasoning_live = None
        self._reasoning_started = None
        self._reasoning_buffer = ""
        self._line_open = False
        self._reasoning_streaming = False

    def _begin_reasoning_stream(self) -> None:
        self._reasoning_streaming = True
        self._reasoning_buffer = ""
        self._reasoning_started = time.monotonic()
        self._line_open = False
        live = Live(
            self._reasoning_renderable(),
            console=self.console,
            refresh_per_second=12,
            transient=True,
            vertical_overflow="visible",
        )
        self._reasoning_live = live
        live.start(refresh=True)

    def _update_reasoning_stream(self) -> None:
        if self._reasoning_live is not None:
            self._reasoning_live.update(self._reasoning_renderable(), refresh=True)

    def _reasoning_renderable(self) -> Text:
        text = Text()
        text.append("Thinking  ", style="italic cyan")
        text.append(self._reasoning_buffer or "…", style="dim")
        return text

    def _render_todo_status(self, result: ToolResult) -> None:
        line = _todo_status_line(result)
        if not line:
            return
        if self.console.is_terminal and self._todo_status.rendered_lines:
            self._clear_rendered_tool_lines(self._todo_status.rendered_lines)
        self.console.print(f"[green]{escape(line)}[/green]")
        self._todo_status = TodoStatusLine(
            text=line, rendered_lines=1 if self.console.is_terminal else 0
        )
        self._line_open = False

    def _freeze_todo_status(self) -> None:
        self._todo_status.rendered_lines = 0

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
        self._render_live_compact_tool_group()

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
        self._render_live_compact_tool_group()

    def _flush_compact_tool_group(self) -> None:
        group = self._compact_tool_group
        if group is None:
            return
        self._compact_tool_group = None
        if not group.last_start and not group.last_end:
            return
        if group.rendered_lines:
            self._line_open = False
            return
        self._print_compact_tool_group(group)

    def _render_live_compact_tool_group(self) -> None:
        group = self._compact_tool_group
        if group is None or not self.console.is_terminal:
            return
        if group.rendered_lines:
            self._clear_rendered_tool_lines(group.rendered_lines)
        lines = self._compact_tool_lines(group)
        style = "red" if group.failures else "green"
        self.console.print(f"[magenta]{escape(lines[0])}[/magenta]")
        self.console.print(f"{_SUMMARY_INDENT}[{style}]{escape(lines[1])}[/{style}]")
        group.rendered_lines = 2
        self._line_open = False

    def _clear_rendered_tool_lines(self, count: int) -> None:
        file = self.console.file
        width = max(self.console.width, 1)
        blanks = " " * width
        try:
            for _ in range(count):
                file.write("\x1b[1A")
                file.write("\r")
                file.write(blanks)
                file.write("\r")
            file.flush()
        except (OSError, ValueError):
            pass

    def _print_compact_tool_group(self, group: CompactToolGroup) -> None:
        lines = self._compact_tool_lines(group)
        self.console.print(f"[magenta]{escape(lines[0])}[/magenta]")
        style = "red" if group.failures else "green"
        self.console.print(f"{_SUMMARY_INDENT}[{style}]{escape(lines[1])}[/{style}]")
        self._line_open = False

    def _compact_tool_lines(self, group: CompactToolGroup) -> tuple[str, str]:
        start = group.last_start or f"❯ {group.tool_name}"
        if group.count > 1:
            start = f"{start} · {group.count} tool calls"
        summary = group.last_end or "… running"
        if group.count > 1:
            summary = f"{summary} · {group.count} {group.tool_name} calls"
        if group.failures:
            summary = f"{summary} · {group.failures} failed"
        return start, summary

    def _finish_stream(self, *, blank_line: bool = True) -> None:
        if self.markdown:
            self._finish_markdown_stream()
        else:
            self._ensure_newline()
        if blank_line:
            self._write_newline()
            self._line_open = False

    def _finish_stream_before_block(self) -> None:
        if self._streaming:
            self._finish_stream(blank_line=False)
            self._streaming = False
        else:
            self._ensure_newline()

    def _reset_markdown_stream(self) -> None:
        self._line_buffer = ""
        self._plain_line_streaming = False
        self._table_buffer = []

    def _render_markdown_delta(self, text: str) -> None:
        for char in text:
            self._render_markdown_char(char)

    def _render_markdown_char(self, char: str) -> None:
        self._line_buffer += char

        if self._plain_line_streaming:
            if char == "\n":
                self._write_newline()
                self._line_buffer = ""
                self._plain_line_streaming = False
                self._line_open = False
            else:
                self.console.print(char, end="")
                self._line_open = True
            return

        if char == "\n":
            line = self._line_buffer[:-1]
            self._line_buffer = ""
            self._process_markdown_line(line)
            return

        if self._table_buffer:
            return

        stripped = self._line_buffer.lstrip()
        if stripped and not stripped.startswith("|"):
            self.console.print(self._line_buffer, end="")
            self._line_open = True
            self._line_buffer = ""
            self._plain_line_streaming = True

    def _process_markdown_line(self, line: str) -> None:
        if self._table_buffer:
            if len(self._table_buffer) == 1:
                if _is_table_divider(line):
                    self._table_buffer.append(line)
                    return
                self._flush_pending_table_as_text()
                self._process_markdown_line(line)
                return
            if _looks_like_table_line(line):
                self._table_buffer.append(line)
                return
            self._flush_table()
            self._process_markdown_line(line)
            return

        if _looks_like_table_line(line):
            self._table_buffer.append(line)
            return
        self.console.print(line)
        self._line_open = False

    def _finish_markdown_stream(self) -> None:
        if self._plain_line_streaming:
            self._plain_line_streaming = False
            self._line_buffer = ""
        elif self._line_buffer:
            line = self._line_buffer
            self._line_buffer = ""
            self._process_markdown_line(line)

        if len(self._table_buffer) >= 2:
            self._flush_table()
        else:
            self._flush_pending_table_as_text()
        self._ensure_newline()

    def _flush_table(self) -> None:
        if not self._table_buffer:
            return
        self.console.print(_build_table(self._table_buffer))
        self._line_open = False
        self._table_buffer = []

    def _flush_pending_table_as_text(self) -> None:
        for line in self._table_buffer:
            self.console.print(line)
            self._line_open = False
        self._table_buffer = []

    def _ensure_newline(self) -> None:
        if not self._line_open:
            return
        self._write_newline()
        self._line_open = False

    def _write_newline(self) -> None:
        self.console.file.write("\r\n")
        self.console.file.flush()


def _looks_like_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    return stripped.count("|") >= 2


def _is_table_divider(line: str) -> bool:
    cells = _split_table_cells(line)
    return bool(cells) and all(_TABLE_DIVIDER_RE.match(cell) for cell in cells)


def _split_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _build_table(lines: list[str]) -> Table:
    header = _split_table_cells(lines[0])
    rows = [_split_table_cells(line) for line in lines[2:]]
    column_count = (
        max(len(header), *(len(row) for row in rows)) if rows else len(header)
    )
    header += [""] * (column_count - len(header))

    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=False)
    for column in header:
        table.add_column(column, overflow="fold")
    for row in rows:
        cells = row + [""] * (column_count - len(row))
        table.add_row(*cells)
    return table
