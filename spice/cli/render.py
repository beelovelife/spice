"""Render agent events in the CLI."""

from __future__ import annotations

import json
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.tools.base import ToolResult


_TABLE_DIVIDER_RE = re.compile(r"^\s*:?-+:?\s*$")
_PREVIEW_FIRST_TOOLS = {"read_file", "search_files"}
_DEFAULT_PREVIEW_CHARS = 1600
_DEFAULT_PREVIEW_LINES = 30
_COMPACT_PREVIEW_CHARS = 900
_COMPACT_PREVIEW_LINES = 12
_ARGS_PREVIEW_CHARS = 700
_ARGS_PREVIEW_LINES = 8
_SELECT_BG = "bg:#3f4a68"
_SUMMARY_INDENT = "  "
_TOOL_ARG_LIMIT = 80
COMPACTABLE_TOOL_NAMES = {"list_dir", "read_file", "read_files", "search_files"}


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


class CliRenderer:
    def __init__(self, console: Console, *, markdown: bool = True) -> None:
        self.console = console
        self.markdown = markdown
        self._allow_file_edits = False
        self._streaming = False
        self._line_buffer = ""
        self._plain_line_streaming = False
        self._table_buffer: list[str] = []
        self._line_open = False
        self._pending_tool_args: dict[str, dict] = {}
        self._allow_read_only_bash = False
        self._compact_tool_group: CompactToolGroup | None = None
        self._todo_status = TodoStatusLine()
        self._waiting_started: float | None = None
        self._waiting_rendered_lines = 0
        self._stream_text_seen = False

    def allow_file_edits_for_session(self) -> None:
        self._allow_file_edits = True

    async def confirm(self, tool_name: str, args: dict) -> bool:
        if _is_file_edit_tool(tool_name) and self._allow_file_edits:
            return True
        if tool_name == "bash" and self._allow_read_only_bash and _is_read_only_bash_command(args):
            return True

        self._finish_stream_before_block()
        question = _confirmation_question(tool_name, args)
        if not _is_file_edit_tool(tool_name):
            args_preview, folded = _preview_text(str(args), max_chars=_ARGS_PREVIEW_CHARS, max_lines=_ARGS_PREVIEW_LINES)
            title = f"[yellow]Confirm {tool_name}{' (preview)' if folded else ''}[/yellow]"
            self.console.print(Panel(args_preview, title=title, border_style="yellow"))

        selected = await _select_confirmation(
            question,
            allow_all_edits=_is_file_edit_tool(tool_name),
            allow_read_only_bash=tool_name == "bash" and _is_read_only_bash_command(args),
        )
        if selected == "allow_all_edits":
            self._allow_file_edits = True
            return True
        if selected == "allow_read_only_bash":
            self._allow_read_only_bash = True
            return True
        return selected == "allow"

    def finish_response(self) -> None:
        """Leave the terminal ready for the next prompt."""
        self._stop_waiting_indicator(clear=True)
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
            self._start_waiting_indicator()
        elif isinstance(event, TextDeltaEvent):
            self._stop_waiting_indicator(clear=True)
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
        elif isinstance(event, ToolExecutionStartEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_stream_before_block()
            args = event.args or {}
            self._pending_tool_args[event.tool_call_id] = args
            if event.tool_name == "update_todo":
                return
            self._freeze_todo_status()
            start = format_tool_start(event.tool_name, args)
            if event.tool_name in COMPACTABLE_TOOL_NAMES:
                self._start_compact_tool(event.tool_name, start)
                return
            self._flush_compact_tool_group()
            self.console.print(f"[magenta]{start}[/magenta]")
            self._line_open = False
        elif isinstance(event, ToolExecutionEndEvent):
            self._stop_waiting_indicator(clear=True)
            self._finish_stream_before_block()
            had_start = event.tool_call_id in self._pending_tool_args
            args = self._pending_tool_args.pop(event.tool_call_id, {})
            summary = format_tool_end(event.tool_name, args, event.result)
            if had_start and event.tool_name in COMPACTABLE_TOOL_NAMES and self._compact_tool_group is not None:
                self._finish_compact_tool(event.tool_name, summary, is_error=event.result.is_error)
                return
            self._flush_compact_tool_group()
            style = _tool_result_style(event.result)
            if event.tool_name == "update_todo" and not event.result.is_error:
                self._render_todo_status(event.result)
            else:
                self._freeze_todo_status()
                self.console.print(f"{_SUMMARY_INDENT}[{style}]{escape(summary)}[/{style}]")
            self._line_open = False
        elif isinstance(event, AgentErrorEvent):
            self._stop_waiting_indicator(clear=True)
            self._flush_compact_tool_group()
            if self._streaming:
                self._finish_stream(blank_line=False)
                self._streaming = False
            self.console.print(f"[bold red]Error:[/bold red] {event.message}")
            self._line_open = False
        elif isinstance(event, TurnEndEvent):
            self._stop_waiting_indicator(clear=True)
            self._flush_compact_tool_group()
            if self._streaming:
                self._finish_stream()
                self._streaming = False
            self._todo_status = TodoStatusLine()

    def _begin_response_stream(self) -> None:
        self.console.print("[bold cyan]Spice:[/bold cyan] ", end="")
        self._line_open = True
        self._streaming = True

    def _start_waiting_indicator(self) -> None:
        self._waiting_started = time.monotonic()
        self._stream_text_seen = False
        if self.console.is_terminal:
            self._render_waiting_indicator()
        else:
            self._begin_response_stream()

    def _render_waiting_indicator(self) -> None:
        if self._waiting_started is None or not self.console.is_terminal:
            return
        if self._waiting_rendered_lines:
            self._clear_rendered_tool_lines(self._waiting_rendered_lines)
        elapsed = max(0, int(time.monotonic() - self._waiting_started))
        self.console.print(f"[bold cyan]Spice:[/bold cyan] [italic yellow]thinking... {_format_elapsed(elapsed)}[/italic yellow]")
        self._waiting_rendered_lines = 1
        self._line_open = False

    def _stop_waiting_indicator(self, *, clear: bool) -> None:
        if self._waiting_started is None:
            return
        if clear and self._waiting_rendered_lines:
            self._clear_rendered_tool_lines(self._waiting_rendered_lines)
        self._waiting_started = None
        self._waiting_rendered_lines = 0

    def _render_todo_status(self, result: ToolResult) -> None:
        line = _todo_status_line(result)
        if not line:
            return
        if self.console.is_terminal and self._todo_status.rendered_lines:
            self._clear_rendered_tool_lines(self._todo_status.rendered_lines)
        self.console.print(f"[green]{escape(line)}[/green]")
        self._todo_status = TodoStatusLine(text=line, rendered_lines=1 if self.console.is_terminal else 0)
        self._line_open = False

    def _freeze_todo_status(self) -> None:
        self._todo_status.rendered_lines = 0

    def _start_compact_tool(self, tool_name: str, start: str) -> None:
        if self._compact_tool_group is not None and self._compact_tool_group.tool_name != tool_name:
            self._flush_compact_tool_group()
        if self._compact_tool_group is None:
            self._compact_tool_group = CompactToolGroup(tool_name=tool_name)
        self._compact_tool_group.count += 1
        self._compact_tool_group.last_start = start
        self._compact_tool_group.last_end = ""
        self._render_live_compact_tool_group()

    def _finish_compact_tool(self, tool_name: str, summary: str, *, is_error: bool) -> None:
        if self._compact_tool_group is None or self._compact_tool_group.tool_name != tool_name:
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


def _format_elapsed(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining = divmod(seconds, 60)
    return f"{minutes}m{remaining:02d}s"


def _build_table(lines: list[str]) -> Table:
    header = _split_table_cells(lines[0])
    rows = [_split_table_cells(line) for line in lines[2:]]
    column_count = max(len(header), *(len(row) for row in rows)) if rows else len(header)
    header += [""] * (column_count - len(header))

    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=False)
    for column in header:
        table.add_column(column, overflow="fold")
    for row in rows:
        cells = row + [""] * (column_count - len(row))
        table.add_row(*cells)
    return table


@dataclass(frozen=True)
class _ToolSpec:
    start: Callable[[dict], str]
    end: Callable[[dict, ToolResult], str]


def format_tool_start(tool_name: str, args: dict) -> str:
    """Single-line description shown when a tool starts running."""
    spec = _TOOL_SPECS.get(tool_name)
    if spec is not None:
        try:
            return spec.start(args or {})
        except Exception:  # noqa: BLE001 - never break rendering
            pass
    target = _summarize_args(args or {})
    return f"❯ {tool_name}{(' ' + target) if target else ''}"


def format_tool_end(tool_name: str, args: dict | None, result: ToolResult) -> str:
    """Single-line summary shown when a tool finishes."""
    if result.is_error:
        return _format_error_summary(result)
    spec = _TOOL_SPECS.get(tool_name)
    if spec is not None:
        try:
            return spec.end(args or {}, result)
        except Exception:  # noqa: BLE001 - never break rendering
            pass
    lines = _line_count(result.content)
    if lines:
        return f"✓ {lines} lines"
    return "✓ done"


def _line_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _detail_int(result: ToolResult, key: str, default):
    if not isinstance(result.details, dict):
        return default
    value = result.details.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _detail_bool(result: ToolResult, key: str) -> bool:
    return bool(result.details.get(key)) if isinstance(result.details, dict) else False


def _plural(count: int, singular: str) -> str:
    if count == 1:
        return singular
    if singular == "entry":
        return "entries"
    return f"{singular}s"


def _diff_counts(diff: str) -> tuple[int, int]:
    plus = minus = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            plus += 1
        elif line.startswith("-"):
            minus += 1
    return plus, minus


def _truncate_inline(value: str, *, limit: int = _TOOL_ARG_LIMIT) -> str:
    cleaned = value.replace("\n", " ").replace("\r", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)] + "…"


def _summarize_args(args: dict) -> str:
    for key in ("path", "file_path", "name", "command", "query", "pattern", "url"):
        value = args.get(key)
        if value:
            return _truncate_inline(str(value))
    return ""


def _format_error_summary(result: ToolResult) -> str:
    guidance = _format_guidance_summary(result)
    if guidance is not None:
        return guidance
    text = (result.content or "").strip()
    if "Traceback (most recent call last):" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return f"✗ {_truncate_inline(lines[-1], limit=120)}"
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return "✗ failed"
    return f"✗ {_truncate_inline(first_line, limit=120)}"


def _tool_result_style(result: ToolResult) -> str:
    if result.is_error and _format_guidance_summary(result) is not None:
        return "yellow"
    return "red" if result.is_error else "green"


def _format_guidance_summary(result: ToolResult) -> str | None:
    if result.details.get("presentation") == "guidance":
        first_line = next((line for line in (result.content or "").splitlines() if line.strip()), "")
        return f"! {_humanize_file_state_guard(first_line)}" if first_line else "! Action needed"
    text = (result.content or "").strip()
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if first_line.startswith("read_file must be called first before "):
        return f"! {_humanize_file_state_guard(first_line)}"
    if first_line.startswith("Full read_file must be called before "):
        return f"! {_humanize_file_state_guard(first_line)}"
    return None


def _humanize_file_state_guard(message: str) -> str:
    if message.startswith("read_file must be called first before overwriting "):
        path = message.removeprefix("read_file must be called first before overwriting ").rstrip(".")
        return f"Read file first before overwriting {path}"
    if message.startswith("read_file must be called first before editing "):
        path = message.removeprefix("read_file must be called first before editing ").rstrip(".")
        return f"Read file first before editing {path}"
    if message.startswith("Full read_file must be called before editing "):
        rest = message.removeprefix("Full read_file must be called before editing ")
        path = rest.split(";", 1)[0].rstrip(".")
        return f"Read the full file before editing {path}"
    return _truncate_inline(message, limit=120)


def _path_or_default(args: dict, default: str = "<unknown>") -> str:
    raw = args.get("path") or args.get("file_path")
    return str(raw).strip() if raw else default


def _read_file_start(args: dict) -> str:
    return f"❯ Read file: {_path_or_default(args)}"


def _read_file_end(args: dict, result: ToolResult) -> str:
    lines = _detail_int(result, "line_count", _line_count(result.content))
    chars = _detail_int(result, "char_count", len(result.content or ""))
    suffix = " · partial" if _detail_bool(result, "partial") else ""
    return f"✓ Read {lines} {_plural(lines, 'line')} · {chars} chars{suffix}"


def _read_files_start(args: dict) -> str:
    files = args.get("files")
    count = len(files) if isinstance(files, list) else 0
    return f"❯ Read {count} files"


def _read_files_end(args: dict, result: ToolResult) -> str:
    files = args.get("files")
    requested = len(files) if isinstance(files, list) else 0
    count = _detail_int(result, "file_count", requested)
    lines = _detail_int(result, "line_count", _line_count(result.content))
    chars = _detail_int(result, "char_count", len(result.content or ""))
    suffix = " · partial" if _detail_int(result, "partial_count", 0) else ""
    if count:
        return f"✓ Read {count} {_plural(count, 'file')} · {lines} {_plural(lines, 'line')} · {chars} chars{suffix}"
    return f"✓ Read {lines} {_plural(lines, 'line')} · {chars} chars{suffix}"


def _write_file_start(args: dict) -> str:
    return f"❯ Write file: {_path_or_default(args)}"


def _write_file_end(args: dict, result: ToolResult) -> str:
    lines = _detail_int(result, "line_count", _line_count(str(args.get("content") or "")))
    chars = _detail_int(result, "char_count", len(str(args.get("content") or "")))
    if lines or chars:
        return f"✓ Wrote {lines} {_plural(lines, 'line')} · {chars} chars"
    text = (result.content or "").strip()
    return f"✓ {text}" if text else "✓ Wrote file"


def _edit_file_start(args: dict) -> str:
    suffix = " (dry-run)" if args.get("dry_run") else ""
    return f"❯ Edit file: {_path_or_default(args)}{suffix}"


def _edit_file_end(args: dict, result: ToolResult) -> str:
    plus = _detail_int(result, "lines_added", 0)
    minus = _detail_int(result, "lines_removed", 0)
    if not plus and not minus:
        plus, minus = _diff_counts(result.content or "")
    if plus or minus:
        prefix = "✓ Previewed" if args.get("dry_run") else "✓ Edited"
        return f"{prefix} (+{plus} −{minus} lines)"
    return "✓ No changes"


def _apply_patch_start(args: dict) -> str:
    operations = args.get("operations")
    count = len(operations) if isinstance(operations, list) else 0
    suffix = " (dry-run)" if args.get("dry_run") else ""
    if count:
        return f"❯ Apply patch: {count} {_plural(count, 'operation')}{suffix}"
    return f"❯ Apply patch{suffix}"


def _apply_patch_end(args: dict, result: ToolResult) -> str:
    plus = _detail_int(result, "lines_added", 0)
    minus = _detail_int(result, "lines_removed", 0)
    files = _detail_int(result, "files_changed", 0)
    if not plus and not minus:
        plus, minus = _diff_counts(result.content or "")
    prefix = "✓ Previewed patch" if args.get("dry_run") or _detail_bool(result, "dry_run") else "✓ Applied patch"
    file_part = f" · {files} {_plural(files, 'file')}" if files else ""
    if plus or minus:
        return f"{prefix}{file_part} · +{plus} −{minus} lines"
    return f"{prefix}{file_part} · no changes"


def _list_dir_start(args: dict) -> str:
    raw = args.get("path") or "."
    return f"❯ List dir: {raw}"


def _list_dir_end(args: dict, result: ToolResult) -> str:
    total = _detail_int(result, "total_entries", None)
    if total is None:
        text = (result.content or "").strip()
        total = 0 if not text or text == "(empty)" else _line_count(text)
    dirs = _detail_int(result, "dir_count", 0)
    files = _detail_int(result, "file_count", 0)
    other = _detail_int(result, "other_count", 0)
    hidden = _detail_int(result, "hidden_count", 0)
    parts = [f"✓ Listed {total} {_plural(total, 'entry')}"]
    if dirs or files or other:
        parts.append(f"{dirs} {_plural(dirs, 'dir')}")
        parts.append(f"{files} {_plural(files, 'file')}")
        if other:
            parts.append(f"{other} other")
    if hidden:
        parts.append(f"{hidden} hidden")
    return " · ".join(parts)


def _search_files_start(args: dict) -> str:
    pattern = _truncate_inline(str(args.get("pattern") or ""), limit=60)
    raw_path = str(args.get("path") or "").strip()
    suffix = f" in {raw_path}" if raw_path and raw_path != "." else ""
    return f'❯ Search "{pattern}"{suffix}'


def _search_files_end(args: dict, result: ToolResult) -> str:
    matches = _detail_int(result, "match_count", None)
    if matches is None:
        text = (result.content or "").strip()
        matches = 0 if not text or text == "No matches." else _line_count(text)
    files = _detail_int(result, "file_count", 0)
    suffix = f" in {files} {_plural(files, 'file')}" if files else ""
    truncated = " · truncated" if _detail_bool(result, "truncated") else ""
    return f"✓ Found {matches} {_plural(matches, 'match')}{suffix}{truncated}"


def _bash_start(args: dict) -> str:
    command = _truncate_inline(str(args.get("command") or ""))
    return f"❯ Run: {command}"


def _bash_end(args: dict, result: ToolResult) -> str:
    code = 0
    if isinstance(result.details, dict):
        try:
            code = int(result.details.get("returncode") or 0)
        except (TypeError, ValueError):
            code = 0
    stdout_lines = _detail_int(result, "stdout_lines", None)
    stderr_lines = _detail_int(result, "stderr_lines", None)
    if stdout_lines is None and stderr_lines is None:
        lines = _line_count(result.content)
        return f"✓ Exit {code} · {lines} {_plural(lines, 'line')} output"
    stdout_lines = stdout_lines or 0
    stderr_lines = stderr_lines or 0
    suffix = " · truncated" if _detail_bool(result, "output_truncated") else ""
    return f"✓ Exit {code} · stdout {stdout_lines} {_plural(stdout_lines, 'line')} · stderr {stderr_lines} {_plural(stderr_lines, 'line')}{suffix}"


def _web_search_start(args: dict) -> str:
    query = _truncate_inline(str(args.get("query") or ""))
    return f"❯ Web search: {query}"


def _web_search_end(args: dict, result: ToolResult) -> str:
    text = (result.content or "").strip()
    if not text or text == "No results.":
        return "✓ Found 0 results"
    items = sum(1 for line in text.splitlines() if line.startswith("- "))
    return f"✓ Found {items or _line_count(text)} results"


def _get_current_time_start(args: dict) -> str:
    return "❯ Get current time"


def _get_current_time_end(args: dict, result: ToolResult) -> str:
    timestamp = (result.content or "").strip()
    return f"✓ {timestamp}" if timestamp else "✓ done"


def _skills_list_start(args: dict) -> str:
    return "❯ List skills"


def _skills_list_end(args: dict, result: ToolResult) -> str:
    items = sum(1 for line in (result.content or "").splitlines() if line.startswith("- "))
    if items:
        return f"✓ Listed {items} skills"
    return "✓ No skills found"


def _skill_view_start(args: dict) -> str:
    name = str(args.get("name") or "<unknown>").strip() or "<unknown>"
    file_path = args.get("file_path")
    if file_path:
        return f"❯ View skill: {name} · {file_path}"
    return f"❯ View skill: {name}"


def _skill_view_end(args: dict, result: ToolResult) -> str:
    lines = _line_count(result.content)
    return f"✓ Read {lines} {_plural(lines, 'line')}"


def _update_todo_start(args: dict) -> str:
    todos = args.get("todos")
    if not isinstance(todos, list):
        return "❯ Read todo list"
    mode = "merge" if args.get("merge") else "replace"
    return f"❯ Update todo ({mode}, {len(todos)} items)"


def _update_todo_end(args: dict, result: ToolResult) -> str:
    payload = _todo_payload(result)
    todos = payload.get("todos") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(todos, list):
        return "✓ todo updated"
    if not isinstance(summary, dict):
        summary = _todo_summary(todos)
    head = (
        "✓ todo updated · "
        f"{int(summary.get('completed') or 0)}/{int(summary.get('total') or len(todos))} completed · "
        f"{int(summary.get('in_progress') or 0)} in progress · "
        f"{int(summary.get('pending') or 0)} pending"
    )
    lines = [head]
    for item in todos:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "?")
        content = _truncate_inline(str(item.get("content") or "(no description)"), limit=100)
        status = str(item.get("status") or "pending")
        lines.append(f"  {_todo_marker(status)} {item_id}. {content} ({status})")
    return "\n".join(lines)


def _todo_status_line(result: ToolResult) -> str:
    items, summary = _todo_items_from_result(result)
    total = int(summary.get("total") or len(items))
    if total <= 0:
        return ""
    completed = int(summary.get("completed") or 0)
    in_progress = int(summary.get("in_progress") or 0)
    pending = int(summary.get("pending") or 0)
    cancelled = int(summary.get("cancelled") or 0)
    active = next((item for item in items if str(item.get("status") or "pending") == "in_progress"), None)
    if active is None:
        active = next((item for item in items if str(item.get("status") or "pending") == "pending"), None)
    if active is None:
        suffix = "completed" if not cancelled else f"completed · {cancelled} cancelled"
    else:
        suffix = _truncate_inline(str(active.get("content") or "(no description)"), limit=100)
    parts = [f"● todo {completed}/{total}"]
    if in_progress:
        parts.append(f"{in_progress} running")
    if pending:
        parts.append(f"{pending} pending")
    return " · ".join(parts + [suffix])


def _todo_display_lines(result: ToolResult, *, include_summary: bool = False) -> list[str]:
    payload = _todo_payload(result)
    todos = payload.get("todos") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(todos, list):
        return []
    if not isinstance(summary, dict):
        summary = _todo_summary(todos)
    total = int(summary.get("total") or len(todos))
    completed = int(summary.get("completed") or 0)
    in_progress = int(summary.get("in_progress") or 0)
    pending = int(summary.get("pending") or 0)
    cancelled = int(summary.get("cancelled") or 0)
    lines: list[str] = []
    if include_summary:
        parts = [f"{completed}/{total} completed"]
        if in_progress:
            parts.append(f"{in_progress} in progress")
        if pending:
            parts.append(f"{pending} pending")
        if cancelled:
            parts.append(f"{cancelled} cancelled")
        lines.append(" · ".join(parts))
    for item in todos:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "?")
        content = _truncate_inline(str(item.get("content") or "(no description)"), limit=100)
        status = str(item.get("status") or "pending")
        lines.append(f"{_todo_marker(status)} {item_id}. {content}")
    return lines


def _todo_items_from_result(result: ToolResult) -> tuple[list[dict], dict[str, int]]:
    payload = _todo_payload(result)
    todos = payload.get("todos") if isinstance(payload, dict) else None
    if not isinstance(todos, list):
        return [], _todo_summary([])
    items = [item for item in todos if isinstance(item, dict)]
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        summary = _todo_summary(items)
    return items, {
        "total": int(summary.get("total") or len(items)),
        "pending": int(summary.get("pending") or 0),
        "in_progress": int(summary.get("in_progress") or 0),
        "completed": int(summary.get("completed") or 0),
        "cancelled": int(summary.get("cancelled") or 0),
    }


def _todo_payload(result: ToolResult) -> dict:
    if isinstance(result.details, dict) and isinstance(result.details.get("todos"), list):
        return result.details
    try:
        parsed = json.loads(result.content or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _todo_summary(todos: list) -> dict[str, int]:
    counts = {"total": 0, "pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
    for item in todos:
        if not isinstance(item, dict):
            continue
        counts["total"] += 1
        status = str(item.get("status") or "pending")
        if status in counts:
            counts[status] += 1
    return counts


def _todo_marker(status: str) -> str:
    return {
        "completed": "✓",
        "in_progress": "›",
        "pending": "·",
        "cancelled": "×",
    }.get(status, "·")


_TOOL_SPECS: dict[str, _ToolSpec] = {
    "read_file": _ToolSpec(_read_file_start, _read_file_end),
    "read_files": _ToolSpec(_read_files_start, _read_files_end),
    "write_file": _ToolSpec(_write_file_start, _write_file_end),
    "edit_file": _ToolSpec(_edit_file_start, _edit_file_end),
    "apply_patch": _ToolSpec(_apply_patch_start, _apply_patch_end),
    "list_dir": _ToolSpec(_list_dir_start, _list_dir_end),
    "search_files": _ToolSpec(_search_files_start, _search_files_end),
    "bash": _ToolSpec(_bash_start, _bash_end),
    "web_search": _ToolSpec(_web_search_start, _web_search_end),
    "get_current_time": _ToolSpec(_get_current_time_start, _get_current_time_end),
    "skills_list": _ToolSpec(_skills_list_start, _skills_list_end),
    "skill_view": _ToolSpec(_skill_view_start, _skill_view_end),
    "update_todo": _ToolSpec(_update_todo_start, _update_todo_end),
}


def _preview_tool_result(tool_name: str, content: str, *, is_error: bool) -> tuple[str, bool]:
    if not is_error and tool_name in _PREVIEW_FIRST_TOOLS:
        return _preview_text(content, max_chars=_COMPACT_PREVIEW_CHARS, max_lines=_COMPACT_PREVIEW_LINES)
    return _preview_text(content, max_chars=_DEFAULT_PREVIEW_CHARS, max_lines=_DEFAULT_PREVIEW_LINES)


def _preview_text(text: str, *, max_chars: int, max_lines: int) -> tuple[str, bool]:
    lines = text.splitlines()
    folded = len(text) > max_chars or len(lines) > max_lines
    if not folded:
        return text, False

    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip()

    omitted_lines = max(0, len(lines) - len(preview_lines))
    omitted_chars = max(0, len(text) - len(preview))
    note_parts = ["output folded in CLI preview"]
    if omitted_lines:
        note_parts.append(f"{omitted_lines} more lines")
    if omitted_chars:
        note_parts.append(f"{omitted_chars} more chars")
    note_parts.append("full result was still sent to the model")
    return f"{preview}\n\n... {'; '.join(note_parts)}.", True


def _is_file_edit_tool(tool_name: str) -> bool:
    return tool_name in {"write_file", "edit_file", "apply_patch"}


def _confirmation_question(tool_name: str, args: dict) -> str:
    path = str(args.get("path") or "").strip()
    target = path or "the target file"
    if tool_name == "write_file":
        return f"Do you want to write {target}?"
    if tool_name == "edit_file":
        return f"Do you want to edit {target}?"
    if tool_name == "bash":
        if _bash_may_modify_files(args):
            return "Do you want to run this command? It may modify files; prefer edit_file or apply_patch when possible."
        return "Do you want to run this command?"
    return f"Do you want to run {tool_name}?"


def _bash_may_modify_files(args: dict) -> bool:
    command = str(args.get("command") or "").strip().lower()
    if not command:
        return False
    write_markers = (
        "sed -i",
        "perl -pi",
        "python -c",
        "python3 -c",
        " write_text(",
        ".write(",
        "open(",
        ">>",
        ">",
    )
    if not any(marker in command for marker in write_markers):
        return False
    return any(
        marker in command
        for marker in (
            "open(",
            ".write(",
            "write_text(",
            "sed -i",
            "perl -pi",
            ">>",
            ">",
            "mode='w'",
            'mode="w"',
            "', 'w'",
            '", "w"',
            "', 'a'",
            '", "a"',
        )
    )


def _is_read_only_bash_command(args: dict) -> bool:
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    try:
        parts = _shell_command_tokens(command)
    except ValueError:
        return False
    if not parts:
        return False
    if any(part in {";", "&", "||", ">", ">>", "<", "<<", "<<<"} for part in parts):
        return False
    for command_parts in _split_shell_commands(parts):
        if not command_parts:
            return False
        program = command_parts[0].rsplit("/", 1)[-1]
        if program == "cd":
            if not _cd_stays_in_workspace(command_parts):
                return False
            continue
        if not _bash_paths_stay_in_workspace(command_parts):
            return False
        if not _is_read_only_program_call(command_parts):
            return False
    return True


def _shell_command_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _split_shell_commands(parts: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    current: list[str] = []
    for part in parts:
        if part in {"&&", "|"}:
            commands.append(current)
            current = []
        else:
            current.append(part)
    commands.append(current)
    return commands


def _is_read_only_program_call(parts: list[str]) -> bool:
    program = parts[0].rsplit("/", 1)[-1]
    if program in {"rg", "grep", "cat", "head", "tail", "ls", "pwd", "wc", "echo", "sort", "uniq"}:
        return True
    if program == "awk":
        script = " ".join(parts[1:]).lower()
        return "system(" not in script and ">" not in script
    if program == "sed":
        return not any(part == "-i" or part.startswith("-i") or part == "--in-place" for part in parts[1:])
    if program == "find":
        forbidden = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        return not any(part in forbidden for part in parts[1:])
    if program == "git":
        if len(parts) < 2:
            return False
        return parts[1] in {
            "branch",
            "describe",
            "diff",
            "grep",
            "log",
            "ls-files",
            "remote",
            "rev-parse",
            "show",
            "status",
        }
    return False


def _cd_stays_in_workspace(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    return _bash_paths_stay_in_workspace(parts)


def _bash_paths_stay_in_workspace(parts: list[str]) -> bool:
    cwd = Path.cwd().resolve()
    for token in parts[1:]:
        if not token or token.startswith("-"):
            continue
        if token in {".", "./"}:
            continue
        if ".." in Path(token).parts:
            return False
        path = Path(token).expanduser()
        if path.is_absolute():
            try:
                resolved = path.resolve()
            except OSError:
                return False
            if resolved != cwd and cwd not in resolved.parents:
                return False
    return True


async def _select_confirmation(
    question: str,
    *,
    allow_all_edits: bool = False,
    allow_read_only_bash: bool = False,
) -> str | None:
    choices = [
        ("allow", "Yes", "allow once"),
        ("deny", "No", "do not run this tool"),
    ]
    if allow_all_edits:
        choices.insert(1, ("allow_all_edits", "Yes, allow all edits during this session", "shift+tab"))
    if allow_read_only_bash:
        choices.insert(1, ("allow_read_only_bash", "Yes, allow read-only shell commands this session", "grep/sed/rg/cat/ls"))
    selected_index = 0
    key_bindings = KeyBindings()

    @key_bindings.add("up")
    def _move_up(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(choices)
        event.app.invalidate()

    @key_bindings.add("down")
    def _move_down(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(choices)
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
                selected_index = (selected_index - 1) % len(choices)
                get_app().invalidate()
            elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                selected_index = (selected_index + 1) % len(choices)
                get_app().invalidate()
            elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                selected_index = index
                get_app().exit(result=choices[selected_index][0])
            return None

        return _handle

    def _fragments():
        fragments = [
            ("", "\n"),
            ("", "  "),
            ("#c7d2fe", question),
            ("", "\n"),
            ("#6b7280", "  Esc to cancel\n\n"),
        ]
        number_width = len(str(len(choices)))
        for index, (value, label, detail) in enumerate(choices):
            mouse_handler = _mouse_handler(index)
            selected = index == selected_index
            arrow_style = f"{_SELECT_BG} #f0abfc bold" if selected else "#f0abfc bold"
            label_style = f"{_SELECT_BG} #7c83ff bold" if selected else "#c7d2fe"
            detail_style = f"{_SELECT_BG} #ffffff" if selected else "#e5e7eb"
            muted_style = f"{_SELECT_BG} #b6bfd8" if selected else "#8f99b7"
            fragments.append(("", " ", mouse_handler))
            fragments.append((arrow_style, "❯" if selected else " ", mouse_handler))
            fragments.append((muted_style, " ", mouse_handler))
            fragments.append((muted_style, f"{index + 1:>{number_width}}. ", mouse_handler))
            fragments.append((label_style, label, mouse_handler))
            if value in {"allow_all_edits", "allow_read_only_bash"}:
                fragments.append((muted_style, "  ", mouse_handler))
                fragments.append((detail_style, detail, mouse_handler))
            fragments.append(("", "\n"))
        return fragments

    control = FormattedTextControl(_fragments, focusable=True)
    application = Application(
        layout=Layout(HSplit([Window(content=control, always_hide_cursor=True)])),
        key_bindings=key_bindings,
        full_screen=False,
        mouse_support=True,
    )
    return await application.run_async()
