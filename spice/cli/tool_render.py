"""Pure tool summaries, previews, and terminal confirmation rendering."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType

from spice.interactive.tool_presentation import tool_icon
from spice.interactive.types import ConfirmChoice
from spice.tools.base import ToolResult

_PREVIEW_FIRST_TOOLS = {"read_file", "search_files"}
_DEFAULT_PREVIEW_CHARS = 1600
_DEFAULT_PREVIEW_LINES = 30
_COMPACT_PREVIEW_CHARS = 900
_COMPACT_PREVIEW_LINES = 12
_SELECT_BG = "bg:#3f4a68"
_TOOL_ARG_LIMIT = 80


@dataclass(frozen=True)
class _ToolSpec:
    start: Callable[[dict], str]
    end: Callable[[dict, ToolResult], str]


def format_tool_start(tool_name: str, args: dict) -> str:
    """Single-line description shown when a tool starts running."""
    spec = _TOOL_SPECS.get(tool_name)
    if spec is not None:
        try:
            text = spec.start(args or {})
            if text.startswith("❯ "):
                text = text[2:]
            return f"{tool_icon(tool_name)} {text}"
        except Exception:  # noqa: BLE001 - never break rendering
            pass
    target = _summarize_args(args or {})
    return f"{tool_icon(tool_name)} {tool_name}{(' ' + target) if target else ''}"


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
    if value is None:
        return default
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
        first_line = next(
            (line for line in (result.content or "").splitlines() if line.strip()), ""
        )
        return (
            f"! {_humanize_file_state_guard(first_line)}"
            if first_line
            else "! Action needed"
        )
    text = (result.content or "").strip()
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if first_line.startswith("read_file must be called first before "):
        return f"! {_humanize_file_state_guard(first_line)}"
    if first_line.startswith("Full read_file must be called before "):
        return f"! {_humanize_file_state_guard(first_line)}"
    return None


def _humanize_file_state_guard(message: str) -> str:
    if message.startswith("read_file must be called first before overwriting "):
        path = message.removeprefix(
            "read_file must be called first before overwriting "
        ).rstrip(".")
        return f"Read file first before overwriting {path}"
    if message.startswith("read_file must be called first before editing "):
        path = message.removeprefix(
            "read_file must be called first before editing "
        ).rstrip(".")
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
    lines = _detail_int(
        result, "line_count", _line_count(str(args.get("content") or ""))
    )
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
    prefix = (
        "✓ Previewed patch"
        if args.get("dry_run") or _detail_bool(result, "dry_run")
        else "✓ Applied patch"
    )
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
    items = sum(
        1 for line in (result.content or "").splitlines() if line.startswith("- ")
    )
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
        content = _truncate_inline(
            str(item.get("content") or "(no description)"), limit=100
        )
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
    active = next(
        (
            item
            for item in items
            if str(item.get("status") or "pending") == "in_progress"
        ),
        None,
    )
    if active is None:
        active = next(
            (
                item
                for item in items
                if str(item.get("status") or "pending") == "pending"
            ),
            None,
        )
    if active is None:
        suffix = "completed" if not cancelled else f"completed · {cancelled} cancelled"
    else:
        suffix = _truncate_inline(
            str(active.get("content") or "(no description)"), limit=100
        )
    parts = [f"● todo {completed}/{total}"]
    if in_progress:
        parts.append(f"{in_progress} running")
    if pending:
        parts.append(f"{pending} pending")
    return " · ".join(parts + [suffix])


def _todo_display_lines(
    result: ToolResult, *, include_summary: bool = False
) -> list[str]:
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
        content = _truncate_inline(
            str(item.get("content") or "(no description)"), limit=100
        )
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
    if isinstance(result.details, dict) and isinstance(
        result.details.get("todos"), list
    ):
        return result.details
    try:
        parsed = json.loads(result.content or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _todo_summary(todos: list) -> dict[str, int]:
    counts = {
        "total": 0,
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "cancelled": 0,
    }
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


def _preview_tool_result(
    tool_name: str, content: str, *, is_error: bool
) -> tuple[str, bool]:
    if not is_error and tool_name in _PREVIEW_FIRST_TOOLS:
        return _preview_text(
            content, max_chars=_COMPACT_PREVIEW_CHARS, max_lines=_COMPACT_PREVIEW_LINES
        )
    return _preview_text(
        content, max_chars=_DEFAULT_PREVIEW_CHARS, max_lines=_DEFAULT_PREVIEW_LINES
    )


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


# Compatibility wrappers — canonical rules live in spice.interactive.confirm.
def _is_file_edit_tool(tool_name: str) -> bool:
    from spice.interactive.confirm import is_file_edit_tool

    return is_file_edit_tool(tool_name)


def _confirmation_question(tool_name: str, args: dict) -> str:
    from spice.interactive.confirm import confirmation_question

    return confirmation_question(tool_name, args)


def _is_read_only_bash_command(args: dict) -> bool:
    from spice.interactive.confirm import is_read_only_bash_command

    return is_read_only_bash_command(args)


async def _select_confirmation(
    question: str,
    *,
    choices: tuple[ConfirmChoice, ...] | None = None,
    allow_all_edits: bool = False,
    allow_read_only_bash: bool = False,
) -> str | None:
    if choices is None:
        legacy_choices = [
            ("allow", "Yes", "allow once"),
            ("deny", "No", "do not run this tool"),
        ]
        if allow_all_edits:
            legacy_choices.insert(
                1,
                (
                    "allow_all_edits",
                    "Yes, allow all edits during this session",
                    "shift+tab",
                ),
            )
        if allow_read_only_bash:
            legacy_choices.insert(
                1,
                (
                    "allow_read_only_bash",
                    "Yes, allow read-only shell commands this session",
                    "grep/sed/rg/cat/ls",
                ),
            )
    else:
        legacy_choices = [
            (choice.id, choice.label, choice.detail) for choice in choices
        ]
    selected_index = 0
    key_bindings = KeyBindings()

    @key_bindings.add("up")
    def _move_up(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(legacy_choices)
        event.app.invalidate()

    @key_bindings.add("down")
    def _move_down(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(legacy_choices)
        event.app.invalidate()

    @key_bindings.add("enter")
    def _confirm(event) -> None:
        event.app.exit(result=legacy_choices[selected_index][0])

    @key_bindings.add("escape")
    @key_bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    def _mouse_handler(index: int):
        def _handle(mouse_event):
            nonlocal selected_index
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                selected_index = (selected_index - 1) % len(legacy_choices)
                get_app().invalidate()
            elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                selected_index = (selected_index + 1) % len(legacy_choices)
                get_app().invalidate()
            elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                selected_index = index
                get_app().exit(result=legacy_choices[selected_index][0])
            return None

        return _handle

    def _fragments():
        fragments: StyleAndTextTuples = [
            ("", "\n"),
            ("", "  "),
            ("#c7d2fe", question),
            ("", "\n"),
            ("#6b7280", "  Esc to cancel\n\n"),
        ]
        number_width = len(str(len(legacy_choices)))
        for index, (value, label, detail) in enumerate(legacy_choices):
            mouse_handler = _mouse_handler(index)
            selected = index == selected_index
            arrow_style = f"{_SELECT_BG} #f0abfc bold" if selected else "#f0abfc bold"
            label_style = f"{_SELECT_BG} #7c83ff bold" if selected else "#c7d2fe"
            detail_style = f"{_SELECT_BG} #ffffff" if selected else "#e5e7eb"
            muted_style = f"{_SELECT_BG} #b6bfd8" if selected else "#8f99b7"
            fragments.append(("", " ", mouse_handler))
            fragments.append((arrow_style, "❯" if selected else " ", mouse_handler))
            fragments.append((muted_style, " ", mouse_handler))
            fragments.append(
                (muted_style, f"{index + 1:>{number_width}}. ", mouse_handler)
            )
            fragments.append((label_style, label, mouse_handler))
            if detail:
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
