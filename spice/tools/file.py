"""Workspace file tools."""

from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path

from spice.sandbox.factory import create_workspace_policy
from spice.sandbox.policy import WorkspacePolicy
from spice.tools.base import FatalToolError, Tool, ToolContext, ToolResult, fatal_tool_error, tool_error, tool_result, truncate_head_tail


async def list_dir(args: dict, context: ToolContext) -> ToolResult:
    try:
        path = _workspace(context).resolve_read_dir(str(args.get("path") or "."))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    if not path.exists():
        return tool_error(f"Path does not exist: {path}")
    if not path.is_dir():
        return tool_error(f"Path is not a directory: {path}")
    rows = []
    dir_count = file_count = other_count = hidden_count = 0
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            hidden_count += 1
        if child.is_dir():
            dir_count += 1
        elif child.is_file():
            file_count += 1
        else:
            other_count += 1
        suffix = "/" if child.is_dir() else ""
        rows.append(f"{child.name}{suffix}")
    return tool_result(
        "\n".join(rows) or "(empty)",
        {
            "total_entries": len(rows),
            "dir_count": dir_count,
            "file_count": file_count,
            "other_count": other_count,
            "hidden_count": hidden_count,
        },
    )


async def read_file(args: dict, context: ToolContext) -> ToolResult:
    try:
        path = _workspace(context).resolve_read(str(args.get("path") or ""))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or 12000)
    if offset < 0:
        return tool_error("offset must be >= 0.")
    if limit < 1:
        return tool_error("limit must be >= 1.")
    if not path.exists():
        return tool_error(f"File does not exist: {path}. Use list_dir to inspect available files.")
    if not path.is_file():
        if path.is_dir():
            return tool_error(f"Path is a directory: {path}. Use list_dir first, then read a specific file.")
        return tool_error(f"Path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return tool_error(f"File is not valid UTF-8 text: {path}")
    total = len(text)
    content = text[offset : offset + limit]
    partial = offset > 0 or offset + limit < total
    if context.file_states:
        context.file_states.record_read(path, partial=partial)
    details = {
        "path": str(path),
        "line_count": _line_count(content),
        "char_count": len(content),
        "total_chars": total,
        "offset": offset,
        "limit": limit,
        "partial": partial,
    }
    if partial:
        header = f"[partial read: chars {offset}-{offset + len(content)} of {total}]\n"
        return tool_result(header + content, details)
    return tool_result(content, details)


async def write_file(args: dict, context: ToolContext) -> ToolResult:
    content = str(args.get("content") or "")
    try:
        path = _workspace(context).resolve_write(str(args.get("path") or ""), content_size=len(content.encode("utf-8")))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    if context.file_states:
        stale_error = context.file_states.check_before_overwrite(path)
        if stale_error:
            return _file_state_guard_error(stale_error, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if context.file_states:
        context.file_states.note_write(path)
    return tool_result(
        f"Wrote {len(content)} characters to {path}",
        {"path": str(path), "line_count": _line_count(content), "char_count": len(content)},
    )


async def edit_file(args: dict, context: ToolContext) -> ToolResult:
    old = str(args.get("old_text") or "")
    new = str(args.get("new_text") or "")
    raw_path = str(args.get("path") or "")
    occurrence = args.get("occurrence")
    dry_run = bool(args.get("dry_run") or False)
    if not old:
        return tool_error("old_text is required.")
    workspace = _workspace(context)
    try:
        path = workspace.resolve_write(raw_path, content_size=0)
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    if not path.exists():
        return tool_error(f"File does not exist: {path}")
    if context.file_states:
        stale_error = context.file_states.check_before_edit(path)
        if stale_error:
            return _file_state_guard_error(stale_error, path)
    content = path.read_text(encoding="utf-8")
    count = content.count(old)
    if count == 0:
        return tool_error(_old_text_not_found_message(content, old))
    if occurrence is None and count != 1:
        return tool_error(f"old_text matched {count} times. Provide occurrence to choose which match to replace.")
    if occurrence is not None:
        try:
            occurrence_index = int(occurrence)
        except (TypeError, ValueError):
            return tool_error("occurrence must be an integer.")
        if occurrence_index < 1 or occurrence_index > count:
            return tool_error(f"occurrence must be between 1 and {count}.")
        updated = _replace_nth(content, old, new, occurrence_index)
    else:
        updated = content.replace(old, new, 1)
    try:
        workspace.resolve_write(raw_path, content_size=len(updated.encode("utf-8")))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    diff = "".join(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    details = {"path": str(path), "dry_run": dry_run, **_diff_stats(diff)}
    preview = truncate_head_tail(diff, 8000)
    if dry_run:
        return tool_result(preview or "No changes.", details, full_content=diff if preview != diff else None)
    path.write_text(updated, encoding="utf-8")
    if context.file_states:
        context.file_states.note_write(path)
    return tool_result(preview or f"Edited {path}", details, full_content=diff if preview != diff else None)


async def apply_patch(args: dict, context: ToolContext) -> ToolResult:
    operations = args.get("operations") or []
    dry_run = bool(args.get("dry_run") or False)
    if not isinstance(operations, list) or not operations:
        return tool_error("operations must be a non-empty array.")

    before_by_path: dict[Path, str | None] = {}
    after_by_path: dict[Path, str | None] = {}
    errors: list[str] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            errors.append(f"operations[{index}] must be an object.")
            continue
        plan, error = _plan_patch_operation(operation, context, after_by_path)
        if error:
            errors.append(f"operations[{index}]: {error}")
            continue
        if plan:
            path, before, after = plan
            before_by_path.setdefault(path, before)
            after_by_path[path] = after
    if errors:
        return tool_error("Patch validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    workspace = _workspace(context)
    for path, after in after_by_path.items():
        if after is None:
            continue
        try:
            workspace.resolve_write(str(path), content_size=len(after.encode("utf-8")))
        except PermissionError as exc:
            return fatal_tool_error(str(exc), code="workspace_policy_denied")

    diff = _multi_file_diff(before_by_path, after_by_path)
    details = {
        "dry_run": dry_run,
        "files_changed": len(after_by_path),
        "operations": len(operations),
        **_diff_stats(diff),
    }
    preview = truncate_head_tail(diff, 12000)
    if dry_run:
        return tool_result(preview or "No changes.", details, full_content=diff if preview != diff else None)

    error = _commit_patch(before_by_path, after_by_path)
    if error is not None:
        return tool_error(error, code="patch_commit_failed")
    if context.file_states:
        for path in after_by_path:
            context.file_states.note_write(path)
    return tool_result(
        preview or f"Applied patch to {len(after_by_path)} file(s).",
        details,
        full_content=diff if preview != diff else None,
    )


def _commit_patch(before_by_path: dict[Path, str | None], after_by_path: dict[Path, str | None]) -> str | None:
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path, after in after_by_path.items():
            if after is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
            temp_path = Path(temp_name)
            staged[path] = temp_path
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(after)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, path.stat().st_mode if path.exists() else 0o644)

        for path in sorted(after_by_path, key=lambda item: str(item)):
            after = after_by_path[path]
            if after is None:
                path.unlink()
            else:
                os.replace(staged.pop(path), path)
            committed.append(path)
    except OSError as exc:
        rollback_errors = _rollback_patch(committed, before_by_path)
        suffix = f" Rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else " Changes were rolled back."
        return f"Patch commit failed after updating {len(committed)} file(s): {exc}.{suffix}"
    finally:
        for temp_path in staged.values():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return None


def _rollback_patch(committed: list[Path], before_by_path: dict[Path, str | None]) -> list[str]:
    errors: list[str] = []
    for path in reversed(committed):
        before = before_by_path.get(path)
        try:
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".rollback", dir=str(path.parent))
                temp_path = Path(temp_name)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(before)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp_path, path)
                finally:
                    temp_path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def _plan_patch_operation(
    operation: dict,
    context: ToolContext,
    after_by_path: dict[Path, str | None],
) -> tuple[tuple[Path, str | None, str | None] | None, str | None]:
    operation_type = str(operation.get("type") or "")
    path_arg = str(operation.get("path") or "")
    if not path_arg:
        return None, "path is required."
    try:
        path = _workspace(context).resolve_write(path_arg, content_size=0)
    except PermissionError as exc:
        raise FatalToolError(str(exc), code="workspace_policy_denied") from exc
    current_loaded = path in after_by_path
    try:
        current = after_by_path[path] if current_loaded else _read_existing_patch_content(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, str(exc)

    if operation_type == "add":
        if current is not None:
            return None, f"cannot add existing file: {path}"
        return (path, None, str(operation.get("content") or "")), None

    if operation_type == "delete":
        if current is None:
            return None, f"file does not exist: {path}"
        if not current_loaded and context.file_states:
            stale_error = context.file_states.check_before_edit(path)
            if stale_error:
                return None, stale_error
        return (path, current, None), None

    if operation_type == "replace":
        old, old_error = _patch_text_value(operation, "old_text", "old_str")
        if old_error:
            return None, old_error
        new, new_error = _patch_text_value(operation, "new_text", "new_str")
        if new_error:
            return None, new_error
        if not old:
            return None, "old_text is required for replace."
        if current is None:
            return None, f"file does not exist: {path}"
        if not current_loaded and context.file_states:
            stale_error = context.file_states.check_before_edit(path)
            if stale_error:
                return None, stale_error
        count = current.count(old)
        if count == 0:
            return None, _old_text_not_found_message(current, old)
        if count != 1:
            return None, f"old_text matched {count} times; make it unique for apply_patch replace."
        return (path, current, current.replace(old, new, 1)), None

    return None, "type must be one of: add, replace, delete."


def _file_state_guard_error(message: str, path: Path) -> ToolResult:
    return tool_error(
        message,
        {
            "path": str(path),
            "category": "file_state_guard",
            "presentation": "guidance",
        },
    )


def _patch_text_value(operation: dict, canonical: str, alias: str) -> tuple[str, str | None]:
    has_canonical = canonical in operation
    has_alias = alias in operation
    if has_canonical and has_alias and operation[canonical] != operation[alias]:
        return "", f"{canonical} and {alias} both provided with different values."
    value = operation.get(canonical) if has_canonical else operation.get(alias)
    return str(value or ""), None


def _read_existing_patch_content(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"path is not a file: {path}")
    return path.read_text(encoding="utf-8")


def _multi_file_diff(before_by_path: dict[Path, str | None], after_by_path: dict[Path, str | None]) -> str:
    chunks: list[str] = []
    for path in sorted(after_by_path, key=lambda item: str(item)):
        before = before_by_path.get(path)
        after = after_by_path[path]
        before_lines = [] if before is None else before.splitlines(keepends=True)
        after_lines = [] if after is None else after.splitlines(keepends=True)
        fromfile = "/dev/null" if before is None else str(path)
        tofile = "/dev/null" if after is None else str(path)
        chunks.append("".join(difflib.unified_diff(before_lines, after_lines, fromfile=fromfile, tofile=tofile)))
    return "".join(chunks)


def _diff_stats(diff: str) -> dict[str, int]:
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"lines_added": added, "lines_removed": removed}


def _replace_nth(content: str, old: str, new: str, occurrence: int) -> str:
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = content.find(old, search_from)
        search_from = start + len(old)
    if start < 0:
        return content
    return content[:start] + new + content[start + len(old) :]


def _old_text_not_found_message(content: str, old: str) -> str:
    old_lines = old.splitlines()
    lines = content.splitlines()
    message = ["old_text was not found."]

    if old.strip() and old.strip() in content:
        message.append("A whitespace-sensitive variant was found; check leading/trailing spaces or blank lines.")
    if old.replace("\r\n", "\n") != old and old.replace("\r\n", "\n") in content:
        message.append("A line-ending-normalized variant was found; use LF line endings in old_text.")

    if old_lines:
        best = _nearest_window(lines, old_lines)
        if best:
            start_line, snippet = best
            message.append(f"Closest nearby content starts at line {start_line}:")
            message.append(snippet)
    return "\n".join(message)


def _nearest_window(lines: list[str], old_lines: list[str]) -> tuple[int, str] | None:
    window_size = max(len(old_lines), 1)
    needle = "\n".join(line.strip() for line in old_lines if line.strip())
    if not needle:
        return None
    best_ratio = 0.0
    best_index = -1
    for index in range(0, max(len(lines) - window_size + 1, 1)):
        window = lines[index : index + window_size]
        candidate = "\n".join(line.strip() for line in window if line.strip())
        ratio = difflib.SequenceMatcher(None, needle, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = index
    if best_index < 0 or best_ratio < 0.45:
        return None
    context_start = max(best_index - 2, 0)
    context_end = min(best_index + window_size + 2, len(lines))
    snippet_lines = [f"{line_no}: {lines[line_no - 1]}" for line_no in range(context_start + 1, context_end + 1)]
    return best_index + 1, "\n".join(snippet_lines)


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode", "dist", "build"}
MAX_SEARCH_MATCHES = 200
MAX_MULTI_READ_FILES = 20
MAX_MULTI_READ_TOTAL_CHARS = 60000


async def read_files(args: dict, context: ToolContext) -> ToolResult:
    files = args.get("files") or []
    if not isinstance(files, list) or not files:
        return tool_error("files must be a non-empty array.")
    if len(files) > MAX_MULTI_READ_FILES:
        return tool_error(f"files must contain at most {MAX_MULTI_READ_FILES} items.")

    chunks: list[str] = []
    total_chars = 0
    files_read = 0
    partial_count = 0
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            return tool_error(f"files[{index}] must be an object.")
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            return tool_error(f"files[{index}].path is required.")
        try:
            offset = int(item.get("offset") or 0)
            limit = int(item.get("limit") or 12000)
        except (TypeError, ValueError):
            return tool_error(f"files[{index}].offset and files[{index}].limit must be integers.")
        if offset < 0:
            return tool_error(f"files[{index}].offset must be >= 0.")
        if limit < 1:
            return tool_error(f"files[{index}].limit must be >= 1.")

        try:
            path = _workspace(context).resolve_read(raw_path)
        except PermissionError as exc:
            return fatal_tool_error(str(exc), code="workspace_policy_denied")
        if not path.exists():
            return tool_error(f"File does not exist: {path}")
        if not path.is_file():
            return tool_error(f"Path is not a file: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return tool_error(f"File is not valid UTF-8 text: {path}")

        content = text[offset : offset + limit]
        total = len(text)
        partial = offset > 0 or offset + limit < total
        if total_chars + len(content) > MAX_MULTI_READ_TOTAL_CHARS:
            remaining = max(0, MAX_MULTI_READ_TOTAL_CHARS - total_chars)
            content = content[:remaining]
            partial = True
        if context.file_states:
            context.file_states.record_read(path, partial=partial)
        if partial:
            partial_count += 1
        label = raw_path
        range_label = f" chars {offset}-{offset + len(content)} of {total}" if partial else f" {total} chars"
        chunks.append(f"--- {label} ({range_label}) ---\n{content}")
        total_chars += len(content)
        files_read += 1
        if total_chars >= MAX_MULTI_READ_TOTAL_CHARS:
            chunks.append(f"[multi-file read truncated at {MAX_MULTI_READ_TOTAL_CHARS} characters]")
            break
    content = "\n\n".join(chunks)
    return tool_result(
        content,
        {
            "file_count": files_read,
            "requested_files": len(files),
            "line_count": _line_count(content),
            "char_count": total_chars,
            "partial_count": partial_count,
            "truncated": total_chars >= MAX_MULTI_READ_TOTAL_CHARS,
        },
    )


async def search_files(args: dict, context: ToolContext) -> ToolResult:
    pattern = str(args.get("pattern") or "")
    try:
        path = _workspace(context).resolve_read_dir(str(args.get("path") or "."))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    if not pattern:
        return tool_error("pattern is required.")
    matches: list[str] = []
    matched_files: set[Path] = set()
    truncated = False
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            try:
                _workspace(context).resolve_read(str(file_path))
            except PermissionError:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    rel = _workspace(context).relative(file_path)
                    matches.append(f"{rel}:{lineno}: {line}")
                    matched_files.add(file_path)
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        truncated = True
                        full_content = "\n".join(matches)
                        content = truncate_head_tail(full_content)
                        return tool_result(
                            content,
                            {"match_count": len(matches), "file_count": len(matched_files), "truncated": truncated},
                            full_content=full_content if content != full_content else None,
                        )
    return tool_result(
        "\n".join(matches) if matches else "No matches.",
        {"match_count": len(matches), "file_count": len(matched_files), "truncated": truncated},
    )


def _workspace(context: ToolContext) -> WorkspacePolicy:
    return context.workspace or create_workspace_policy(None, cwd=context.cwd)


def create_file_tools() -> list[Tool]:
    path_schema = {"type": "string", "description": "Path relative to the current workspace."}
    patch_operation_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["add", "replace", "delete"]},
            "path": path_schema,
            "content": {"type": "string", "description": "Full file content for add operations."},
            "old_text": {"type": "string", "description": "Exact text to replace for replace operations."},
            "new_text": {"type": "string", "description": "Replacement text for replace operations."},
            "old_str": {"type": "string", "description": "Alias for old_text."},
            "new_str": {"type": "string", "description": "Alias for new_text."},
        },
        "required": ["type", "path"],
        "additionalProperties": False,
    }
    return [
        Tool(
            "list_dir",
            "List directory contents in the workspace, sorted alphabetically, with '/' suffix for directories. Use this instead of bash ls when you only need directory entries.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the current workspace. Defaults to '.'.",
                    }
                },
            },
            list_dir,
            concurrency="parallel",
        ),
        Tool(
            "read_file",
            "Read a UTF-8 text file from the workspace. Use this instead of bash cat/head/tail/sed for reading files. Use offset/limit for large files, and use search_files first when you only need to locate text.",
            {
                "type": "object",
                "properties": {
                    "path": path_schema,
                    "offset": {"type": "integer", "minimum": 0, "description": "Character offset to start reading from."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum characters to read. Defaults to 12000.",
                    },
                },
                "required": ["path"],
            },
            read_file,
            concurrency="parallel",
        ),
        Tool(
            "read_files",
            "Read multiple UTF-8 text files or file ranges from the workspace in one tool call. Use this for multi-file analysis instead of many repeated read_file calls. Use search_files first when you only need to locate text.",
            {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_MULTI_READ_FILES,
                        "description": "Files or ranges to read, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": path_schema,
                                "offset": {"type": "integer", "minimum": 0, "description": "Character offset to start reading from."},
                                "limit": {"type": "integer", "minimum": 1, "description": "Maximum characters to read from this file. Defaults to 12000."},
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["files"],
            },
            read_files,
            concurrency="parallel",
        ),
        Tool(
            "write_file",
            "Write UTF-8 content to a workspace file, completely replacing existing content. Use only when the user explicitly asks to save/create/overwrite a file or when implementing requested code changes. Before overwriting an existing file, first read it with read_file/read_files; otherwise this tool will be rejected. Do not use for pure examples, explanations, Markdown, or tables that should be printed in the response. Creates parent directories automatically; use edit_file for targeted changes.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the current workspace. The file will be created if missing and overwritten if it exists.",
                    },
                    "content": {"type": "string", "description": "Complete file content to write."},
                },
                "required": ["path", "content"],
            },
            write_file,
            requires_confirmation=True,
        ),
        Tool(
            "edit_file",
            "Edit a workspace text file by exact old_text replacement. Read the target file first with read_file/read_files so old_text is based on current content; otherwise this tool will be rejected. Use for precise targeted edits to existing files; use write_file only for new files or complete rewrites. old_text must match exactly, including whitespace, and should be as small as possible while still unique.",
            {
                "type": "object",
                "properties": {
                    "path": path_schema,
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace. Must uniquely identify the intended change unless occurrence is provided. Include enough surrounding context for uniqueness, but avoid large unchanged regions.",
                    },
                    "new_text": {"type": "string", "description": "Replacement text."},
                    "occurrence": {"type": "integer", "description": "1-based match index when old_text appears multiple times."},
                    "dry_run": {"type": "boolean", "description": "Return the diff without writing the file."},
                },
                "required": ["path", "old_text", "new_text"],
            },
            edit_file,
            requires_confirmation=True,
        ),
        Tool(
            "apply_patch",
            "Apply a validated multi-file patch using structured operations. Use for coordinated multi-file edits, adding files, or deleting files. Before replace/delete operations on existing files, read the target files first with read_file/read_files; otherwise those operations will be rejected. All operations are validated before any file is written; dry_run returns the combined diff without writing.",
            {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "items": patch_operation_schema,
                        "description": "Patch operations applied in order after validation.",
                    },
                    "dry_run": {"type": "boolean", "description": "Return the combined diff without writing files."},
                },
                "required": ["operations"],
            },
            apply_patch,
            requires_confirmation=True,
        ),
        Tool(
            "search_files",
            "Search UTF-8 text files in the workspace for a literal string. Use this instead of bash grep/rg/find when looking for text in files. Returns matching file paths and line numbers.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Literal text to search for."},
                    "path": path_schema,
                },
                "required": ["pattern"],
            },
            search_files,
            concurrency="parallel",
        ),
    ]
