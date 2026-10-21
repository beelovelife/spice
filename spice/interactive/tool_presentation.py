"""Shared, UI-neutral presentation metadata for tool activity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPresentation:
    """A compact identity that every frontend can render consistently."""

    icon: str
    label: str
    category: str


_TOOL_PRESENTATIONS: dict[str, ToolPresentation] = {
    "read_file": ToolPresentation("◇", "Read file", "file_read"),
    "read_files": ToolPresentation("◇", "Read files", "file_read"),
    "list_dir": ToolPresentation("▦", "List directory", "file_read"),
    "search_files": ToolPresentation("⌕", "Search files", "search"),
    "write_file": ToolPresentation("◆", "Write file", "file_write"),
    "edit_file": ToolPresentation("◆", "Edit file", "file_edit"),
    "apply_patch": ToolPresentation("◆", "Apply patch", "file_edit"),
    "bash": ToolPresentation("⌘", "Run command", "shell"),
    "web_search": ToolPresentation("◎", "Search web", "web"),
    "get_current_time": ToolPresentation("◷", "Get time", "utility"),
    "skills_list": ToolPresentation("✦", "List skills", "skill"),
    "skill_view": ToolPresentation("✦", "View skill", "skill"),
    "update_todo": ToolPresentation("☑", "Update tasks", "todo"),
    "memory": ToolPresentation("◈", "Update memory", "memory"),
    "spawn_subagents": ToolPresentation("⇶", "Run subagents", "subagent"),
    "complete_long_task": ToolPresentation("✓", "Complete task", "task"),
}


def tool_presentation(tool_name: str) -> ToolPresentation:
    """Return stable metadata with a safe generic fallback for MCP/extensions."""

    known = _TOOL_PRESENTATIONS.get(tool_name)
    if known is not None:
        return known
    label = tool_name.replace("_", " ").strip().title() or "Tool"
    return ToolPresentation("·", label, "generic")


def tool_icon(tool_name: str) -> str:
    return tool_presentation(tool_name).icon

