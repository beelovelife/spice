"""System prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spice.llm.config import CONFIG_DIR
from spice.skills.loader import PROJECT_SKILLS_DIR, USER_SKILLS_DIR, load_skills
from spice.agent.tool_contract import TOOL_CONTRACT
from spice.tools.base import Tool

SPICE_CONTEXT_FILENAME = "SPICE.md"


@dataclass(frozen=True)
class _CacheEntry:
    signature: tuple[Any, ...]
    value: Any


_spice_context_cache: dict[str, _CacheEntry] = {}
_skills_cache: dict[str, _CacheEntry] = {}


def build_system_prompt(
    cwd: Path,
    tools: list[Tool],
    *,
    runtime_model: str | None = None,
    memory_context: str | None = None,
) -> str:
    has_update_todo = any(tool.name == "update_todo" for tool in tools)
    parts = [
        "You are Spice, a general-purpose CLI agent.",
        f"The current workspace is: {cwd}",
        f"Current runtime model: {runtime_model}" if runtime_model else "Current runtime model: unknown",
        "Available tools:",
        _format_tools(tools),
        TOOL_CONTRACT,
        "Use tools when they are needed to answer or act accurately. If no tool is needed, answer directly.",
        "Choose tools according to their descriptions and parameter descriptions.",
        "Keep responses concise and practical.",
    ]
    if has_update_todo:
        parts.insert(
            -1,
            "For complex tasks, maintain progress with update_todo. Use it when work spans multiple files or modules, needs substantial tool exploration, or follows an approved plan. Examples include repo analysis, feature implementation, debugging across files, and multi-step recovery. For these tasks, call update_todo before other exploratory work, keep exactly one item in_progress, and update items as work completes. Follow the detailed update_todo tool description for merge behavior, stable ids, and when to replace old task lists.",
        )
    if memory_context and memory_context.strip():
        parts.append(memory_context.strip())
    spice_context = _cached_spice_context(cwd)
    if spice_context:
        parts.append("SPICE.md instructions:")
        parts.append(spice_context)
    skill_result = _cached_load_skills(cwd)
    if skill_result.skills:
        parts.append("Available skills:")
        parts.extend(_format_skill_index(skill) for skill in skill_result.skills)
        parts.append("Skill descriptions are an index only; use skill_view to load full instructions when a skill seems relevant.")
    return "\n".join(parts)


def _format_tools(tools: list[Tool]) -> str:
    if not tools:
        return "(none)"
    return "\n".join(f"- {tool.name}: {_first_sentence(tool.description)}" for tool in tools)


def load_spice_context(cwd: Path) -> str | None:
    sections: list[tuple[str, str]] = []
    global_path = CONFIG_DIR / SPICE_CONTEXT_FILENAME
    project_path = find_project_spice_file(cwd)

    global_content = _read_non_empty_text(global_path)
    if global_content:
        sections.append((f"Global {SPICE_CONTEXT_FILENAME} ({global_path})", global_content))

    if project_path is not None and project_path.resolve() != global_path.resolve():
        project_content = _read_non_empty_text(project_path)
        if project_content:
            sections.append((f"Project {SPICE_CONTEXT_FILENAME} ({project_path})", project_content))

    if not sections:
        return None
    return "\n\n".join(f"{title}:\n{content}" for title, content in sections)


def _cached_spice_context(cwd: Path) -> str | None:
    cache_key = _cwd_cache_key(cwd)
    global_path = CONFIG_DIR / SPICE_CONTEXT_FILENAME
    project_path = find_project_spice_file(cwd)
    signature = (
        _path_signature(global_path),
        _path_signature(project_path) if project_path is not None else None,
    )
    cached = _spice_context_cache.get(cache_key)
    if cached and cached.signature == signature:
        return cached.value
    result = load_spice_context(cwd)
    _spice_context_cache[cache_key] = _CacheEntry(signature, result)
    return result


def _cached_load_skills(cwd: Path):
    cache_key = _cwd_cache_key(cwd)
    signature = _skills_signature(cwd)
    cached = _skills_cache.get(cache_key)
    if cached and cached.signature == signature:
        return cached.value
    result = load_skills(cwd=cwd)
    _skills_cache[cache_key] = _CacheEntry(signature, result)
    return result


def find_project_spice_file(cwd: Path) -> Path | None:
    current = cwd.resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        candidate = path / SPICE_CONTEXT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _cwd_cache_key(cwd: Path) -> str:
    try:
        return str(cwd.resolve())
    except OSError:
        return str(cwd)


def _path_signature(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _skills_signature(cwd: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [*_skill_files_in(USER_SKILLS_DIR)]
    project_root = _project_skills_dir(cwd)
    if project_root is not None:
        paths.extend(_skill_files_in(project_root))
    return tuple(_path_signature(path) for path in paths)


def _skill_files_in(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(path for path in root.glob("*/SKILL.md") if path.is_file())


def _project_skills_dir(cwd: Path) -> Path | None:
    current = cwd.resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        root = path / PROJECT_SKILLS_DIR
        if root.exists():
            return root
    return None


def _read_non_empty_text(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stripped = text.strip()
    return stripped or None


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return ""
    for marker in (". ", "? ", "! "):
        index = stripped.find(marker)
        if index != -1:
            return stripped[: index + 1]
    return stripped


def _format_skill_index(skill) -> str:
    extras: list[str] = []
    if skill.triggers:
        extras.append("triggers: " + ", ".join(skill.triggers))
    if skill.always:
        extras.append("always")
    suffix = f" ({'; '.join(extras)})" if extras else ""
    return f"- {skill.name} [{skill.source}]: {skill.description}{suffix}"
