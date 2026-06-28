"""Explicit built-in tool registry.

Tools receive cwd at execution time via ToolContext, so creation takes no arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spice.llm.types import ToolSchema
from spice.tools.base import Tool
from spice.tools.base import normalize_tool_arguments
from spice.tools.bash import create_bash_tools
from spice.tools.core import create_core_tools
from spice.tools.file import create_file_tools
from spice.tools.memory import create_memory_tools
from spice.tools.skill import create_skill_tools
from spice.tools.subagent import create_subagent_tool
from spice.tools.web import create_web_tools

TOOLSETS = {
    "core": ["get_current_time"],
    "file": ["list_dir", "read_file", "read_files", "write_file", "edit_file", "apply_patch", "search_files"],
    "bash": ["bash"],
    "web": ["web_search"],
    "skill": ["skills_list", "skill_view"],
    "memory": ["memory"],
    "subagent": ["spawn_subagents"],
}

READ_ONLY_TOOLS = ["get_current_time", "list_dir", "read_file", "read_files", "search_files", "web_search", "skills_list", "skill_view"]


@dataclass(frozen=True)
class ToolCallPlan:
    tool: Tool
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallError:
    message: str
    errors: list[str]


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        self._order: list[str] = []
        duplicates: list[str] = []
        for tool in tools:
            if tool.name in self._tools:
                duplicates.append(tool.name)
                continue
            self._tools[tool.name] = tool
            self._order.append(tool.name)
        if duplicates:
            names = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate tool names: {names}")

    @property
    def tools(self) -> list[Tool]:
        return [self._tools[name] for name in self._order]

    def schemas(self) -> list[ToolSchema]:
        return [ToolSchema(tool.name, tool.description, tool.parameters) for tool in self.tools]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def prepare_call(self, name: str, arguments: dict[str, Any]) -> ToolCallPlan | ToolCallError:
        tool = self.get(name)
        if tool is None:
            return ToolCallError(
                message=f"Tool not found: {name}",
                errors=[self._unknown_tool_message(name)],
            )
        normalized, errors = normalize_tool_arguments(tool.parameters, arguments)
        if errors:
            return ToolCallError(
                message="Invalid tool arguments: " + "; ".join(errors),
                errors=errors,
            )
        return ToolCallPlan(tool=tool, arguments=normalized)

    def _unknown_tool_message(self, name: str) -> str:
        if not self._order:
            return "No tools are registered."
        nearby = _nearest_names(name, self._order)
        if nearby:
            return f"Available tools include: {', '.join(nearby)}"
        return f"Available tools: {', '.join(self._order)}"


def create_all_tools(*, memory_enabled: bool = False, subagents_enabled: bool = False) -> dict[str, Tool]:
    tools = [
        *create_core_tools(),
        *create_file_tools(),
        *create_bash_tools(),
        *create_web_tools(),
        *create_skill_tools(),
    ]
    if memory_enabled:
        tools.extend(create_memory_tools())
    if subagents_enabled:
        tools.append(create_subagent_tool())
    return {tool.name: tool for tool in tools}


def create_read_only_tools() -> list[Tool]:
    registry = create_all_tools()
    return [registry[name] for name in READ_ONLY_TOOLS if name in registry]


def create_coding_tools(*, memory_enabled: bool = False, subagents_enabled: bool = False) -> list[Tool]:
    registry = create_all_tools(memory_enabled=memory_enabled, subagents_enabled=subagents_enabled)
    excluded = set()
    if not memory_enabled:
        excluded.add("memory")
    if not subagents_enabled:
        excluded.add("subagent")
    toolsets = {key: value for key, value in TOOLSETS.items() if key not in excluded}
    names = [name for names in toolsets.values() for name in names]
    return [registry[name] for name in names if name in registry]


def _nearest_names(name: str, choices: list[str]) -> list[str]:
    scored = sorted((_similarity(name, choice), choice) for choice in choices)
    nearby = [choice for score, choice in reversed(scored) if score >= 0.45]
    return nearby[:5]


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_parts = set(left.lower().replace("-", "_").split("_"))
    right_parts = set(right.lower().replace("-", "_").split("_"))
    overlap = len(left_parts & right_parts)
    return overlap / max(len(left_parts | right_parts), 1)
