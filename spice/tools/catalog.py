"""Shared built-in tool catalog for CLI and interactive frontends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from spice.llm.config import SpiceConfig, get_api_key
from spice.tools.browser import browser_dependency_available
from spice.tools.tool_registry import READ_ONLY_TOOLS, TOOLSETS, create_all_tools

ToolStatus = Literal["enabled", "disabled", "unavailable"]


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    toolset: str
    status: ToolStatus
    risk: str
    confirmation: str
    execution: str
    timeout: str
    description: str
    parameters: dict[str, Any]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_tool_catalog(
    config: SpiceConfig,
    *,
    subagents_enabled: bool | None = None,
) -> tuple[ToolCatalogEntry, ...]:
    """Describe every built-in tool under the effective feature configuration."""
    memory_enabled = config.memory_enabled
    effective_subagents = (
        config.subagents_enabled if subagents_enabled is None else subagents_enabled
    )
    registry = create_all_tools(memory_enabled=True, subagents_enabled=True)
    rows: list[ToolCatalogEntry] = []

    for toolset, names in TOOLSETS.items():
        feature_enabled = not (
            (toolset == "memory" and not memory_enabled)
            or (toolset == "subagent" and not effective_subagents)
        )
        for name in names:
            tool = registry.get(name)
            if tool is None:
                continue
            status: ToolStatus = "enabled"
            note = ""
            if not feature_enabled:
                status = "disabled"
                note = f"{toolset} is disabled in settings"
            elif name == "web_search" and not get_api_key("tavily"):
                status = "unavailable"
                note = "TAVILY_API_KEY is not configured"
            elif name == "browser_check" and not browser_dependency_available():
                status = "unavailable"
                note = "Install the browser extra: uv sync --extra browser; requires Chromium or Chrome"

            risk = (
                "read-only"
                if name in READ_ONLY_TOOLS and not tool.requires_confirmation
                else "write/exec"
            )
            rows.append(
                ToolCatalogEntry(
                    name=name,
                    toolset=toolset,
                    status=status,
                    risk=risk,
                    confirmation="required" if tool.requires_confirmation else "no",
                    execution=tool.concurrency,
                    timeout=f"{tool.timeout_seconds:g}s"
                    if tool.timeout_seconds is not None
                    else "default",
                    description=tool.description,
                    parameters=tool.parameters,
                    note=note,
                )
            )
    return tuple(rows)


def find_tool_entry(
    entries: tuple[ToolCatalogEntry, ...], name: str
) -> ToolCatalogEntry | None:
    normalized = name.strip().lower().replace("-", "_")
    return next((entry for entry in entries if entry.name == normalized), None)


def tool_summary(description: str, *, limit: int = 84) -> str:
    """Return a compact first-sentence description suitable for a table row."""
    first_line = " ".join(description.strip().splitlines())
    first_sentence = first_line.split(". ", 1)[0].rstrip(".")
    if len(first_sentence) <= limit:
        return first_sentence
    return f"{first_sentence[: limit - 1].rstrip()}…"
