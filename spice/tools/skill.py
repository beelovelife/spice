"""Skill inspection tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spice.skills.loader import linked_files, load_skills
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result


async def skills_list(args: dict[str, Any], context: ToolContext) -> ToolResult:
    result = load_skills(cwd=context.cwd)
    lines = [f"- {skill.name} [{skill.source}]: {skill.description}" for skill in result.skills]
    if result.diagnostics:
        lines.append("")
        lines.append("diagnostics:")
        lines.extend(f"- {diagnostic.type}: {diagnostic.message}" for diagnostic in result.diagnostics)
    return tool_result("\n".join(lines) if lines else "(no skills)", {"skill_count": len(result.skills)})


async def skill_view(args: dict[str, Any], context: ToolContext) -> ToolResult:
    name = str(args.get("name") or "")
    file_path = str(args.get("file_path") or "")
    result = load_skills(cwd=context.cwd)
    skill = next((item for item in result.skills if item.name == name), None)
    if skill is None:
        return tool_error(f"Skill not found: {name}")
    if file_path:
        target = (skill.path.parent / file_path).resolve()
        root = skill.path.parent.resolve()
        if root not in (target, *target.parents):
            return tool_error("file_path must stay inside the skill directory.")
        if not target.is_file():
            return tool_error(f"Linked skill file does not exist: {file_path}")
        return tool_result(target.read_text(encoding="utf-8"), {"path": str(target)})

    content = skill.path.read_text(encoding="utf-8")
    linked = linked_files(skill)
    if linked:
        content = content.rstrip() + "\n\nlinked_files:\n"
        content += "\n".join(f"- {path.relative_to(skill.path.parent)}" for path in linked)
    return tool_result(content, {"path": str(skill.path), "source": skill.source})


def create_skill_tools() -> list[Tool]:
    return [
        Tool(
            name="skills_list",
            description="List available Spice skills with source and short descriptions.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=skills_list,
        ),
        Tool(
            name="skill_view",
            description="Read full instructions for one skill, or one linked file inside that skill.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "file_path": {"type": "string"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            execute=skill_view,
        ),
    ]
