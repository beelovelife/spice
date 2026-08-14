"""Load Spice skills from user, project, and explicit paths."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spice.llm.config import CONFIG_DIR

USER_SKILLS_DIR = CONFIG_DIR / "skills"
PROJECT_SKILLS_DIR = ".spice/skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    source: str
    triggers: list[str] = field(default_factory=list)
    always: bool = False

    @property
    def priority(self) -> int:
        return {"user": 0, "project": 1, "path": 2}.get(self.source, 0)


@dataclass(frozen=True)
class SkillDiagnostic:
    type: str
    name: str
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class SkillLoadResult:
    skills: list[Skill]
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)


def load_skills(
    *,
    cwd: Path,
    skill_paths: list[Path] | None = None,
    user_skills_dir: Path | None = None,
) -> SkillLoadResult:
    diagnostics: list[SkillDiagnostic] = []
    by_name: dict[str, Skill] = {}

    sources: list[tuple[str, list[Path]]] = [
        ("user", _skill_files_in(user_skills_dir or USER_SKILLS_DIR)),
        ("project", _project_skill_files(cwd)),
        ("path", [path for path in (skill_paths or [])]),
    ]
    priority = {"user": 0, "project": 1, "path": 2}

    for source, paths in sources:
        for path in paths:
            try:
                skill = read_skill_file(path, source=source)
            except ValueError as exc:
                diagnostics.append(SkillDiagnostic("invalid", path.stem, str(exc), path))
                continue
            existing = by_name.get(skill.name)
            if existing and priority[source] >= priority[existing.source]:
                diagnostics.append(
                    SkillDiagnostic(
                        "collision",
                        skill.name,
                        f"Skill {skill.name!r} from {source} overrides {existing.source}.",
                        path,
                    )
                )
                by_name[skill.name] = skill
            elif existing:
                diagnostics.append(
                    SkillDiagnostic(
                        "collision",
                        skill.name,
                        f"Skill {skill.name!r} from {source} ignored; {existing.source} has priority.",
                        path,
                    )
                )
            else:
                by_name[skill.name] = skill
    return SkillLoadResult(skills=sorted(by_name.values(), key=lambda item: item.name), diagnostics=diagnostics)


def read_skill_file(path: Path, *, source: str = "path") -> Skill:
    if path.is_dir():
        path = path / "SKILL.md"
    if not path.is_file():
        raise ValueError(f"Skill file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    metadata = _frontmatter(text)
    name = str(metadata.get("name") or path.parent.name).strip()
    description = str(metadata.get("description") or "").strip()
    if not name:
        raise ValueError("Skill name is required.")
    if not description:
        raise ValueError(f"Skill {name!r} is missing a description.")
    triggers = metadata.get("triggers")
    if isinstance(triggers, str):
        trigger_list = [triggers]
    elif isinstance(triggers, list):
        trigger_list = [str(item) for item in triggers if str(item).strip()]
    else:
        trigger_list = []
    return Skill(
        name=name,
        description=description,
        path=path,
        source=source,
        triggers=trigger_list,
        always=bool(metadata.get("always", False)),
    )


def read_skill_content(name: str, file_path: str | None = None, *, cwd: Path) -> str:
    result = load_skills(cwd=cwd)
    skill = next((item for item in result.skills if item.name == name), None)
    if skill is None:
        raise ValueError(f"Skill not found: {name}")
    target = skill.path
    if file_path:
        root = skill.path.parent.resolve()
        target = (root / file_path).resolve()
        if target != root and root not in target.parents:
            raise ValueError("Skill file must stay inside the skill directory.")
        if not target.is_file():
            raise ValueError(f"Linked skill file does not exist: {file_path}")
    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read skill file: {target}") from exc


def linked_files(skill: Skill) -> list[Path]:
    root = skill.path.parent
    rows: list[Path] = []
    for directory in (root / "references", root / "assets"):
        if not directory.exists():
            continue
        rows.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
    return rows


def _skill_files_in(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(path for path in root.glob("*/SKILL.md") if path.is_file())


def _project_skill_files(cwd: Path) -> list[Path]:
    current = cwd.resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        root = path / PROJECT_SKILLS_DIR
        if root.exists():
            return _skill_files_in(root)
    return []


def _frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    match = re.match(r"^---\n(.*?)\n---(?:\n|$)", text, re.S)
    if not match:
        return {}
    data: dict[str, Any] = {}
    current_list: str | None = None
    for raw_line in match.group(1).splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_list:
            data.setdefault(current_list, []).append(line[4:].strip())
            continue
        current_list = None
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_list = key
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            data[key] = value.strip("'\"")
    return data
