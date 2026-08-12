from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spice.skills.loader import load_skills, read_skill_content
from spice.tools.base import ToolContext
from spice.tools.skill import skill_view, skills_list


def _write_skill(root: Path, name: str, description: str, body: str = "Body") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_md


class SkillLoaderTests(unittest.TestCase):
    def test_loads_path_project_and_user_with_collision_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "repo"
            user_dir = root / "home-skills"
            explicit_dir = root / "explicit"
            project_dir = cwd / ".spice" / "skills"

            _write_skill(user_dir, "global-skill", "Global helper")
            _write_skill(project_dir, "project-skill", "Project helper")
            _write_skill(user_dir, "same-name", "User copy")
            _write_skill(project_dir, "same-name", "Project copy")
            explicit_skill = _write_skill(explicit_dir, "path-skill", "Explicit helper")

            result = load_skills(cwd=cwd, skill_paths=[explicit_skill], user_skills_dir=user_dir)
            by_name = {skill.name: skill for skill in result.skills}

            self.assertEqual(by_name["path-skill"].source, "path")
            self.assertEqual(by_name["project-skill"].source, "project")
            self.assertEqual(by_name["global-skill"].source, "user")
            self.assertEqual(by_name["same-name"].source, "project")
            self.assertTrue(any(diagnostic.type == "collision" and diagnostic.name == "same-name" for diagnostic in result.diagnostics))

    def test_read_skill_content_supports_linked_files_and_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            skill_md = _write_skill(cwd / ".spice" / "skills", "repo-reader", "Read this repo", "Instructions")
            references = skill_md.parent / "references"
            references.mkdir()
            (references / "flow.md").write_text("Flow reference", encoding="utf-8")

            self.assertIn("Instructions", read_skill_content("repo-reader", cwd=cwd))
            self.assertEqual(read_skill_content("repo-reader", "references/flow.md", cwd=cwd), "Flow reference")
            with self.assertRaisesRegex(ValueError, "stay inside"):
                read_skill_content("repo-reader", "../../outside.md", cwd=cwd)


class SkillToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_skills_list_and_skill_view_include_source_and_linked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            skill_md = _write_skill(cwd / ".spice" / "skills", "repo-reader", "Read this repo", "Instructions")
            references = skill_md.parent / "references"
            references.mkdir()
            (references / "flow.md").write_text("Flow reference", encoding="utf-8")

            context = ToolContext(cwd=cwd)
            list_result = await skills_list({}, context)
            self.assertFalse(list_result.is_error)
            self.assertIn("- repo-reader [project]: Read this repo", list_result.content)

            view_result = await skill_view({"name": "repo-reader"}, context)
            self.assertFalse(view_result.is_error)
            self.assertIn("Instructions", view_result.content)
            self.assertIn("linked_files:", view_result.content)
            self.assertIn("- references/flow.md", view_result.content)

            linked_result = await skill_view({"name": "repo-reader", "file_path": "references/flow.md"}, context)
            self.assertFalse(linked_result.is_error)
            self.assertEqual(linked_result.content, "Flow reference")
