from __future__ import annotations

import os
from pathlib import Path

import spice.agent.prompts as prompts_module
from spice.agent.prompts import build_system_prompt
from spice.skills.loader import SkillLoadResult
from spice.tools.base import Tool, tool_result


async def _noop(args, context):
    return tool_result("ok")


def test_system_prompt_delegates_tool_choice_to_tool_descriptions() -> None:
    prompt = build_system_prompt(
        Path("/workspace"),
        [
            Tool(
                name="write_file",
                description="Write a file.",
                parameters={"type": "object"},
                execute=_noop,
                requires_confirmation=True,
            )
        ],
    )

    assert "Use tools when they are needed" in prompt
    assert "If no tool is needed, answer directly" in prompt
    assert "Choose tools according to their descriptions" in prompt
    assert "Do not use tools for pure content generation" not in prompt


def test_system_prompt_includes_tool_contract() -> None:
    prompt = build_system_prompt(Path("/workspace"), [])

    assert "Tool contract:" in prompt
    assert "Before calling edit_file, apply_patch replace/delete, or write_file on an existing file" in prompt
    assert "Do not rely on the tool error to remind you" in prompt
    assert "Do not use bash/python/sed/perl scripts to modify workspace files" in prompt
    assert "If a tool call fails, adjust the next call" in prompt


def test_system_prompt_lists_tool_summaries() -> None:
    prompt = build_system_prompt(
        Path("/workspace"),
        [
            Tool(
                name="read_file",
                description="Read a UTF-8 text file from the workspace. Use this instead of shell commands.",
                parameters={"type": "object"},
                execute=_noop,
            )
        ],
    )

    assert "Available tools:\n- read_file: Read a UTF-8 text file from the workspace." in prompt
    assert "Use this instead of shell commands" not in prompt


def test_system_prompt_includes_todo_guidance_when_tool_available() -> None:
    prompt = build_system_prompt(
        Path("/workspace"),
        [
            Tool(
                name="update_todo",
                description="Manage the task list.",
                parameters={"type": "object"},
                execute=_noop,
            )
        ],
    )

    assert "maintain progress with update_todo" in prompt
    assert "spans multiple files or modules" in prompt
    assert "substantial tool exploration" in prompt
    assert "Examples include repo analysis" in prompt
    assert "call update_todo before other exploratory" in prompt
    assert "Follow the detailed update_todo tool description" in prompt
    assert "merge behavior" in prompt


def test_system_prompt_omits_todo_guidance_when_tool_unavailable() -> None:
    prompt = build_system_prompt(Path("/workspace"), [])

    assert "maintain progress with update_todo" not in prompt


def test_system_prompt_includes_current_runtime_model() -> None:
    prompt = build_system_prompt(Path("/workspace"), [], runtime_model="deepseek/deepseek-v4-flash")

    assert "Current runtime model: deepseek/deepseek-v4-flash" in prompt


def test_system_prompt_includes_memory_context() -> None:
    prompt = build_system_prompt(Path("/workspace"), [], memory_context="Persistent memory snapshot:\n- User prefers Chinese.")

    assert "Persistent memory snapshot:" in prompt
    assert "User prefers Chinese." in prompt


def test_system_prompt_omits_empty_memory_context() -> None:
    prompt = build_system_prompt(Path("/workspace"), [], memory_context="")

    assert "Persistent memory snapshot:" not in prompt


def test_system_prompt_handles_empty_tools() -> None:
    prompt = build_system_prompt(Path("/workspace"), [])

    assert "Available tools:\n(none)" in prompt


def test_system_prompt_omits_spice_context_when_files_are_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", tmp_path / "global")

    prompt = build_system_prompt(tmp_path / "project", [])

    assert "SPICE.md instructions:" not in prompt


def test_system_prompt_omits_blank_spice_context_files(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    (global_dir / "SPICE.md").write_text("  \n\t\n", encoding="utf-8")
    (project_dir / "SPICE.md").write_text("\n\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", global_dir)

    prompt = build_system_prompt(project_dir, [])

    assert "SPICE.md instructions:" not in prompt


def test_system_prompt_loads_global_spice_context(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    (global_dir / "SPICE.md").write_text("Prefer concise answers.\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", global_dir)

    prompt = build_system_prompt(project_dir, [])

    assert "SPICE.md instructions:" in prompt
    assert "Global SPICE.md" in prompt
    assert "Prefer concise answers." in prompt
    assert "Project SPICE.md" not in prompt


def test_system_prompt_loads_project_spice_context(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    (project_dir / "SPICE.md").write_text("Use project conventions.\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", global_dir)

    prompt = build_system_prompt(project_dir, [])

    assert "SPICE.md instructions:" in prompt
    assert "Project SPICE.md" in prompt
    assert "Use project conventions." in prompt
    assert "Global SPICE.md" not in prompt


def test_system_prompt_loads_global_before_project_spice_context(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    skill_dir = project_dir / ".spice" / "skills" / "repo-reader"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: repo-reader\n"
        "description: Read a repository.\n"
        "---\n",
        encoding="utf-8",
    )
    (global_dir / "SPICE.md").write_text("Global preference.\n", encoding="utf-8")
    (project_dir / "SPICE.md").write_text("Project preference.\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", global_dir)

    prompt = build_system_prompt(project_dir, [])

    assert prompt.index("Global preference.") < prompt.index("Project preference.")
    assert prompt.index("SPICE.md instructions:") < prompt.index("Available skills:")


def test_system_prompt_finds_nearest_project_spice_context_from_child_directory(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    child_dir = project_dir / "src" / "package"
    global_dir.mkdir()
    child_dir.mkdir(parents=True)
    (project_dir / "SPICE.md").write_text("Parent project instructions.\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", global_dir)

    prompt = build_system_prompt(child_dir, [])

    assert "Parent project instructions." in prompt


def test_system_prompt_lists_available_skills(tmp_path) -> None:
    skill_dir = tmp_path / ".spice" / "skills" / "repo-reader"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: repo-reader\n"
        "description: Read a repository from its CLI entrypoint downward.\n"
        "---\n"
        "\n"
        "Start at the requested entrypoint.\n",
        encoding="utf-8",
    )

    prompt = build_system_prompt(tmp_path, [])

    assert "Available skills:" in prompt
    assert "- repo-reader [project]: Read a repository from its CLI entrypoint downward." in prompt
    assert "Skill descriptions are an index only" in prompt


def test_system_prompt_lists_skill_metadata_without_body(tmp_path) -> None:
    skill_dir = tmp_path / ".spice" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: frontend\n"
        "description: Build UI.\n"
        "triggers:\n"
        "  - react\n"
        "  - css\n"
        "always: true\n"
        "---\n"
        "\n"
        "Long private instructions.\n",
        encoding="utf-8",
    )

    prompt = build_system_prompt(tmp_path, [])

    assert "- frontend [project]: Build UI. (triggers: react, css; always)" in prompt
    assert "Long private instructions." not in prompt


def test_system_prompt_reuses_static_context_cache(tmp_path, monkeypatch) -> None:
    calls = {"spice": 0, "skills": 0}

    def fake_load_spice_context(cwd: Path) -> str | None:
        calls["spice"] += 1
        return "Cached context."

    def fake_load_skills(*, cwd: Path) -> SkillLoadResult:
        calls["skills"] += 1
        return SkillLoadResult(skills=[])

    monkeypatch.setattr(prompts_module, "load_spice_context", fake_load_spice_context)
    monkeypatch.setattr(prompts_module, "load_skills", fake_load_skills)

    first = build_system_prompt(tmp_path, [])
    second = build_system_prompt(tmp_path, [])

    assert "Cached context." in first
    assert "Cached context." in second
    assert calls == {"spice": 1, "skills": 1}


def test_system_prompt_refreshes_cached_spice_context_when_file_changes(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    spice_file = project_dir / "SPICE.md"
    spice_file.write_text("First instruction.\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "CONFIG_DIR", global_dir)

    first = build_system_prompt(project_dir, [])

    spice_file.write_text("Second instruction.\n", encoding="utf-8")
    stat = spice_file.stat()
    os.utime(spice_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = build_system_prompt(project_dir, [])

    assert "First instruction." in first
    assert "Second instruction." in second
    assert "First instruction." not in second


def test_system_prompt_refreshes_cached_skills_when_skill_file_changes(tmp_path) -> None:
    skill_file = tmp_path / ".spice" / "skills" / "repo-reader" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\n"
        "name: repo-reader\n"
        "description: First description.\n"
        "---\n",
        encoding="utf-8",
    )

    first = build_system_prompt(tmp_path, [])

    skill_file.write_text(
        "---\n"
        "name: repo-reader\n"
        "description: Second description.\n"
        "---\n",
        encoding="utf-8",
    )
    stat = skill_file.stat()
    os.utime(skill_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = build_system_prompt(tmp_path, [])

    assert "First description." in first
    assert "Second description." in second
    assert "First description." not in second
