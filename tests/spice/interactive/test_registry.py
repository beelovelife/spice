"""Tests for the shared interactive command registry."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from spice.interactive.commands import SlashCommandRegistry, handle_help
from spice.interactive.confirm import ConfirmPolicy
from spice.interactive.types import (
    ChoiceRequest,
    CommandContext,
    CommandResult,
    ConfirmRequest,
    PanelView,
    TableView,
    TextView,
)


class FakePort:
    def __init__(self, *, choice: str | None = None, decision: str = "deny") -> None:
        self.choice = choice
        self.decision = decision
        self.choices: list[ChoiceRequest] = []
        self.confirms: list[ConfirmRequest] = []

    async def choose(self, request: ChoiceRequest) -> str | None:
        self.choices.append(request)
        return self.choice

    async def confirm(self, request: ConfirmRequest) -> str:
        self.confirms.append(request)
        return self.decision


def test_help_completion_and_execute_share_same_registry() -> None:
    registry = SlashCommandRegistry()
    names = [cmd.name for cmd in registry.list_commands()]
    triggers = {cmd.trigger for cmd in registry.list_commands()}
    completion_triggers = {item[0] for item in registry.completion_items()}

    assert "mcp" in names
    assert "help" in names
    assert "quit" in names
    assert "/mcp" in triggers
    assert "/help" in completion_triggers
    assert "/exit" in completion_triggers
    # Every real command trigger is completable.
    for trigger in triggers:
        assert trigger in completion_triggers


def test_skill_completion_does_not_duplicate_colon_entry() -> None:
    items = SlashCommandRegistry().completion_items()
    texts = [trigger for trigger, _, _ in items if trigger.startswith("/skill")]
    assert texts == ["/skills", "/skill"]


@pytest.fixture
def skill_file(tmp_path, monkeypatch):
    monkeypatch.setattr("spice.skills.loader.USER_SKILLS_DIR", tmp_path / "user-skills")
    path = tmp_path / ".spice" / "skills" / "demo" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: demo\ndescription: 测试技能\n---\n检查项目并报告结果。\n",
        encoding="utf-8",
    )
    return path


def _execute_skill(raw: str, cwd: Path) -> CommandResult:
    return asyncio.run(
        SlashCommandRegistry().execute(
            raw,
            CommandContext(
                session=SimpleNamespace(),  # type: ignore[arg-type]
                cwd=cwd,
                port=FakePort(),
                raw=raw,
                args="",
                confirm_policy=ConfirmPolicy(),
            ),
        )
    )


@pytest.mark.parametrize(
    ("raw", "arguments"),
    [
        ("/skill:demo", ""),
        ("/skill demo", ""),
        ("/skill:demo 检查 src，保留现有改动", "检查 src，保留现有改动"),
        ("/skill demo 检查 src\n只报告错误", "检查 src\n只报告错误"),
    ],
)
def test_skill_invocation_returns_workflow_prompt(
    tmp_path, skill_file, raw, arguments
) -> None:
    result = _execute_skill(raw, tmp_path)

    assert result.handled is True
    assert result.session is None
    assert result.followup_prompt is not None
    assert skill_file.read_text(encoding="utf-8") in result.followup_prompt
    assert str(skill_file.parent.resolve()) in result.followup_prompt
    if arguments:
        assert result.followup_prompt.endswith(arguments)
    assert all(not isinstance(view, TextView) or view.style != "error" for view in result.views)


def test_skill_show_only_returns_content(tmp_path, skill_file) -> None:
    result = _execute_skill("/skill --show demo", tmp_path)

    assert result.followup_prompt is None
    assert result.views == [TextView(skill_file.read_text(encoding="utf-8"))]


@pytest.mark.parametrize(
    "raw", ["/skill", "/skill:", "/skill --show", "/skill --show demo extra", "/skill:missing"]
)
def test_invalid_skill_command_does_not_start_a_turn(tmp_path, skill_file, raw) -> None:
    result = _execute_skill(raw, tmp_path)

    assert result.followup_prompt is None
    assert any(isinstance(view, TextView) and view.style == "error" for view in result.views)


def test_skill_read_failure_does_not_start_a_turn(tmp_path, skill_file, monkeypatch) -> None:
    from spice.skills.loader import Skill, SkillLoadResult

    monkeypatch.setattr(
        "spice.interactive.commands.load_skills",
        lambda **kwargs: SkillLoadResult(
            [Skill("demo", "测试技能", skill_file.parent / "missing.md", "project")]
        ),
    )
    result = _execute_skill("/skill:demo", tmp_path)

    assert result.followup_prompt is None
    assert any(isinstance(view, TextView) and view.style == "error" for view in result.views)


def test_each_command_has_bound_handler() -> None:
    for command in SlashCommandRegistry().list_commands():
        assert callable(command.handler)


def test_help_handler_returns_table_view() -> None:
    registry = SlashCommandRegistry()
    session = SimpleNamespace()
    ctx = CommandContext(
        session=session,  # type: ignore[arg-type]
        cwd=Path.cwd(),
        port=FakePort(),
        raw="/help",
        args="",
        confirm_policy=ConfirmPolicy(),
        extras={"registry": registry},
    )
    result = asyncio.run(handle_help(ctx))
    assert result.handled is True
    assert any(isinstance(view, TableView) for view in result.views)
    table = next(view for view in result.views if isinstance(view, TableView))
    command_names = {row[0] for row in table.rows}
    assert "/mcp" in command_names
    assert "/help" in command_names


def test_quit_and_clear_flags() -> None:
    registry = SlashCommandRegistry()
    session = SimpleNamespace()
    policy = ConfirmPolicy()
    port = FakePort()

    async def run(raw: str) -> CommandResult:
        return await registry.execute(
            raw,
            CommandContext(
                session=session,  # type: ignore[arg-type]
                cwd=Path.cwd(),
                port=port,
                raw=raw,
                args="",
                confirm_policy=policy,
            ),
        )

    quit_result = asyncio.run(run("/quit"))
    assert quit_result.exit_requested is True

    clear_result = asyncio.run(run("/clear"))
    assert clear_result.clear_requested is True


def test_thinking_command_reads_frontend_ephemeral_reasoning() -> None:
    registry = SlashCommandRegistry()
    result = asyncio.run(
        registry.execute(
            "/thinking",
            CommandContext(
                session=SimpleNamespace(),  # type: ignore[arg-type]
                cwd=Path.cwd(),
                port=FakePort(),
                raw="/thinking",
                args="",
                confirm_policy=ConfirmPolicy(),
                extras={"last_reasoning": lambda: "checked the repository"},
            ),
        )
    )

    assert result.views == [PanelView("Latest thinking", "checked the repository")]


def test_unknown_command_returns_error_view() -> None:
    registry = SlashCommandRegistry()
    result = asyncio.run(
        registry.execute(
            "/nope",
            CommandContext(
                session=SimpleNamespace(),  # type: ignore[arg-type]
                cwd=Path.cwd(),
                port=FakePort(),
                raw="/nope",
                args="",
                confirm_policy=ConfirmPolicy(),
            ),
        )
    )
    assert any(isinstance(v, TextView) and "Unknown command" in v.text for v in result.views)


def test_cli_and_core_command_sets_match() -> None:
    from spice.cli.commands import SlashCommandRegistry as CliRegistry

    core = {cmd.name for cmd in SlashCommandRegistry().list_commands()}
    cli = {cmd.name for cmd in CliRegistry().commands}
    assert core == cli
    assert "mcp" in core
