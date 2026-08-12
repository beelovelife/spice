"""Tests for the shared interactive command registry."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from spice.interactive.commands import SlashCommandRegistry, handle_help
from spice.interactive.confirm import ConfirmPolicy
from spice.interactive.types import (
    ChoiceRequest,
    CommandContext,
    CommandResult,
    ConfirmRequest,
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
