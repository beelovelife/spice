"""Tests for ConfirmPolicy."""

from __future__ import annotations

import asyncio

from spice.interactive.confirm import ConfirmPolicy, is_file_edit_tool, is_read_only_bash_command
from spice.interactive.types import ChoiceRequest, ConfirmRequest


class FakePort:
    def __init__(self, decision: str = "allow") -> None:
        self.decision = decision
        self.calls = 0

    async def choose(self, request: ChoiceRequest) -> str | None:
        return None

    async def confirm(self, request: ConfirmRequest) -> str:
        self.calls += 1
        return self.decision


def test_file_edit_session_allow_skips_second_prompt() -> None:
    policy = ConfirmPolicy()
    port = FakePort(decision="allow_all_edits")

    first = asyncio.run(policy.confirm("write_file", {"path": "a.py"}, port))
    second = asyncio.run(policy.confirm("write_file", {"path": "b.py"}, port))

    assert first is True
    assert second is True
    assert port.calls == 1
    assert policy.allow_file_edits is True


def test_new_policy_does_not_inherit_prior_session_flags() -> None:
    first = ConfirmPolicy()
    first.allow_file_edits = True
    second = ConfirmPolicy()
    assert second.allow_file_edits is False
    assert is_file_edit_tool("edit_file")
    assert second.precheck("edit_file", {"path": "x.py"}) is None


def test_mutating_command_forms_are_not_read_only_bash() -> None:
    commands = (
        "sort -o target.txt input.txt",
        "uniq input.txt output.txt",
        "git branch -D old",
        "git remote add origin https://example.com/repo.git",
        "git diff --output=changes.patch",
        "find . -fprint matches.txt",
        "awk 'BEGIN { print | \"touch marker\" }' data.txt",
    )

    for command in commands:
        assert is_read_only_bash_command({"command": command}) is False
