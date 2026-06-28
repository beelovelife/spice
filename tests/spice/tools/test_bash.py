from __future__ import annotations

import asyncio

import pytest

from spice.tools.base import ToolContext
from spice.tools.bash import bash, create_bash_tools


def test_create_bash_tools_registers_confirmed_shell_tool() -> None:
    tools = create_bash_tools()

    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "bash"
    assert tool.requires_confirmation is True
    assert "command" in tool.parameters["required"]
    assert tool.parameters["properties"]["timeout"]["maximum"] == 600


def test_bash_runs_command_in_workspace(tmp_path) -> None:
    result = asyncio.run(
        bash(
            {"command": "pwd", "timeout": 5},
            ToolContext(cwd=tmp_path),
        )
    )

    assert result.is_error is False
    assert str(tmp_path) in result.content
    assert result.details["exit_code"] == 0


def test_bash_reports_nonzero_exit_as_error(tmp_path) -> None:
    result = asyncio.run(
        bash(
            {"command": "printf problem >&2; exit 7", "timeout": 5},
            ToolContext(cwd=tmp_path),
        )
    )

    assert result.is_error is True
    assert "problem" in result.content
    assert result.details["exit_code"] == 7


def test_bash_rejects_empty_command(tmp_path) -> None:
    result = asyncio.run(bash({"command": ""}, ToolContext(cwd=tmp_path)))

    assert result.is_error is True
    assert "command is required" in result.content


def test_bash_kills_process_group_on_cancellation(tmp_path, monkeypatch) -> None:
    killed: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 12345
        returncode = None

        async def communicate(self):
            await asyncio.sleep(60)
            return b"", b""

        async def wait(self):
            self.returncode = -9
            return self.returncode

    async def fake_create_subprocess_shell(*args, **kwargs):
        return FakeProcess()

    def fake_killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_create_subprocess_shell)
    monkeypatch.setattr("spice.tools.bash.os.killpg", fake_killpg)

    async def run_and_cancel() -> None:
        task = asyncio.create_task(bash({"command": "sleep 60", "timeout": 30}, ToolContext(cwd=tmp_path)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())

    assert killed
