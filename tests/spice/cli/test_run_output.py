from __future__ import annotations

import asyncio
import json
from io import StringIO

import pytest
from typer.testing import CliRunner

import spice.cli.main as cli_main
from spice.agent.events import (
    AgentEndEvent, AgentErrorEvent, AgentStartEvent, AssistantMessageEvent,
    TextDeltaEvent, ToolExecutionEndEvent, ToolExecutionStartEvent, TurnEndEvent,
    TurnFinalizationStartEvent,
)
from spice.agent.turn_outcome import TurnCheckpoint
from spice.cli.run_output import RunOutput
from spice.tools.base import tool_result


def install_session(monkeypatch, *, error=None, empty=False, paused=False):
    closed = []

    class Session:
        session_label = "test-session"

        def __init__(self, **kwargs):
            assert kwargs["confirm"] is None  # Captured/noninteractive input cannot approve tools.

        async def prompt(self, _prompt):
            yield AgentStartEvent("test-session")
            if error:
                yield AgentErrorEvent("diagnostic", kind=error)
                yield TurnEndEvent("", stop_reason=error)
            elif paused:
                checkpoint = TurnCheckpoint("task", 100, 100, summary="已保存进度，剩余验收。")
                yield TurnFinalizationStartEvent(100, 100)
                yield TextDeltaEvent(checkpoint.summary)
                yield AssistantMessageEvent(checkpoint.summary, [])
                yield TurnEndEvent(checkpoint.summary, "max_tool_rounds", checkpoint=checkpoint)
            elif not empty:
                yield ToolExecutionStartEvent("call", "read_file", {})
                yield ToolExecutionEndEvent("call", "read_file", tool_result("tool data", full_content="private buffer"))
                yield TextDeltaEvent("answer [literal]")
                yield AssistantMessageEvent("answer [literal]", [])
                yield TurnEndEvent("answer [literal]", "stop")
            yield AgentEndEvent("test-session")

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(cli_main, "AgentSession", Session)
    return closed


def test_run_keeps_stdout_scriptable(monkeypatch):
    closed = install_session(monkeypatch)
    result = CliRunner().invoke(cli_main.app, ["run", "hello"])
    assert result.exit_code == 0, result.exception
    assert result.stdout == "answer [literal]\n"
    assert "Session: test-session" in result.stderr
    assert "Tool: read_file" in result.stderr
    assert "\x1b" not in result.stdout and "\r" not in result.stdout
    assert closed == [True]


def test_run_json_is_a_versioned_event_stream(monkeypatch):
    closed = install_session(monkeypatch)
    result = CliRunner().invoke(cli_main.app, ["run", "hello", "--json"])
    assert result.exit_code == 0, result.exception
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert all(row["format"] == "spice-events-1" for row in rows)
    assert rows[0]["type"] == "agent_start"
    assert rows[-1]["type"] == "agent_end"
    assert "private buffer" not in result.stdout
    assert closed == [True]


@pytest.mark.parametrize("error,code", [("error", 1), ("context_limit", 1), ("user_interrupted", 130)])
@pytest.mark.parametrize("json_mode", [False, True])
def test_run_failures_have_nonzero_exit_codes(monkeypatch, error, code, json_mode):
    closed = install_session(monkeypatch, error=error)
    result = CliRunner().invoke(cli_main.app, ["run", "hello", *(["--json"] if json_mode else [])])
    assert result.exit_code == code
    assert "diagnostic" in result.stderr
    if not json_mode:
        assert result.stdout == ""
    assert closed == [True]


def test_run_without_turn_end_does_not_claim_success(monkeypatch):
    install_session(monkeypatch, empty=True)
    result = CliRunner().invoke(cli_main.app, ["run", "hello"])
    assert result.exit_code == 1


@pytest.mark.parametrize("json_mode", [False, True])
def test_paused_run_has_distinct_exit_code_and_a_persistable_checkpoint(monkeypatch, json_mode):
    closed = install_session(monkeypatch, paused=True)
    result = CliRunner().invoke(cli_main.app, ["run", "work", *(["--json"] if json_mode else [])])
    assert result.exit_code == 2
    assert "暂停待继续" in result.stderr
    assert "Error:" not in result.stderr
    assert "\x1b" not in result.stdout
    if json_mode:
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        end = next(row["data"] for row in rows if row["type"] == "turn_end")
        assert end["status"] == "paused"
        assert end["checkpoint"]["round_limit"] == 100
        assert end["checkpoint"]["summary"] == "已保存进度，剩余验收。"
    else:
        assert result.stdout == "已保存进度，剩余验收。\n"
    assert closed == [True]


def test_run_setup_errors_are_sanitized_and_sent_to_stderr(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("api_key=private-value")

    monkeypatch.setattr(cli_main, "AgentSession", fail)
    result = CliRunner().invoke(cli_main.app, ["run", "hello"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "private-value" not in result.stderr
    assert "[redacted]" in result.stderr


def test_consumer_error_closes_prompt_before_session_resources():
    closed = []

    class BrokenOutput(StringIO):
        def write(self, text):
            raise BrokenPipeError("pipe closed")

    class Session:
        async def prompt(self, _prompt):
            try:
                yield TextDeltaEvent("answer")
                await asyncio.Event().wait()
            finally:
                closed.append("prompt")

        async def aclose(self):
            closed.append("session")

    output = RunOutput(stdout=BrokenOutput(), stderr=StringIO())
    with pytest.raises(BrokenPipeError):
        asyncio.run(cli_main._run_once(Session(), output, "run"))
    assert closed == ["prompt", "session"]
