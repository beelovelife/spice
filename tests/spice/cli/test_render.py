from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

from rich.console import Console

import spice.cli.render as render_module
from spice.agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    AgentErrorEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.cli.render import CliRenderer
from spice.tools.base import tool_error, tool_result


def _console() -> tuple[Console, StringIO]:
    output = StringIO()
    return Console(file=output, force_terminal=False, color_system=None, width=100), output


def test_markdown_renderer_renders_streamed_tables() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="table"))
    renderer.render_event(TextDeltaEvent("| Name | Notes |\n| --- | --- |\n| a | short |\n| longer | x |"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))

    rendered = output.getvalue()
    assert "| Name | Notes |" not in rendered
    assert "| --- | --- |" not in rendered
    assert "Name" in rendered
    assert "Notes" in rendered
    assert "longer" in rendered
    assert "╭" in rendered
    assert "╰" in rendered
    assert rendered.endswith("\r\n")


def test_markdown_renderer_flushes_table_after_tool_round() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="table after tool"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[object()]))  # type: ignore[list-item]
    renderer.render_event(
        ToolExecutionStartEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            args={"todos": [{"id": "1", "content": "Analyze", "status": "in_progress"}], "merge": False},
        )
    )
    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            result=tool_result(
                "",
                details={
                    "todos": [{"id": "1", "content": "Analyze", "status": "completed"}],
                    "summary": {"total": 1, "pending": 0, "in_progress": 0, "completed": 1, "cancelled": 0},
                },
            ),
        )
    )
    renderer.render_event(TextDeltaEvent("#### 四、外部依赖\n\n| 包 | 用途 |\n| --- | --- |\n| `rich` | 渲染 |\n"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))
    renderer.render_event(TurnEndEvent(text=""))

    rendered = output.getvalue()
    assert "四、外部依赖" in rendered
    assert "包" in rendered
    assert "用途" in rendered
    assert "rich" in rendered
    assert "╭" in rendered


def test_renderer_ignores_agent_lifecycle_events() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(AgentStartEvent(session_id="session"))
    renderer.render_event(AgentEndEvent(session_id="session"))

    assert output.getvalue() == ""


def test_renderer_shows_retry_fallback_and_fatal_tool_stop() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(ModelRetryEvent("openai", "primary", 1, 2, 3, 0.5, "temporary"))
    renderer.render_event(ModelFallbackEvent("primary", "openai", "primary", "backup", "anthropic", "backup", "server", 0, 1))
    renderer.render_event(AgentErrorEvent("Tool denied by user: bash", kind="fatal_tool"))

    rendered = output.getvalue()
    assert "Retrying 2/3 in 0.5s" in rendered
    assert "openai/primary -> anthropic/backup" in rendered
    assert "Current turn stopped: Tool denied by user: bash" in rendered


def test_markdown_renderer_streams_plain_text_before_turn_end() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="plain"))
    renderer.render_event(TextDeltaEvent("你好，这是一段普通回答"))

    rendered = output.getvalue()
    assert "你好，这是一段普通回答" in rendered
    assert "╭" not in rendered


def test_renderer_does_not_emit_waiting_indicator_for_non_terminal_output() -> None:
    console, output = _console()
    renderer = CliRenderer(console, markdown=False)

    renderer.render_event(TurnStartEvent(prompt="plain"))
    renderer.refresh_waiting_indicator()
    renderer.render_event(TextDeltaEvent("你好"))
    renderer.render_event(AssistantMessageEvent(text="你好", tool_calls=[]))

    rendered = output.getvalue()
    assert rendered == "Spice: 你好\r\n\r\n"
    assert "thinking" not in rendered


def test_renderer_prints_final_text_when_no_delta_was_streamed() -> None:
    console, output = _console()
    renderer = CliRenderer(console, markdown=False)

    renderer.render_event(TurnStartEvent(prompt="plain"))
    renderer.render_event(AssistantMessageEvent(text="final answer", tool_calls=[]))

    assert output.getvalue() == "Spice: final answer\r\n\r\n"


def test_markdown_renderer_finishes_plain_stream_on_new_line() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="plain"))
    renderer.render_event(TextDeltaEvent("最后一句没有换行"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))

    rendered = output.getvalue()
    assert rendered.endswith("最后一句没有换行\r\n\r\n")


def test_markdown_renderer_finishes_text_before_tool_start() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="plain then tool"))
    renderer.render_event(TextDeltaEvent("先说明一句，不带换行"))
    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="bash", args={"command": "echo ok"}))

    rendered = output.getvalue()
    assert "先说明一句，不带换行\r\n❯ Run: echo ok" in rendered


def test_markdown_renderer_flushes_pending_table_before_tool_start() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="table then tool"))
    renderer.render_event(TextDeltaEvent("| 名称 | 值 |\n| --- | --- |\n| a | b |"))
    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="bash", args={"command": "echo ok"}))

    rendered = output.getvalue()
    assert "名称" in rendered
    assert "值" in rendered
    assert "╭" in rendered
    assert "❯ Run: echo ok" in rendered
    assert rendered.index("╭") < rendered.index("❯ Run: echo ok")


def test_markdown_renderer_does_not_treat_pipe_text_as_table_without_divider() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="pipe"))
    renderer.render_event(TextDeltaEvent("| this is just a pipe line |\nnot a table"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))

    rendered = output.getvalue()
    assert "| this is just a pipe line |" in rendered
    assert "not a table" in rendered
    assert "╭" not in rendered


def test_plain_renderer_keeps_direct_streaming_output() -> None:
    console, output = _console()
    renderer = CliRenderer(console, markdown=False)

    renderer.render_event(TurnStartEvent(prompt="table"))
    renderer.render_event(TextDeltaEvent("| Name | Notes |"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))

    rendered = output.getvalue()
    assert "| Name | Notes |" in rendered
    assert "╭" not in rendered


def test_plain_renderer_finishes_stream_on_new_line() -> None:
    console, output = _console()
    renderer = CliRenderer(console, markdown=False)

    renderer.render_event(TurnStartEvent(prompt="plain"))
    renderer.render_event(TextDeltaEvent("raw text"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))

    assert output.getvalue().endswith("raw text\r\n\r\n")


def test_finish_response_uses_carriage_return_before_next_prompt() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(TurnStartEvent(prompt="plain"))
    renderer.render_event(TextDeltaEvent("answer"))
    renderer.finish_response()
    output.write("spice ❯ ")

    assert output.getvalue().endswith("answer\r\n\r\nspice ❯ ")


def test_renderer_summarizes_read_file_results() -> None:
    console, output = _console()
    renderer = CliRenderer(console)
    content = "\n".join(f"line {index}" for index in range(40))

    renderer.render_event(ToolExecutionEndEvent(tool_call_id="tc1", tool_name="read_file", result=tool_result(content)))

    rendered = output.getvalue()
    assert "Read 40 lines" in rendered
    # body content must not leak into the summary line
    assert "line 0" not in rendered
    assert "line 20" not in rendered


def test_renderer_summarizes_read_files_results() -> None:
    console, output = _console()
    renderer = CliRenderer(console)
    content = "--- a.txt ---\na\n\n--- b.txt ---\nb\n"

    renderer.render_event(
        ToolExecutionStartEvent(
            tool_call_id="tc1",
            tool_name="read_files",
            args={"files": [{"path": "a.txt"}, {"path": "b.txt"}]},
        )
    )
    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="read_files",
            result=tool_result(content),
        )
    )
    renderer.finish_response()

    rendered = output.getvalue()
    assert "Read 2 files" in rendered
    assert "Read 2 files · 5 lines · 33 chars" in rendered


def test_renderer_summarizes_list_dir_details_without_entries() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="list_dir",
            result=tool_result(
                "agent/\ncli/\nREADME.md\n.pytest_cache/\n",
                details={"total_entries": 4, "dir_count": 3, "file_count": 1, "other_count": 0, "hidden_count": 1},
            ),
        )
    )

    rendered = output.getvalue()
    assert "Listed 4 entries · 3 dirs · 1 file · 1 hidden" in rendered
    assert "README.md" not in rendered


def test_renderer_summarizes_bash_details_without_output() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="bash",
            result=tool_result(
                "very noisy line\n" * 20,
                details={"returncode": 0, "stdout_lines": 20, "stderr_lines": 0, "stdout_chars": 320, "stderr_chars": 0},
            ),
        )
    )

    rendered = output.getvalue()
    assert "Exit 0 · stdout 20 lines · stderr 0 lines" in rendered
    assert "very noisy line" not in rendered


def test_renderer_summarizes_apply_patch_details() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(
        ToolExecutionStartEvent(
            tool_call_id="tc1",
            tool_name="apply_patch",
            args={"dry_run": True},
        )
    )
    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="apply_patch",
            result=tool_result(
                "--- a.txt\n+++ a.txt\n-old\n+new\n",
                details={"dry_run": True, "files_changed": 1, "lines_added": 1, "lines_removed": 1},
            ),
        )
    )

    rendered = output.getvalue()
    assert "Previewed patch · 1 file · +1 −1 lines" in rendered
    assert "--- a.txt" not in rendered


def test_renderer_compacts_consecutive_same_exploration_tools() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    for index, path in enumerate([".", "agent", "cli"], start=1):
        renderer.render_event(
            ToolExecutionStartEvent(
                tool_call_id=f"tc{index}",
                tool_name="list_dir",
                args={"path": path},
            )
        )
        renderer.render_event(
            ToolExecutionEndEvent(
                tool_call_id=f"tc{index}",
                tool_name="list_dir",
                result=tool_result("a\nb\nc\n"),
            )
        )
    renderer.render_event(TextDeltaEvent("done"))

    rendered = output.getvalue()
    assert "List dir: ." not in rendered
    assert "List dir: agent" not in rendered
    assert "List dir: cli · 3 tool calls" in rendered
    assert "Listed 3 entries · 3 list_dir calls" in rendered


def test_renderer_overwrites_live_consecutive_same_exploration_tools() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=100)
    renderer = CliRenderer(console)

    for index, path in enumerate([".", "agent", "cli"], start=1):
        renderer.render_event(
            ToolExecutionStartEvent(
                tool_call_id=f"tc{index}",
                tool_name="list_dir",
                args={"path": path},
            )
        )
        renderer.render_event(
            ToolExecutionEndEvent(
                tool_call_id=f"tc{index}",
                tool_name="list_dir",
                result=tool_result("a\nb\nc\n"),
            )
        )
    renderer.render_event(TurnEndEvent(text=""))

    rendered = output.getvalue()
    assert "\x1b[1A" in rendered
    assert "List dir: cli · 3 tool calls" in rendered
    assert "Listed 3 entries · 3 list_dir calls" in rendered


def test_renderer_does_not_compact_same_tool_across_other_tools() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="list_dir", args={"path": "agent"}))
    renderer.render_event(ToolExecutionEndEvent(tool_call_id="tc1", tool_name="list_dir", result=tool_result("a\n")))
    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc2", tool_name="read_file", args={"path": "agent/session.py"}))
    renderer.render_event(ToolExecutionEndEvent(tool_call_id="tc2", tool_name="read_file", result=tool_result("line\n")))
    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc3", tool_name="list_dir", args={"path": "tools"}))
    renderer.render_event(ToolExecutionEndEvent(tool_call_id="tc3", tool_name="list_dir", result=tool_result("b\n")))
    renderer.render_event(TurnEndEvent(text=""))

    rendered = output.getvalue()
    assert "List dir: agent" in rendered
    assert "Read file: agent/session.py" in rendered
    assert "List dir: tools" in rendered
    assert "list_dir calls" not in rendered


def test_renderer_summarizes_update_todo_with_items() -> None:
    console, output = _console()
    renderer = CliRenderer(console)
    payload = {
        "todos": [
            {"id": "1", "content": "Inspect current implementation", "status": "completed"},
            {"id": "2", "content": "Wire renderer summary", "status": "in_progress"},
            {"id": "3", "content": "Run tests", "status": "pending"},
        ],
        "summary": {"total": 3, "pending": 1, "in_progress": 1, "completed": 1, "cancelled": 0},
    }

    renderer.render_event(
        ToolExecutionStartEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            args={"todos": payload["todos"], "merge": False},
        )
    )
    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            result=tool_result("", details=payload),
        )
    )

    rendered = output.getvalue()
    assert "● todo 1/3 · 1 running · 1 pending · Wire renderer summary" in rendered
    assert "Inspect current implementation" not in rendered
    assert "Run tests" not in rendered
    assert "Update todo" not in rendered
    assert "1 lines" not in rendered


def test_renderer_overwrites_live_todo_status() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=100)
    renderer = CliRenderer(console)
    first = {
        "todos": [
            {"id": "1", "content": "Inspect current implementation", "status": "in_progress"},
            {"id": "2", "content": "Run tests", "status": "pending"},
        ],
        "summary": {"total": 2, "pending": 1, "in_progress": 1, "completed": 0, "cancelled": 0},
    }
    second = {
        "todos": [
            {"id": "1", "content": "Inspect current implementation", "status": "completed"},
            {"id": "2", "content": "Run tests", "status": "in_progress"},
        ],
        "summary": {"total": 2, "pending": 0, "in_progress": 1, "completed": 1, "cancelled": 0},
    }

    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="update_todo", args={}))
    renderer.render_event(ToolExecutionEndEvent(tool_call_id="tc1", tool_name="update_todo", result=tool_result("", details=first)))
    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc2", tool_name="update_todo", args={}))
    renderer.render_event(ToolExecutionEndEvent(tool_call_id="tc2", tool_name="update_todo", result=tool_result("", details=second)))

    rendered = output.getvalue()
    assert "\x1b[1A" in rendered
    assert "● todo 0/2 · 1 running · 1 pending · Inspect current implementation" in rendered
    assert "● todo 1/2 · 1 running · Run tests" in rendered


def test_renderer_does_not_replay_completed_todo_after_followup_text() -> None:
    console, output = _console()
    renderer = CliRenderer(console)
    payload = {
        "todos": [
            {"id": "1", "content": "Inspect current implementation", "status": "completed"},
            {"id": "2", "content": "Write summary", "status": "completed"},
        ],
        "summary": {"total": 2, "pending": 0, "in_progress": 0, "completed": 2, "cancelled": 0},
    }

    renderer.render_event(TurnStartEvent(prompt="analyze"))
    renderer.render_event(AssistantMessageEvent(text="", tool_calls=[]))
    renderer.render_event(
        ToolExecutionStartEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            args={"todos": [payload["todos"][1]], "merge": True},
        )
    )
    renderer.render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            result=tool_result("", details=payload),
        )
    )
    renderer.render_event(TextDeltaEvent("## Result\n\nLong final answer."))
    renderer.render_event(TurnEndEvent(text=""))

    rendered = output.getvalue()
    assert "✓ todo completed · 2/2" not in rendered
    assert "Long final answer." in rendered


def test_renderer_summarizes_bash_start_with_truncated_command() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="bash", args={"command": "x" * 1000}))

    rendered = output.getvalue()
    assert "Run:" in rendered
    # full command must not be dumped, the summary trims with an ellipsis
    assert "x" * 1000 not in rendered
    assert "…" in rendered


def test_renderer_error_summary_prefers_traceback_exception_line() -> None:
    result = tool_error(
        "Traceback (most recent call last):\n"
        '  File "<string>", line 12, in <module>\n'
        "TypeError: 'int' object is not subscriptable\n"
    )

    assert render_module._format_error_summary(result) == "✗ TypeError: 'int' object is not subscriptable"


def test_renderer_shows_file_state_guard_as_guidance() -> None:
    result = tool_error(
        "read_file must be called first before overwriting /tmp/demo.py.",
        {"presentation": "guidance", "category": "file_state_guard"},
    )

    assert render_module._format_error_summary(result) == "! Read file first before overwriting /tmp/demo.py"
    assert render_module._tool_result_style(result) == "yellow"


def test_renderer_treats_full_read_guard_as_guidance() -> None:
    result = tool_error(
        "Full read_file must be called before editing /tmp/demo.py; the last read was partial. "
        "Read the remaining file range or call read_file with a larger limit until partial=false."
    )

    assert render_module._format_error_summary(result) == "! Read the full file before editing /tmp/demo.py"
    assert render_module._tool_result_style(result) == "yellow"


def test_renderer_write_file_start_shows_path_without_content() -> None:
    console, output = _console()
    renderer = CliRenderer(console)

    renderer.render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="write_file", args={"path": "test.md", "content": "x" * 1000}))

    rendered = output.getvalue()
    assert "Write file: test.md" in rendered
    # content body must never reach the terminal
    assert "x" * 1000 not in rendered


def test_file_confirmation_question_mentions_target_path() -> None:
    assert render_module._confirmation_question("write_file", {"path": "test.md"}) == "Do you want to write test.md?"
    assert render_module._confirmation_question("edit_file", {"path": "story.md"}) == "Do you want to edit story.md?"


def test_bash_confirmation_warns_for_file_modifying_commands() -> None:
    question = render_module._confirmation_question(
        "bash",
        {"command": "python3 -c \"from pathlib import Path; Path('x').write_text('y')\""},
    )

    assert "may modify files" in question
    assert "prefer edit_file or apply_patch" in question


def test_allow_all_file_edits_skips_later_file_confirmations(monkeypatch) -> None:
    console, _ = _console()
    renderer = CliRenderer(console)
    selections: list[tuple[str, bool]] = []

    async def fake_select(question: str, *, allow_all_edits: bool = False, allow_read_only_bash: bool = False):
        selections.append((question, allow_all_edits, allow_read_only_bash))
        return "allow_all_edits"

    monkeypatch.setattr(render_module, "_select_confirmation", fake_select)

    assert asyncio.run(renderer.confirm("write_file", {"path": "test.md"})) is True
    assert asyncio.run(renderer.confirm("edit_file", {"path": "story.md"})) is True
    assert selections == [("Do you want to write test.md?", True, False)]


def test_allow_read_only_bash_skips_later_safe_shell_confirmations(monkeypatch) -> None:
    console, _ = _console()
    renderer = CliRenderer(console)
    selections: list[tuple[str, bool, bool]] = []

    async def fake_select(question: str, *, allow_all_edits: bool = False, allow_read_only_bash: bool = False):
        selections.append((question, allow_all_edits, allow_read_only_bash))
        return "allow_read_only_bash"

    monkeypatch.setattr(render_module, "_select_confirmation", fake_select)

    assert asyncio.run(renderer.confirm("bash", {"command": "grep -n test tests/test_sessions.py"})) is True
    assert asyncio.run(renderer.confirm("bash", {"command": "sed -n '1,20p' tests/test_sessions.py | grep test"})) is True
    assert selections == [("Do you want to run this command?", False, True)]


def test_read_only_bash_detection_is_conservative() -> None:
    assert render_module._is_read_only_bash_command({"command": "rg todo agent tools"})
    assert render_module._is_read_only_bash_command({"command": "sed -n '1,20p' tests/test_sessions.py"})
    assert render_module._is_read_only_bash_command({"command": 'grep -n "a\\|b" tests/test_sessions.py'})
    assert render_module._is_read_only_bash_command({"command": "grep x file.txt | head"})
    assert render_module._is_read_only_bash_command({"command": "echo title && grep -n test tests/test_sessions.py | sort | uniq -c"})
    assert render_module._is_read_only_bash_command(
        {"command": f"cd {Path.cwd()} && echo title && sed -n '1,20p' tests/spice/cli/test_render.py | grep '^def'"}
    )
    assert render_module._is_read_only_bash_command({"command": "awk '{print $2}' tests/test_sessions.py | sort -n"})
    assert not render_module._is_read_only_bash_command({"command": "sed -i 's/a/b/' file.txt"})
    assert not render_module._is_read_only_bash_command({"command": "echo x > file.txt"})
    assert not render_module._is_read_only_bash_command({"command": "grep x file.txt || true"})
    assert not render_module._is_read_only_bash_command({"command": "awk '{print $2 > \"out\"}' tests/test_sessions.py"})
    assert not render_module._is_read_only_bash_command({"command": "uv run pytest"})
    assert not render_module._is_read_only_bash_command({"command": "cat /etc/passwd"})
    assert not render_module._is_read_only_bash_command({"command": "cat ../outside.txt"})
