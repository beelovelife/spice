from __future__ import annotations

import asyncio
from types import SimpleNamespace

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from spice.agent.events import (
    AgentErrorEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnStartEvent,
)
from spice.llm.messages import ToolCall
from spice.tools.base import tool_result
from spice.tui.app import SpiceTUI, _InputBufferControl, _MessageBufferControl
from spice.cli.terminal import enable_cursor_blink_after_render


def _minimal_tui() -> SpiceTUI:
    tui = object.__new__(SpiceTUI)
    tui._message_buffer = Buffer(read_only=True)
    tui._message_follow_tail = True
    tui._message_vertical_scroll = 0
    tui._message_window = SimpleNamespace(
        render_info=SimpleNamespace(window_height=24, window_width=100)
    )
    tui._response_start = 0
    tui._response_parts = []
    tui._streaming = False
    tui._waiting_started = None
    tui._waiting_start = None
    tui._waiting_label = "Waiting for model"
    tui._reasoning_streaming = False
    tui._pending_tool_args = {}
    tui._compact_tool_group = None
    tui._todo_items = []
    tui._todo_summary = {}
    tui._app = SimpleNamespace(invalidate=lambda: None)
    return tui


def test_tui_reenables_cursor_blink_after_every_render() -> None:
    tui = SpiceTUI()

    assert enable_cursor_blink_after_render in tui._app.after_render._handlers


def test_tui_keeps_tool_output_between_assistant_segments() -> None:
    tui = _minimal_tui()

    tui._render_event(TurnStartEvent(prompt="analyze"))
    tui._render_event(TextDeltaEvent("第一段分析"))
    tui._render_event(
        AssistantMessageEvent(
            text="第一段分析",
            tool_calls=[ToolCall(id="tc1", name="update_todo", arguments={})],
        )
    )
    tui._render_event(
        ToolExecutionStartEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            args={"merge": True, "todos": []},
        )
    )
    tui._render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            result=tool_result(
                "",
                details={
                    "todos": [{"id": "1", "content": "收尾", "status": "completed"}],
                    "summary": {
                        "total": 1,
                        "completed": 1,
                        "in_progress": 0,
                        "pending": 0,
                        "cancelled": 0,
                    },
                },
            ),
        )
    )
    tui._render_event(TextDeltaEvent("第二段总结"))
    tui._render_event(AssistantMessageEvent(text="第二段总结", tool_calls=[]))

    rendered = tui._message_buffer.text
    assert "第一段分析" in rendered
    assert "Update todo" not in rendered
    assert "todo updated" not in rendered
    assert "第二段总结" in rendered

    todo_text = "".join(fragment[1] for fragment in tui._todo_fragments())
    assert "Todo" in todo_text
    assert "✓ 1. 收尾" in todo_text


def test_tui_replaces_waiting_activity_with_answer() -> None:
    tui = _minimal_tui()

    tui._render_event(TurnStartEvent(prompt="hello"))
    assert tui._message_buffer.text == "⠋ Waiting for model · 0s\n"
    assert tui._waiting_started is not None

    tui._render_event(TextDeltaEvent("你好"))
    tui._render_event(AssistantMessageEvent(text="你好", tool_calls=[]))

    assert tui._message_buffer.text == "Spice: 你好\n"


def test_tui_streams_reasoning_before_answer() -> None:
    tui = _minimal_tui()

    tui._render_event(TurnStartEvent(prompt="hello"))
    tui._render_event(ReasoningDeltaEvent("先检查", kind="reasoning"))
    tui._render_event(ReasoningDeltaEvent("上下文", kind="reasoning"))
    tui._render_event(TextDeltaEvent("结论"))
    tui._render_event(AssistantMessageEvent(text="结论", tool_calls=[]))

    assert tui._message_buffer.text == "Thinking: 先检查上下文\nSpice: 结论\n"


def test_tui_shows_retry_fallback_and_fatal_tool_stop() -> None:
    tui = _minimal_tui()

    tui._render_event(ModelRetryEvent("openai", "primary", 1, 2, 3, 0.5, "temporary"))
    tui._render_event(
        ModelFallbackEvent(
            "primary",
            "openai",
            "primary",
            "backup",
            "anthropic",
            "backup",
            "server",
            0,
            1,
        )
    )
    tui._render_event(AgentErrorEvent("Tool denied by user: bash", kind="fatal_tool"))

    rendered = tui._message_buffer.text
    assert "Retrying 2/3 in 0.5s" in rendered
    assert "openai/primary -> anthropic/backup" in rendered
    assert "Current turn stopped: Tool denied by user: bash" in rendered
    assert "thinking" not in tui._message_buffer.text
    assert tui._waiting_started is None


def test_tui_renders_final_text_when_no_delta_was_streamed() -> None:
    tui = _minimal_tui()

    tui._render_event(TurnStartEvent(prompt="hello"))
    tui._render_event(AssistantMessageEvent(text="final answer", tool_calls=[]))

    assert tui._message_buffer.text == "Spice: final answer\n"


def test_tui_turn_gap_keeps_rounds_from_touching() -> None:
    tui = _minimal_tui()
    tui._append("Spice: previous answer\n")

    tui._ensure_turn_gap()
    tui._append("You: next question\n")

    assert tui._message_buffer.text == "Spice: previous answer\n\nYou: next question\n"


def test_tui_activity_status_distinguishes_generation_pause_and_tool(
    monkeypatch,
) -> None:
    tui = _minimal_tui()
    tui._agent_session = None
    tui._busy = True
    tui._activity_started = 100.0
    tui._streaming = True
    tui._last_stream_delta_at = 104.5
    tui._active_tool_name = None
    monkeypatch.setattr("spice.tui.app.time.monotonic", lambda: 105.0)

    assert "".join(text for _, text in tui._mode_fragments()) == " ⠋ Generating · 5s"

    tui._last_stream_delta_at = 102.0
    assert (
        "".join(text for _, text in tui._mode_fragments())
        == " ⠋ Waiting for model · 5s"
    )

    tui._active_tool_name = "web_search"
    assert (
        "".join(text for _, text in tui._mode_fragments())
        == " ⠋ Running web_search · 5s"
    )


def test_tui_styles_role_labels_and_tool_rows_without_changing_text() -> None:
    tui = _minimal_tui()
    original = (
        "You: question\n\nSpice: answer\n  tool: web_search()\n  web_search -> done"
    )
    tui._append(original)

    fragments = tui._message_fragments()

    assert "".join(text for _, text in fragments) == original
    assert ("class:message.user-label", "You:") in fragments
    assert ("class:message.assistant-label", "Spice:") in fragments
    assert any(
        style == "class:message.tool" and "tool: web_search" in text
        for style, text in fragments
    )
    assert any(
        style == "class:message.tool-result" and "web_search -> done" in text
        for style, text in fragments
    )


def test_tui_markdown_lists_are_compact() -> None:
    tui = _minimal_tui()

    rendered = tui._render_markdown_text("- first\n\n- second\n\n- third")

    assert rendered == " \u2022 first\n \u2022 second\n \u2022 third\n"


def test_tui_markdown_tables_render_as_tables() -> None:
    tui = _minimal_tui()

    rendered = tui._render_markdown_text(
        "| 项目 | 详情 |\n| --- | --- |\n| 天气 | 多云 |\n"
    )

    assert "项目" in rendered
    assert "详情" in rendered
    assert "天气" in rendered
    assert "多云" in rendered
    assert "╭" in rendered
    assert "│" in rendered


def test_tui_markdown_heading_gets_breathing_room() -> None:
    tui = _minimal_tui()

    rendered = tui._render_markdown_text("## 标题\n正文")

    assert rendered == "标题\n\n正文\n"


def test_tui_compact_uses_agent_session_engine() -> None:
    async def exercise() -> None:
        tui = _minimal_tui()
        calls = []

        async def compact(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(tokens_before=100, tokens_after=40)

        tui._agent_session = SimpleNamespace(
            session=SimpleNamespace(id="session-1"), compact=compact
        )
        await tui._show_compact("focus on tests")

        assert calls == [{"focus": "focus on tests", "reason": "manual", "force": True}]
        assert "~100 -> ~40 tokens" in tui._message_buffer.text

    asyncio.run(exercise())


def test_tui_scrolls_message_history_off_tail() -> None:
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(
        render_info=SimpleNamespace(window_height=3, window_width=80)
    )
    tui._append("\n".join(f"line {index}" for index in range(10)))

    assert tui._message_vertical_scroll == 7

    tui._scroll_messages(-3)

    assert tui._message_vertical_scroll == 4
    assert tui._message_follow_tail is False


def test_tui_scrolls_wrapped_message_history_off_tail() -> None:
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(
        render_info=SimpleNamespace(window_height=3, window_width=11)
    )
    tui._append("0123456789" * 5)

    assert tui._message_display_lines() == ["0123456789"] * 5
    assert tui._message_vertical_scroll == 2

    tui._scroll_messages(-1)

    assert tui._message_vertical_scroll == 1
    assert tui._message_follow_tail is False


def test_tui_rendered_window_keeps_tail_and_manual_scroll_visible() -> None:
    tui = SpiceTUI()
    tui._append("\n".join(f"line {index}" for index in range(20)))

    content = tui._message_control.create_content(width=20, height=5)
    tui._message_window._scroll(content, width=20, height=5)

    assert tui._message_vertical_scroll == 15
    assert tui._message_window.vertical_scroll == 15

    tui._scroll_messages(-3)
    content = tui._message_control.create_content(width=20, height=5)
    tui._message_window._scroll(content, width=20, height=5)

    assert tui._message_vertical_scroll == 12
    assert tui._message_window.vertical_scroll == 12


def test_tui_message_content_supports_preferred_height_probe() -> None:
    tui = SpiceTUI()
    tui._append("history\n" * 20)

    content = tui._message_control.create_content(width=80, height=None)
    preferred_height = tui._message_control.preferred_height(80, 100, False, None)

    assert content.line_count == 21
    assert preferred_height == 21
    assert tui._message_cursor_position().y == 20


def test_tui_mouse_wheel_scrolls_message_history_from_message_and_input_controls() -> (
    None
):
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(
        render_info=SimpleNamespace(window_height=3, window_width=80)
    )
    tui._append("\n".join(f"line {index}" for index in range(10)))
    invalidated = []
    tui._app = SimpleNamespace(invalidate=lambda: invalidated.append(True))
    event = MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.SCROLL_UP,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )

    message_control = _MessageBufferControl(tui, buffer=tui._message_buffer)
    input_control = _InputBufferControl(tui, buffer=Buffer())

    assert message_control.mouse_handler(event) is None
    assert tui._message_vertical_scroll == 4
    assert input_control.mouse_handler(event) is None
    assert tui._message_vertical_scroll == 1
    assert len(invalidated) == 2


def test_tui_mouse_wheel_moves_session_selector_and_keeps_choice_visible() -> None:
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(
        render_info=SimpleNamespace(window_height=10, window_width=80)
    )
    tui._session_choices = [f"session-{index}" for index in range(8)]
    tui._session_choice_rows = [
        (session_id, "provider/model", f"details {index}")
        for index, session_id in enumerate(tui._session_choices)
    ]
    tui._session_choice_index = 0
    tui._session_choice_start = 0
    tui._session_choice_mode = "resume"
    tui._agent_session = None
    tui._render_session_selector()
    event = MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.SCROLL_DOWN,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )

    control = _InputBufferControl(tui, buffer=Buffer())
    assert control.mouse_handler(event) is None

    assert tui._session_choice_index == 1
    assert "❯   session-1" in tui._message_buffer.text
    assert "session-7" not in tui._message_buffer.text


def test_tui_resume_replaces_old_view_follows_latest_and_remains_scrollable(
    monkeypatch,
) -> None:
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(
        render_info=SimpleNamespace(window_height=5, window_width=21)
    )
    tui._provider = None
    tui._model = None
    tui._refresh_completer = lambda: None
    tui._append("old conversation\n" * 10)
    tui._scroll_messages(-3)
    assert tui._message_follow_tail is False

    resumed = SimpleNamespace(
        session_label="session-123",
        runtime_model_label="provider/model",
        messages=[
            SimpleNamespace(role="user", content="question", tool_calls=None),
            SimpleNamespace(
                role="assistant", content="latest answer " * 20, tool_calls=None
            ),
        ],
    )
    monkeypatch.setattr("spice.tui.app.AgentSession", lambda **_kwargs: resumed)

    tui._resume_session("session-123")

    assert "old conversation" not in tui._message_buffer.text
    assert "latest answer" in tui._message_buffer.text
    assert tui._message_follow_tail is True

    assert tui._message_vertical_scroll == tui._message_bottom_scroll()
    assert tui._message_vertical_scroll > 0

    bottom = tui._message_vertical_scroll
    tui._scroll_messages(-3)
    assert tui._message_vertical_scroll == bottom - 3
    assert tui._message_follow_tail is False
    tui._scroll_messages(3)
    assert tui._message_vertical_scroll == bottom
    assert tui._message_follow_tail is True


def test_shared_resume_replaces_view_replays_history_and_resets_policy(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        tui = SpiceTUI()
        tui._append("old conversation\n")
        tui.confirm_policy.allow_file_edits = True
        tui.confirm_policy.allow_all_tools = True
        closed = []

        class OldSession:
            extensions = None
            model = SimpleNamespace(provider="openai", id="old")

            async def aclose(self) -> None:
                closed.append(True)

        new_session = SimpleNamespace(
            session_id="session-new",
            session_label="session-new",
            runtime_model_label="openai/new",
            messages=[
                SimpleNamespace(
                    role="user", content="restored question", tool_calls=None
                ),
                SimpleNamespace(
                    role="assistant", content="restored answer", tool_calls=None
                ),
            ],
            confirm=None,
        )

        monkeypatch.setattr(
            "spice.interactive.commands.replace_session",
            lambda *_args, **_kwargs: new_session,
        )
        tui._agent_session = OldSession()
        tui._bind_trace_session = lambda: None
        tui._refresh_completer = lambda: None

        await tui._handle_slash_command("/resume session-new")

        assert "old conversation" not in tui._message_buffer.text
        assert "restored question" in tui._message_buffer.text
        assert "restored answer" in tui._message_buffer.text
        assert tui.confirm_policy.allow_file_edits is False
        assert tui.confirm_policy.allow_all_tools is False
        assert closed == [True]

    asyncio.run(exercise())


def test_shared_reset_clears_old_view_before_painting_result() -> None:
    async def exercise() -> None:
        tui = SpiceTUI()
        tui._append("old conversation\n")

        class Session:
            session = object()
            session_id = "session-1"
            extensions = None

            def reset(self) -> None:
                return None

        async def confirm(_request) -> str:
            return "yes"

        tui._agent_session = Session()
        tui._port_confirm = confirm

        await tui._handle_slash_command("/reset")

        assert "old conversation" not in tui._message_buffer.text
        assert "Reset session: session-1" in tui._message_buffer.text

    asyncio.run(exercise())


def test_tui_submitting_message_leaves_history_mode_and_follows_response() -> None:
    async def exercise() -> None:
        tui = _minimal_tui()
        tui._message_window = SimpleNamespace(
            render_info=SimpleNamespace(window_height=4, window_width=30)
        )
        tui._busy = False
        tui._show_edit_mode_hint = False
        tui._agent_session = SimpleNamespace(
            plan_state=SimpleNamespace(mode="edit", is_plan_mode=False),
        )
        tui._append("old line\n" * 20)
        tui._scroll_messages(-6)
        assert tui._message_follow_tail is False

        prompts = []

        async def run_prompt(message: str) -> None:
            prompts.append(message)

        tui._run_prompt = run_prompt
        input_buffer = Buffer()
        input_buffer.text = "new question"

        assert tui._on_accept(input_buffer) is True
        await tui._pending_task

        assert prompts == ["new question"]
        assert tui._message_follow_tail is True
        assert tui._message_vertical_scroll == tui._message_bottom_scroll()
        assert tui._message_buffer.text.endswith("You: new question\n\n")

        tui._render_event(TurnStartEvent(prompt="new question"))
        tui._render_event(TextDeltaEvent("visible response"))
        assert tui._message_follow_tail is True
        assert tui._message_vertical_scroll == tui._message_bottom_scroll()
        assert tui._message_buffer.text.endswith("Spice: visible response")

    asyncio.run(exercise())


def test_tui_message_history_does_not_take_focus_from_input() -> None:
    tui = _minimal_tui()
    message_control = _MessageBufferControl(
        tui, buffer=tui._message_buffer, focusable=False
    )
    input_control = _InputBufferControl(tui, buffer=Buffer(), focus_on_click=True)

    assert message_control.is_focusable() is False
    assert input_control.is_focusable() is True
    assert input_control.focus_on_click() is True
