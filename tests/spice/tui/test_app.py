from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from spice.agent.events import AssistantMessageEvent, TextDeltaEvent, ToolExecutionEndEvent, ToolExecutionStartEvent, TurnStartEvent
from spice.llm.messages import ToolCall
from spice.tools.base import tool_result
from spice.tui.app import SpiceTUI, _InputBufferControl, _MessageBufferControl
from spice.cli.terminal import enable_cursor_blink_after_render


def _minimal_tui() -> SpiceTUI:
    tui = object.__new__(SpiceTUI)
    tui._message_buffer = Buffer(read_only=True)
    tui._message_follow_tail = True
    tui._message_vertical_scroll = 0
    tui._message_window = SimpleNamespace(render_info=SimpleNamespace(window_height=24, window_width=100))
    tui._response_start = 0
    tui._response_parts = []
    tui._streaming = False
    tui._waiting_started = None
    tui._waiting_start = None
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
    tui._render_event(ToolExecutionStartEvent(tool_call_id="tc1", tool_name="update_todo", args={"merge": True, "todos": []}))
    tui._render_event(
        ToolExecutionEndEvent(
            tool_call_id="tc1",
            tool_name="update_todo",
            result=tool_result(
                "",
                details={
                    "todos": [{"id": "1", "content": "收尾", "status": "completed"}],
                    "summary": {"total": 1, "completed": 1, "in_progress": 0, "pending": 0, "cancelled": 0},
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


def test_tui_renders_thought_label_between_user_and_answer() -> None:
    tui = _minimal_tui()

    tui._render_event(TurnStartEvent(prompt="hello"))
    assert tui._message_buffer.text == "Thought for 0s\n\n"
    assert tui._waiting_started is not None

    tui._render_event(TextDeltaEvent("你好"))
    tui._render_event(AssistantMessageEvent(text="你好", tool_calls=[]))

    assert tui._message_buffer.text == "Thought for 0s\n\nSpice: 你好\n"
    assert "thinking" not in tui._message_buffer.text
    assert tui._waiting_started is None


def test_tui_renders_final_text_when_no_delta_was_streamed() -> None:
    tui = _minimal_tui()

    tui._render_event(TurnStartEvent(prompt="hello"))
    tui._render_event(AssistantMessageEvent(text="final answer", tool_calls=[]))

    assert tui._message_buffer.text == "Thought for 0s\n\nSpice: final answer\n"


def test_tui_turn_gap_keeps_rounds_from_touching() -> None:
    tui = _minimal_tui()
    tui._append("Spice: previous answer\n")

    tui._ensure_turn_gap()
    tui._append("You: next question\n")

    assert tui._message_buffer.text == "Spice: previous answer\n\nYou: next question\n"


def test_tui_markdown_lists_are_compact() -> None:
    tui = _minimal_tui()

    rendered = tui._render_markdown_text("- first\n\n- second\n\n- third")

    assert rendered == " \u2022 first\n \u2022 second\n \u2022 third\n"


def test_tui_markdown_tables_render_as_tables() -> None:
    tui = _minimal_tui()

    rendered = tui._render_markdown_text("| 项目 | 详情 |\n| --- | --- |\n| 天气 | 多云 |\n")

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


def test_tui_scrolls_message_history_off_tail() -> None:
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(render_info=SimpleNamespace(window_height=3, window_width=80))
    tui._append("\n".join(f"line {index}" for index in range(10)))

    assert tui._message_vertical_scroll == 7

    tui._scroll_messages(-3)

    assert tui._message_vertical_scroll == 4
    assert tui._message_follow_tail is False


def test_tui_mouse_wheel_scrolls_message_history_from_message_and_input_controls() -> None:
    tui = _minimal_tui()
    tui._message_window = SimpleNamespace(render_info=SimpleNamespace(window_height=3, window_width=80))
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


def test_tui_message_history_does_not_take_focus_from_input() -> None:
    tui = _minimal_tui()
    message_control = _MessageBufferControl(tui, buffer=tui._message_buffer, focusable=False)
    input_control = _InputBufferControl(tui, buffer=Buffer(), focus_on_click=True)

    assert message_control.is_focusable() is False
    assert input_control.is_focusable() is True
    assert input_control.focus_on_click() is True
