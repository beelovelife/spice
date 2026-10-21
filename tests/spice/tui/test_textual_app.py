from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets.markdown import MarkdownBlock

from spice.agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    TurnFinalizationStartEvent,
)
from spice.agent.turn_outcome import TurnCheckpoint
from spice.agent.session_history import AGENT_ERROR_ENTRY
from spice.agent.sessions import SessionStore
from spice.interactive.types import CommandResult
from spice.llm.config import SpiceConfig
from spice.llm.models import Model
from spice.llm.messages import Message
from spice.tools.base import tool_result
from spice.tui.screens import ApprovalScreen
from spice.tui.textual_app import SpiceTextualApp
from spice.tui.widgets import (
    AssistantMessage,
    CommandSuggestions,
    Composer,
    NoticeMessage,
    ReasoningMessage,
    ToolMessage,
    UserMessage,
)


class _FakeSession:
    cwd = Path.cwd()
    runtime_model_label = "test/model"
    model = SimpleNamespace(protocol="openai-responses")
    messages: list = []
    session_label = "new"
    extensions = None
    plan_state = SimpleNamespace(is_plan_mode=False, steps=[])

    def __init__(self, events) -> None:
        self.events = list(events)
        self.prompts: list[str] = []
        self.confirm = None
        self.closed = False

    def set_confirm(self, confirm) -> None:
        self.confirm = confirm

    async def prompt(self, _text: str):
        self.prompts.append(_text)
        for event in self.events:
            yield event

    async def aclose(self) -> None:
        self.closed = True


async def _wait_until_idle(app: SpiceTextualApp, pilot) -> None:
    for _ in range(40):
        await pilot.pause()
        if not app._busy:
            return
    raise AssertionError("Textual app did not finish the fake turn")


def test_textual_tui_restores_errors_on_startup_and_session_replacement(tmp_path):
    store = SessionStore(tmp_path / "sessions", cwd=tmp_path)
    info = store.create(cwd=tmp_path, provider="fake", model="fake")
    store.append_message(info.id, Message(role="user", content="first request"))
    store.append_custom(info.id, {"customType": AGENT_ERROR_ENTRY, "message": "request too large", "kind": "error"})
    store.append_message(info.id, Message(role="user", content="continue"))

    def restored_session():
        session = _FakeSession([])
        session.session_store = store
        session.session = info
        session.messages = store.load_messages(info.id)
        return session

    def check(app):
        children = list(app.query_one("#messages").children)
        assert [type(child) for child in children[:3]] == [UserMessage, NoticeMessage, UserMessage]
        assert children[1].render().plain == "历史错误：request too large"
        assert children[1].has_class("notice-error")
        assert len([child for child in children if child.has_class("notice-error")]) == 1

    async def run():
        app = SpiceTextualApp(session=restored_session())
        async with app.run_test(size=(100, 32)):
            check(app)
            await app._apply_command_result(CommandResult(session=restored_session(), replay_session_history=True))
            check(app)

    asyncio.run(run())


def test_textual_tui_separates_budget_summary_from_the_previous_round():
    async def run():
        checkpoint = TurnCheckpoint("task", 100, 100, summary="已保存进度，剩余验收。")
        session = _FakeSession([
            TurnStartEvent("work"), TextDeltaEvent("正在修改代码"),
            AssistantMessageEvent("正在修改代码", []),
            TurnFinalizationStartEvent(100, 100), TextDeltaEvent(checkpoint.summary),
            AssistantMessageEvent(checkpoint.summary, []),
            TurnEndEvent(checkpoint.summary, "max_tool_rounds", checkpoint=checkpoint),
        ])
        app = SpiceTextualApp(session=session)
        async with app.run_test(size=(100, 32)) as pilot:
            await app._dispatch_input("work")
            await _wait_until_idle(app, pilot)
            assistants = list(app.query(AssistantMessage))
            assert len(assistants) == 2
            assert assistants[-1].text == checkpoint.summary
            notices = [notice.render().plain for notice in app.query(NoticeMessage)]
            assert any("暂停待继续" in text for text in notices)
            assert not any("Error:" in text for text in notices)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("trace", "explicit_path"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_trace_follows_replacement_session_without_duplicate_events(
    tmp_path, monkeypatch, trace, explicit_path
) -> None:
    monkeypatch.setattr("spice.agent.trace.TRACE_DIR", tmp_path / "traces")

    class TraceSession(_FakeSession):
        def __init__(self, session_id: str) -> None:
            super().__init__([])
            self.cwd = tmp_path
            self.model = Model(id="test", provider="openai")
            self.config = SpiceConfig()
            self.session = SimpleNamespace(id=session_id)
            self.session_label = session_id
            self.messages = []
            self.subagents_enabled = False
            self.listeners = []

        def get_active_tools(self):
            return []

        def subscribe(self, listener):
            self.listeners.append(listener)
            return lambda: self.listeners.remove(listener)

        def emit(self, event):
            for listener in list(self.listeners):
                listener(event)

    async def run() -> None:
        first, second = TraceSession("first"), TraceSession("second")
        app = SpiceTextualApp(
            session=first,
            trace=trace,
            trace_file=tmp_path / "chat.json" if explicit_path else None,
        )
        async with app.run_test(size=(100, 32)):
            first.emit(AgentStartEvent("first"))
            first.emit(TurnEndEvent("first answer"))
            await app._apply_command_result(CommandResult(session=second))
            await app._apply_command_result(CommandResult(session=second))
            assert first.closed is True
            assert first.listeners == []
            assert second.confirm == app.confirm_tool
            first.emit(AgentStartEvent("stale"))
            second.emit(AgentStartEvent("second"))
            second.emit(TurnEndEvent("second answer"))
            second.emit(AgentEndEvent("second"))

            if trace or explicit_path:
                assert len(second.listeners) == 1
                assert app._trace_path is not None
                data = json.loads(app._trace_path.read_text(encoding="utf-8"))
                assert data["session_id"] == "second"
                assert [
                    event["session_id"]
                    for event in data["events"]
                    if event["type"] == "agent_start"
                ] == ["first", "second"]
                assert [
                    event["text"]
                    for event in data["events"]
                    if event["type"] == "turn_end"
                ] == ["first answer", "second answer"]
            else:
                assert second.listeners == []
                assert app._trace_path is None
                assert not list(tmp_path.rglob("*.json"))

    asyncio.run(run())


@pytest.mark.parametrize(
    ("raw", "invoked"),
    [("/skill:demo 检查项目", True), ("/skill --show demo", False), ("/skill:missing", False)],
)
def test_skill_command_starts_only_requested_workflow(tmp_path, monkeypatch, raw, invoked) -> None:
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / ".spice" / "skills" / "demo" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: demo\ndescription: 测试技能\n---\n检查项目并报告结果。\n",
        encoding="utf-8",
    )

    async def run() -> None:
        session = _FakeSession([])
        app = SpiceTextualApp(session=session)
        async with app.run_test(size=(100, 32)) as pilot:
            policy = app.confirm_policy
            await app._dispatch_input(raw)
            await _wait_until_idle(app, pilot)
            assert app.agent_session is session
            assert app.confirm_policy is policy
            assert len(session.prompts) == int(invoked)
            if invoked:
                assert skill_file.read_text(encoding="utf-8") in session.prompts[0]
                assert session.prompts[0].endswith("检查项目")
            elif "--show" in raw:
                assert any(
                    "检查项目并报告结果。" in notice.render().plain
                    for notice in app.query(NoticeMessage)
                )

    asyncio.run(run())


def test_reasoning_stream_is_expanded_then_collapses_and_can_reopen() -> None:
    async def run() -> None:
        session = _FakeSession(
            [
                TurnStartEvent("hello"),
                ReasoningDeltaEvent("先检查上下文"),
                TextDeltaEvent("结论"),
                AssistantMessageEvent("结论", []),
                TurnEndEvent("结论"),
            ]
        )
        app = SpiceTextualApp(session=session)
        async with app.run_test(size=(100, 32)) as pilot:
            await app._dispatch_input("hello")
            await _wait_until_idle(app, pilot)

            reasoning = app.query(ReasoningMessage).first()
            assert reasoning.text == "先检查上下文"
            assert reasoning.collapsed is True
            assert "click to expand" in reasoning.title
            assert app._last_reasoning == "先检查上下文"

            clicked = await pilot.click(reasoning.query_one("CollapsibleTitle"))
            await pilot.pause()
            assert clicked is True
            assert reasoning.collapsed is False
            assert len(app.query(AssistantMessage)) == 1

    asyncio.run(run())


def test_chat_uses_styled_user_prompt_and_unprefixed_assistant() -> None:
    async def run() -> None:
        answer = "第一段回答。\n\n第二段应该恢复完整宽度。"
        session = _FakeSession(
            [
                TurnStartEvent("请检查这个足够长、需要自动换行的问题"),
                TextDeltaEvent(answer),
                AssistantMessageEvent(answer, []),
                TurnEndEvent(answer),
            ]
        )
        app = SpiceTextualApp(session=session)
        async with app.run_test(size=(54, 24)) as pilot:
            await app._dispatch_input("请检查这个足够长、需要自动换行的问题")
            await _wait_until_idle(app, pilot)

            user = app.query(UserMessage).first()
            assistant = app.query(AssistantMessage).first()
            content_blocks = [
                block
                for block in assistant.query(MarkdownBlock)
                if block._content.plain
            ]

            user_content = user.render()
            assert user_content.plain.startswith(
                ">\N{NO-BREAK SPACE}\N{NO-BREAK SPACE}请检查"
            )
            assert user_content.spans[0].style == "bold #67e8f9"
            assert content_blocks[0]._content.plain == "第一段回答。"
            assert content_blocks[1]._content.plain == "第二段应该恢复完整宽度。"
            assert content_blocks[0].region.x == assistant.content_region.x
            assert content_blocks[1].region.x == assistant.content_region.x
            assert user.styles.background == assistant.styles.background

    asyncio.run(run())


def test_empty_session_uses_shared_random_welcome(monkeypatch) -> None:
    monkeypatch.setattr(
        "spice.tui.textual_app.choose_welcome_quote",
        lambda: ("A fresh beginning.", "新的开始。"),
    )

    async def run() -> None:
        app = SpiceTextualApp(session=_FakeSession([]))
        async with app.run_test(size=(100, 32)):
            welcome = app.query_one(".notice-welcome", NoticeMessage)
            content = welcome.render()
            assert "A fresh beginning." in content.plain
            assert "新的开始。" in content.plain
            assert "Type / for commands." not in content.plain

    asyncio.run(run())


def test_tool_card_uses_icon_and_settles_in_place() -> None:
    async def run() -> None:
        session = _FakeSession(
            [
                TurnStartEvent("pwd"),
                ToolExecutionStartEvent("tc1", "bash", {"command": "pwd"}),
                ToolExecutionEndEvent(
                    "tc1", "bash", tool_result("ok", details={"returncode": 0})
                ),
                TurnEndEvent(""),
            ]
        )
        app = SpiceTextualApp(session=session)
        async with app.run_test(size=(100, 32)) as pilot:
            await app._dispatch_input("pwd")
            await _wait_until_idle(app, pilot)

            tools = list(app.query(ToolMessage))
            assert len(tools) == 1
            assert tools[0].tool_call_id == "tc1"
            assert tools[0].title.startswith("⌘ Run command · ✓ exit 0")
            assert tools[0].collapsed is True

    asyncio.run(run())


def test_busy_input_is_queued_instead_of_rejected() -> None:
    async def run() -> None:
        gate = asyncio.Event()

        class SlowSession(_FakeSession):
            async def prompt(self, text: str):
                yield TurnStartEvent(text)
                if text == "first":
                    await gate.wait()
                yield TextDeltaEvent(text)
                yield AssistantMessageEvent(text, [])
                yield TurnEndEvent(text)

        app = SpiceTextualApp(session=SlowSession([]))
        async with app.run_test(size=(100, 32)) as pilot:
            await app._dispatch_input("first")
            await pilot.pause()
            app._queued_messages.append("second")
            app._refresh_queue()
            assert list(app._queued_messages) == ["second"]
            assert app.query_one("#queue-strip").display is True

            gate.set()
            for _ in range(60):
                await pilot.pause()
                if not app._busy and not app._queued_messages:
                    break
            assert not app._queued_messages
            assert app.query_one("#queue-strip").display is False

    asyncio.run(run())


def test_slash_suggestions_come_from_shared_registry_and_tab_accepts() -> None:
    async def run() -> None:
        app = SpiceTextualApp(session=_FakeSession([]))
        async with app.run_test(size=(100, 32)) as pilot:
            composer = app.query_one(Composer)
            composer.insert("/thi")
            await pilot.pause()

            suggestions = app.query_one(CommandSuggestions)
            assert suggestions.display is True
            assert suggestions.highlighted_option is not None
            assert suggestions.highlighted_option.id == "/thinking"

            composer.action_complete()
            await pilot.pause()
            assert composer.text == "/thinking "
            assert suggestions.display is False

    asyncio.run(run())


def test_slash_suggestions_support_arrow_selection_and_enter() -> None:
    async def run() -> None:
        app = SpiceTextualApp(session=_FakeSession([]))
        async with app.run_test(size=(100, 32)) as pilot:
            composer = app.query_one(Composer)
            composer.insert("/")
            await pilot.pause()

            suggestions = app.query_one(CommandSuggestions)
            first_id = suggestions.highlighted_option.id
            await pilot.press("down")
            assert suggestions.highlighted_option.id != first_id
            selected_id = suggestions.highlighted_option.id

            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == f"{selected_id} "
            assert suggestions.display is False

    asyncio.run(run())


def test_empty_composer_opens_question_history_and_enter_inserts_selection() -> None:
    async def run() -> None:
        session = _FakeSession([])
        session.messages = [
            SimpleNamespace(role="user", content="first question"),
            SimpleNamespace(role="assistant", content="first answer"),
            SimpleNamespace(role="user", content="latest question"),
        ]
        app = SpiceTextualApp(session=session)
        async with app.run_test(size=(100, 32)) as pilot:
            composer = app.query_one(Composer)
            suggestions = app.query_one(CommandSuggestions)

            await pilot.press("up")
            await pilot.pause()
            assert suggestions.display is True
            assert suggestions.mode == "history"
            assert composer.text == ""
            highlight = suggestions.get_component_rich_style(
                "option-list--option-highlighted"
            )
            assert highlight.bgcolor is not None
            assert tuple(highlight.bgcolor.get_truecolor()) == (51, 65, 85)

            await pilot.press("up")
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == "first question"
            assert suggestions.display is False

    asyncio.run(run())


def test_approval_can_allow_all_tools_for_only_this_session() -> None:
    async def run() -> None:
        app = SpiceTextualApp(session=_FakeSession([]))
        async with app.run_test(size=(100, 32)) as pilot:
            worker = app.run_worker(
                app.confirm_tool("bash", {"command": "pwd"}),
                name="confirm-test",
            )
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, ApprovalScreen):
                    break
            approval = app.screen
            assert isinstance(approval, ApprovalScreen)
            approval.ready_at = 0
            approval.action_choose(1)
            await pilot.pause()

            assert await worker.wait() is True
            assert app.confirm_policy.allow_all_tools is True

    asyncio.run(run())
