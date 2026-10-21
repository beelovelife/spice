"""Textual conversation frontend for Spice."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.panel import Panel
from rich.table import Table
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.events import Resize
from textual.widgets import Static
from textual.worker import Worker

from spice.agent.agent_session import AgentSession
from spice.agent.session_history import HistoryError, session_history
from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    RoundCompleteEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnFinalizationStartEvent,
    TurnStartEvent,
)
from spice.agent.trace import RunTraceWriter, attach_trace_writer
from spice.agent.turn_outcome import pause_notice
from spice.cli.welcome import choose_welcome_quote
from spice.interactive.commands import SlashCommandRegistry, is_execute_request
from spice.interactive.confirm import ConfirmPolicy
from spice.interactive.tool_presentation import tool_presentation
from spice.interactive.types import (
    ChoiceRequest,
    CommandContext,
    CommandResult,
    ConfirmRequest,
    PanelView,
    TableView,
    TextView,
)
from spice.tools.base import ToolResult
from spice.tui.screens import ApprovalScreen, ChoiceScreen
from spice.tui.widgets import (
    ActivityLane,
    AssistantMessage,
    CommandSuggestions,
    Composer,
    NoticeMessage,
    QueueStrip,
    ReasoningMessage,
    StatusBar,
    ToolMessage,
    UserMessage,
)


class _TextualConsole:
    """Small compatibility adapter for trusted local extension commands."""

    def __init__(self, app: "SpiceTextualApp") -> None:
        self.app = app

    def print(self, *objects: object, **_: object) -> None:
        for value in objects:
            self.app.call_later(self.app._mount_renderable, value)


class _TextualInteractivePort:
    def __init__(self, app: "SpiceTextualApp", *, preview: str = "") -> None:
        self.app = app
        self.preview = preview

    async def choose(self, request: ChoiceRequest) -> str | None:
        return await self.app.push_screen_wait(ChoiceScreen(request))

    async def confirm(self, request: ConfirmRequest) -> str:
        return await self.app.push_screen_wait(
            ApprovalScreen(request, preview=self.preview)
        )


class SpiceTextualApp(App[None]):
    """A component-based TUI that consumes the existing AgentEvent stream."""

    CSS_PATH = "app.tcss"
    TITLE = "Spice"
    SUB_TITLE = "Local coding agent"

    BINDINGS = [
        Binding("escape", "interrupt", "Stop", priority=True),
        Binding("ctrl+c", "interrupt", "Stop", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_conversation", "Clear", show=False),
    ]

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        continue_session: bool = False,
        trace: bool = False,
        trace_file: Path | None = None,
        session: AgentSession | Any | None = None,
    ) -> None:
        super().__init__()
        self.confirm_policy = ConfirmPolicy()
        self._queued_messages: deque[str] = deque()
        self._turn_worker: Worker | None = None
        self._busy = False
        self._current_assistant: AssistantMessage | None = None
        self._current_reasoning: ReasoningMessage | None = None
        self._tool_messages: dict[str, ToolMessage] = {}
        self._last_reasoning = ""
        self._trace_writer: RunTraceWriter | None = None
        self._trace_path: Path | None = None
        self._input_history: list[str] = []

        self.agent_session: Any = session or AgentSession(
            cwd=Path.cwd(),
            provider=provider,
            model_id=model,
            confirm=self.confirm_tool,
            session_id=session_id,
            continue_latest=continue_session,
        )
        set_confirm = getattr(self.agent_session, "set_confirm", None)
        if set_confirm is not None:
            set_confirm(self.confirm_tool)
        else:
            self.agent_session.confirm = self.confirm_tool
        self._load_input_history()
        if trace or trace_file is not None:
            self._trace_writer, self._trace_path = attach_trace_writer(
                self.agent_session, path=trace_file
            )

    def compose(self) -> ComposeResult:
        yield Static("spice", id="header", markup=False)
        with VerticalScroll(id="chat-scroll"):
            yield Vertical(id="messages")
        yield ActivityLane()
        yield QueueStrip()
        yield CommandSuggestions()
        yield Composer(
            id="composer",
            placeholder="Ask Spice…  Enter send · Shift+Enter new line · / for commands",
            soft_wrap=True,
            show_line_numbers=False,
            compact=True,
        )
        yield StatusBar()

    async def on_mount(self) -> None:
        self._apply_size_classes(self.size.width, self.size.height)
        self.query_one(QueueStrip).display = False
        self.query_one(CommandSuggestions).display = False
        await self._show_initial_content()
        self._refresh_status()
        self.query_one(Composer).focus()

    def on_resize(self, event: Resize) -> None:
        self._apply_size_classes(event.size.width, event.size.height)

    def _apply_size_classes(self, width: int, height: int) -> None:
        self.screen.set_class(width < 70, "narrow")
        self.screen.set_class(height < 24, "short")

    async def on_unmount(self) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            self._turn_worker.cancel()
        close = getattr(self.agent_session, "aclose", None)
        if close is not None:
            await close()

    async def _show_initial_content(self) -> None:
        messages = session_history(self.agent_session)
        if not messages:
            english, chinese = choose_welcome_quote()
            await self._mount_message(
                NoticeMessage(
                    f"{english}\n{chinese}",
                    kind="welcome",
                ),
                follow=False,
            )
        else:
            for message in messages:
                if isinstance(message, HistoryError):
                    await self._mount_message(
                        NoticeMessage(message.display_text, kind="error"), follow=False,
                    )
                    continue
                role = getattr(message, "role", "")
                content = str(getattr(message, "content", "") or "")
                if role == "user":
                    await self._mount_message(UserMessage(content), follow=False)
                elif role == "assistant" and content:
                    widget = AssistantMessage()
                    await self._mount_message(widget, follow=False)
                    widget.settle(content)
                elif role == "tool":
                    prefix = (
                        "Tool error" if getattr(message, "is_error", False) else "Tool"
                    )
                    await self._mount_message(
                        NoticeMessage(f"{prefix}: {content}", kind="tool-history"),
                        follow=False,
                    )
        if self._trace_path is not None:
            await self._mount_message(
                NoticeMessage(f"Trace: {self._trace_path}", kind="info"),
                follow=False,
            )
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(
            animate=False, immediate=True
        )

    @on(Composer.Submitted)
    async def _on_composer_submitted(self, event: Composer.Submitted) -> None:
        composer = self.query_one(Composer)
        suggestions = self.query_one(CommandSuggestions)
        if suggestions.display and suggestions.accept_highlighted():
            return
        composer.load_text("")
        text = event.text.strip()
        if not text:
            return
        if not text.startswith("/"):
            self._remember_input(text)
        if self._busy:
            self._queued_messages.append(text)
            self._refresh_queue()
            self.notify(
                f"Queued message {len(self._queued_messages)}",
                title="Spice is still working",
                timeout=2,
            )
            return
        await self._dispatch_input(text)

    @on(Composer.Changed)
    def _on_composer_changed(self, event: Composer.Changed) -> None:
        value = event.text_area.text.lstrip()
        suggestions = self.query_one(CommandSuggestions)
        if not value.startswith("/") or any(char.isspace() for char in value):
            suggestions.show_commands(())
            return
        registry = SlashCommandRegistry(getattr(self.agent_session, "extensions", None))
        prefix = value.lower()
        matches = [
            item
            for item in registry.completion_items()
            if item[0].lower().startswith(prefix)
        ][:7]
        suggestions.show_commands(matches)

    @on(Composer.CompletionRequested)
    def _on_completion_requested(self) -> None:
        self.query_one(CommandSuggestions).accept_highlighted()

    @on(Composer.NavigationRequested)
    def _on_composer_navigation(self, event: Composer.NavigationRequested) -> None:
        composer = self.query_one(Composer)
        suggestions = self.query_one(CommandSuggestions)
        if suggestions.display:
            if event.direction < 0:
                suggestions.action_cursor_up()
            else:
                suggestions.action_cursor_down()
            return
        if not composer.text.strip() and self._input_history:
            suggestions.show_history(self._input_history)
            return
        composer.move_cursor_vertically(event.direction)

    @on(CommandSuggestions.Accepted)
    def _on_command_suggestion_accepted(
        self, event: CommandSuggestions.Accepted
    ) -> None:
        suffix = "" if event.trigger.endswith(":") else " "
        self._load_composer_value(f"{event.trigger}{suffix}")

    @on(CommandSuggestions.HistoryAccepted)
    def _on_history_suggestion_accepted(
        self, event: CommandSuggestions.HistoryAccepted
    ) -> None:
        self._load_composer_value(event.text)

    async def _dispatch_input(self, text: str) -> None:
        if text in {"clear", "cls"}:
            text = "/clear"
        if text in {"exit", "quit"}:
            self.exit()
            return
        if text.startswith("/"):
            self._busy = True
            self._turn_worker = self.run_worker(
                self._run_command(text), name="slash-command", exclusive=True
            )
            return
        if (
            getattr(self.agent_session, "plan_state", None) is not None
            and self.agent_session.plan_state.is_plan_mode
            and is_execute_request(text)
        ):
            text = self.agent_session.approve_plan("manual")
            await self._mount_message(
                NoticeMessage("Switched to edit mode. Executing the approved plan.")
            )
        await self._mount_message(UserMessage(text))
        self._busy = True
        self._turn_worker = self.run_worker(
            self._run_prompt(text), name="agent-turn", exclusive=True
        )

    async def _run_prompt(self, text: str) -> None:
        self._start_activity("Waiting for model", self._model_label())
        followup: str | None = None
        try:
            async for event in self.agent_session.prompt(text):
                await self._handle_event(event)
            followup = await self._plan_followup()
        except asyncio.CancelledError:
            await self._settle_reasoning()
            await self._mount_message(
                NoticeMessage("Current turn interrupted.", kind="warning")
            )
            raise
        except Exception as exc:  # noqa: BLE001
            await self._mount_message(NoticeMessage(f"Error: {exc}", kind="error"))
        finally:
            self.query_one(ActivityLane).stop()
            self._busy = False
            self._turn_worker = None
            self._refresh_status()
            if followup:
                self.call_later(self._dispatch_input, followup)
            else:
                self.call_later(self._drain_queue)

    async def _run_command(self, text: str) -> None:
        try:
            registry = SlashCommandRegistry(
                getattr(self.agent_session, "extensions", None)
            )
            ctx = CommandContext(
                session=self.agent_session,
                cwd=Path.cwd(),
                port=_TextualInteractivePort(self),
                raw=text,
                args="",
                confirm_policy=self.confirm_policy,
                extras={
                    "registry": registry,
                    "last_reasoning": lambda: self._last_reasoning,
                    "extension_context": SimpleNamespace(
                        console=_TextualConsole(self),
                        agent_session=self.agent_session,
                        cwd=Path.cwd(),
                    ),
                },
            )
            result = await registry.execute(text, ctx)
            await self._apply_command_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._mount_message(
                NoticeMessage(f"Command failed: {exc}", kind="error")
            )
        finally:
            self._busy = False
            self._turn_worker = None
            self._refresh_status()
            self.call_later(self._drain_queue)

    async def _apply_command_result(self, result: CommandResult) -> None:
        if result.exit_requested:
            self.exit()
            return
        previous = self.agent_session
        new_session = result.session
        replacing = new_session is not None and new_session is not previous
        if result.clear_requested or (replacing and result.replay_session_history):
            await self._clear_messages()
        if replacing and new_session is not None:
            self.confirm_policy = ConfirmPolicy()
            self.agent_session = new_session
            set_confirm = getattr(self.agent_session, "set_confirm", None)
            if set_confirm is not None:
                set_confirm(self.confirm_tool)
            else:
                self.agent_session.confirm = self.confirm_tool
            self._load_input_history()
            if self._trace_writer is not None:
                self._trace_writer.bind(self.agent_session)
            close = getattr(previous, "aclose", None)
            if close is not None:
                await close()
            if result.replay_session_history:
                await self._show_initial_content()
        for view in result.views:
            await self._paint_view(view)
        if replacing:
            await self._mount_message(
                NoticeMessage(
                    f"Session {self.agent_session.session_label} · {self._model_label()}",
                    kind="success",
                )
            )
        if result.followup_prompt:
            await self._mount_message(UserMessage(result.followup_prompt))
            self._busy = True
            await self._run_prompt(result.followup_prompt)

    async def _paint_view(self, view: TextView | TableView | PanelView) -> None:
        if isinstance(view, TextView):
            kind = {
                "success": "success",
                "warning": "warning",
                "error": "error",
                "dim": "dim",
            }.get(view.style, "info")
            await self._mount_message(NoticeMessage(view.text, kind=kind))
        elif isinstance(view, TableView):
            table = Table(*view.columns, title=view.title)
            for row in view.rows:
                table.add_row(*row)
            await self._mount_renderable(table)
        elif isinstance(view, PanelView):
            await self._mount_renderable(Panel(view.body, title=view.title))

    async def _mount_renderable(self, renderable: Any) -> None:
        await self._mount_message(Static(renderable, classes="command-view"))

    async def _handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, (AgentStartEvent, AgentEndEvent)):
            return
        if isinstance(event, TurnStartEvent):
            self._start_activity("Waiting for model", self._model_label())
        elif isinstance(event, ReasoningDeltaEvent):
            self._start_activity("Thinking")
            if self._current_reasoning is None:
                self._current_reasoning = ReasoningMessage(kind=event.kind)
                await self._mount_message(self._current_reasoning)
            self._current_reasoning.append_delta(event.text)
        elif isinstance(event, TextDeltaEvent):
            await self._settle_reasoning()
            self._start_activity("Generating response")
            if self._current_assistant is None:
                self._current_assistant = AssistantMessage()
                await self._mount_message(self._current_assistant)
            self._current_assistant.append_delta(event.text)
        elif isinstance(event, AssistantMessageEvent):
            await self._settle_reasoning()
            if event.text and self._current_assistant is None:
                self._current_assistant = AssistantMessage()
                await self._mount_message(self._current_assistant)
            if self._current_assistant is not None:
                self._current_assistant.settle(event.text)
                self._current_assistant = None
        elif isinstance(event, ToolExecutionStartEvent):
            await self._settle_reasoning()
            if self._current_assistant is not None:
                self._current_assistant.settle()
                self._current_assistant = None
            presentation = tool_presentation(event.tool_name)
            self._start_activity(f"Running {presentation.label.lower()}")
            tool = ToolMessage(event.tool_call_id, event.tool_name, event.args or {})
            self._tool_messages[event.tool_call_id] = tool
            await self._mount_message(tool)
        elif isinstance(event, ToolExecutionUpdateEvent):
            tool = self._tool_messages.get(event.tool_call_id)
            if tool is not None:
                tool.append_update(event.text)
        elif isinstance(event, ToolExecutionEndEvent):
            tool = self._tool_messages.get(event.tool_call_id)
            if tool is not None:
                tool.settle(_tool_summary(event.result), event.result)
            if event.result.is_error:
                self._start_activity("Recovering from tool error")
            else:
                self._start_activity("Waiting for model", self._model_label())
        elif isinstance(event, ModelRetryEvent):
            await self._settle_reasoning()
            await self._mount_message(
                NoticeMessage(
                    f"Retry {event.next_attempt}/{event.max_attempts} in "
                    f"{event.delay_seconds:.1f}s · {event.error}",
                    kind="warning",
                )
            )
            self._start_activity(
                f"Retrying model {event.next_attempt}/{event.max_attempts}",
                f"in {event.delay_seconds:.1f}s",
            )
        elif isinstance(event, ModelFallbackEvent):
            await self._settle_reasoning()
            await self._mount_message(
                NoticeMessage(
                    f"Model fallback: {event.from_provider}/{event.from_model} → "
                    f"{event.to_provider}/{event.to_model}",
                    kind="warning",
                )
            )
            self._start_activity(
                "Switching model", f"{event.to_provider}/{event.to_model}"
            )
        elif isinstance(event, RoundCompleteEvent):
            await self._settle_reasoning()
            if self._current_assistant is not None:
                self._current_assistant.settle()
                self._current_assistant = None
            self._start_activity("Waiting for model", self._model_label())
        elif isinstance(event, TurnFinalizationStartEvent):
            await self._settle_reasoning()
            if self._current_assistant is not None:
                self._current_assistant.settle()
                self._current_assistant = None
            self._start_activity("整理进度", f"已执行 {event.rounds_used} 轮")
        elif isinstance(event, AgentErrorEvent):
            await self._settle_reasoning()
            kind = (
                "warning"
                if event.kind in {"fatal_tool", "user_interrupted"}
                else "error"
            )
            await self._mount_message(NoticeMessage(event.message, kind=kind))
        elif isinstance(event, TurnEndEvent):
            await self._settle_reasoning()
            if self._current_assistant is not None:
                self._current_assistant.settle(event.text)
                self._current_assistant = None
            self.query_one(ActivityLane).stop()
            if event.status == "paused":
                await self._mount_message(NoticeMessage(
                    pause_notice(event.checkpoint.round_limit if event.checkpoint else None),
                    kind="warning",
                ))

    async def _settle_reasoning(self) -> None:
        reasoning = self._current_reasoning
        if reasoning is None:
            return
        self._last_reasoning = reasoning.text
        reasoning.settle()
        self._current_reasoning = None

    async def _plan_followup(self) -> str | None:
        plan_state = getattr(self.agent_session, "plan_state", None)
        if plan_state is None or not plan_state.is_plan_mode or not plan_state.steps:
            return None
        choice = await _TextualInteractivePort(self).choose(
            ChoiceRequest(
                title="Plan ready",
                items=(
                    _choice(
                        "auto", "Execute in auto mode", "allow file edits this session"
                    ),
                    _choice("refine", "Refine the plan", "stay in plan mode"),
                ),
                current_id="auto",
            )
        )
        if choice == "auto":
            self.confirm_policy.allow_file_edits = True
            return self.agent_session.approve_plan("auto")
        return None

    async def confirm_tool(self, tool_name: str, args: dict) -> bool:
        presentation = tool_presentation(tool_name)
        try:
            args_text = json.dumps(args, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            args_text = str(args)
        preview = f"{presentation.icon} {presentation.label}\n{args_text}".strip()
        allowed = await self.confirm_policy.confirm(
            tool_name,
            args,
            _TextualInteractivePort(self, preview=preview),
        )
        self._refresh_status()
        return allowed

    async def _mount_message(self, widget, *, follow: bool = True) -> None:
        await self.query_one("#messages", Vertical).mount(widget)
        if follow:
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(
                animate=False, immediate=True
            )

    async def _clear_messages(self) -> None:
        await self.query_one("#messages", Vertical).remove_children()
        self._tool_messages.clear()
        self._current_assistant = None
        self._current_reasoning = None

    def _start_activity(self, label: str, detail: str = "") -> None:
        self.query_one(ActivityLane).start(label, detail)

    def _model_label(self) -> str:
        return str(getattr(self.agent_session, "runtime_model_label", "unknown model"))

    def _refresh_status(self) -> None:
        model = getattr(self.agent_session, "model", None)
        protocol = str(getattr(model, "protocol", None) or "provider default")
        if self.confirm_policy.allow_all_tools:
            permission = "session/all"
        elif self.confirm_policy.allow_file_edits:
            permission = "session/file-edits"
        elif self.confirm_policy.allow_read_only_bash:
            permission = "session/read-only-shell"
        else:
            permission = "ask"
        self.query_one(StatusBar).set_status(
            cwd=str(getattr(self.agent_session, "cwd", Path.cwd())),
            model=self._model_label(),
            protocol=protocol,
            permission=permission,
        )

    def _refresh_queue(self) -> None:
        self.query_one(QueueStrip).show_messages(self._queued_messages)

    def _load_composer_value(self, value: str) -> None:
        composer = self.query_one(Composer)
        composer.load_text(value)
        lines = value.split("\n")
        composer.cursor_location = (len(lines) - 1, len(lines[-1]))
        self.query_one(CommandSuggestions).show_commands(())
        composer.focus()

    def _load_input_history(self) -> None:
        self._input_history = []
        for message in getattr(self.agent_session, "messages", ()):
            if getattr(message, "role", "") != "user":
                continue
            text = str(getattr(message, "content", "") or "").strip()
            if text and (not self._input_history or self._input_history[-1] != text):
                self._input_history.append(text)

    def _remember_input(self, text: str) -> None:
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)

    async def _drain_queue(self) -> None:
        if self._busy or not self._queued_messages:
            self._refresh_queue()
            return
        text = self._queued_messages.popleft()
        self._refresh_queue()
        await self._dispatch_input(text)

    def action_interrupt(self) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            self._start_activity("Stopping")
            self._turn_worker.cancel()
            return
        if self._queued_messages:
            self._queued_messages.pop()
            self._refresh_queue()
            self.notify("Removed the last queued message", timeout=2)

    async def action_clear_conversation(self) -> None:
        await self._clear_messages()


def _choice(choice_id: str, label: str, detail: str):
    from spice.interactive.types import ChoiceItem

    return ChoiceItem(choice_id, label, detail)


def _tool_summary(result: ToolResult) -> str:
    if result.is_error:
        first = next(
            (
                line.strip()
                for line in (result.content or "").splitlines()
                if line.strip()
            ),
            "failed",
        )
        return first[:120]
    details = result.details if isinstance(result.details, dict) else {}
    if "returncode" in details:
        return f"exit {details.get('returncode')}"
    if "line_count" in details:
        return f"{details.get('line_count')} lines"
    if "match_count" in details:
        return f"{details.get('match_count')} matches"
    if "total_entries" in details:
        return f"{details.get('total_entries')} entries"
    return "done"


def run_textual_tui(
    *,
    provider: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    continue_session: bool = False,
    trace: bool = False,
    trace_file: Path | None = None,
) -> None:
    app = SpiceTextualApp(
        provider=provider,
        model=model,
        session_id=session_id,
        continue_session=continue_session,
        trace=trace,
        trace_file=trace_file,
    )
    app.run()
