"""AgentSession composition root."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Literal

from spice.agent.event_dispatcher import AgentEventDispatcher, AgentSessionListener
from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    ToolExecutionEndEvent,
    TurnEndEvent,
)
from spice.agent.file_references import FileReferenceError, expand_file_references
from spice.agent.logging_config import get_logger
from spice.agent.long_task import LONG_TASK_REF_ENTRY, LONG_TASK_STATE_ENTRY, LongTaskRef, LongTaskState
from spice.agent.long_task_tools import create_long_task_tools
from spice.agent.subagent import SubagentManager
from spice.agent.plan_state import (
    PLAN_STATE_ENTRY,
    ApprovalMode,
    InteractionMode,
    PlanState,
    execution_prompt,
    extract_plan_steps,
    plan_prompt,
)
from spice.agent.todo_state import TODO_STATE_ENTRY, TodoState, todos_from_plan_steps
from spice.agent.compaction import (
    CompactionCheck,
    CompactionError,
    CompactionResult,
    CompactionSettings,
    check_compaction_needed,
    estimate_messages_tokens,
    generate_summary,
    prepare_compaction,
)
from spice.agent.loop import run_turn
from spice.agent.prompts import build_system_prompt
from spice.agent.sessions import SessionInfo, SessionStore
from spice.agent.tool_results import cleanup_tool_results, prepare_tool_message_for_session
from spice.extensions.manager import ExtensionManager
from spice.llm.config import get_api_key, load_config
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.model_registry import ModelRegistry, find_initial_model
from spice.llm.routing import build_model_route
from spice.llm.types import ModelRequestOptions
from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.storage.factory import create_long_task_store, create_memory_store, create_session_store
from spice.tools.base import ConfirmFn
from spice.tools.file_state import FileStateStore
from spice.tools.todo import create_update_todo_tool
from spice.tools.tool_registry import create_coding_tools, create_read_only_tools

logger = get_logger(__name__)
LONG_TASK_TOOL_NAMES = {"complete_long_task"}
LONG_TASK_COMPLETION_TOOL_NAMES = {"complete_long_task"}


class AgentSession:
    def __init__(
        self,
        cwd: Path | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        confirm: ConfirmFn | None = None,
        session_id: str | None = None,
        continue_latest: bool = False,
        session_store: SessionStore | None = None,
        extension_manager: ExtensionManager | None = None,
    ) -> None:
        self.cwd = (cwd or Path.cwd()).resolve()
        self.config = load_config()
        self.workspace_policy = create_workspace_policy(self.config.sandbox, cwd=self.cwd)
        self.environment = create_environment(self.config.sandbox, cwd=self.cwd)
        self.session_store = session_store or create_session_store(self.config, cwd=self.cwd)
        self.session: SessionInfo | None = None

        if session_id:
            self.session = self.session_store.resolve(session_id, cwd=self.cwd)
        elif continue_latest:
            self.session = self.session_store.latest(cwd=self.cwd)

        self.registry = ModelRegistry()
        result = find_initial_model(self.registry, self.config, provider=provider, model_id=model_id)
        if not result.model:
            raise RuntimeError(result.message or "No model configured.")
        self.model = result.model
        self.extensions = extension_manager or ExtensionManager(cwd=self.cwd)
        if extension_manager is None:
            self.extensions.load_default()
        self.event_dispatcher = AgentEventDispatcher(self.extensions)
        self.file_states = FileStateStore()
        self.confirm = confirm
        self.memory_store = create_memory_store(self.config)
        self.memory_context = self.memory_store.context_snapshot() if self.config.memory_enabled else ""
        self.subagents_enabled = self.config.subagents_enabled
        self.long_task_store = create_long_task_store(self.config, file_base_dir=self.session_store.base_dir.parent / "tasks")
        self.plan_state = self._load_plan_state()
        self.todo_state = self._load_todo_state()
        self.long_task_state = self._load_long_task_state()
        self._defer_custom_state = False
        self._dirty_plan_state = False
        self._dirty_todo_state = False
        self._dirty_long_task_state = False
        self.subagent_manager = SubagentManager(
            cwd=self.cwd,
            model=self.model,
            options_factory=self._request_options,
            confirm=self.confirm,
            tools_factory=self._subagent_tool_candidates,
            max_concurrent=self.config.max_concurrent_subagents,
        )
        self._edit_tools = [
            *[
                tool
                for tool in create_coding_tools(
                    memory_enabled=self.config.memory_enabled,
                    subagents_enabled=True,
                )
                if tool.name not in LONG_TASK_TOOL_NAMES
            ],
            create_update_todo_tool(get_state=lambda: self.todo_state, set_state=self._set_todo_state),
            *create_long_task_tools(
                get_state=lambda: self.long_task_state,
                set_state=self._set_long_task_state,
                can_complete=self._can_complete_long_task,
            ),
            *self.extensions.tools(),
        ]
        self._plan_tools = create_read_only_tools()
        self.tools = self._edit_tools

        self.messages: list[Message] = [self._build_system_message()]
        if self.session is not None:
            session_context = self.session_store.build_context(self.session.id)
            self.messages.extend(session_context.messages)

    def subscribe(self, listener: AgentSessionListener) -> Callable[[], None]:
        return self.event_dispatcher.subscribe(listener)

    @property
    def listener_errors(self) -> list[str]:
        return self.event_dispatcher.listener_errors

    async def prompt(self, text: str) -> AsyncIterator[AgentEvent]:
        text = await self.extensions.transform_input(text)
        user_text = text
        if self.plan_state.is_plan_mode:
            text = plan_prompt(text, self.plan_state)
        runtime_context = self._runtime_context()
        try:
            text = expand_file_references(text, cwd=self.cwd, model=self.model).message
        except FileReferenceError as exc:
            error_event = AgentErrorEvent(str(exc))
            await self._dispatch_event(error_event)
            yield error_event
            return
        self._ensure_session()
        options = self._request_options()
        model_route = build_model_route(
            self.model,
            routing_settings=self.config.model_routing,
            resolve_model=lambda profile: self.registry.find(None, profile),
            options_factory=self._request_options_for_model,
        )
        check_messages = [*self.messages, Message(role="user", content=text)]
        if self.compaction_status(check_messages).should_compact:
            try:
                await self.compact(reason="auto", force=False, options=options)
            except CompactionError as exc:
                error_event = AgentErrorEvent(f"Auto-compaction failed: {exc}")
                await self._dispatch_event(error_event)
                yield error_event
                return
        persisted_from = len(self.messages)
        start_event = AgentStartEvent(session_id=self.session_id)
        await self._dispatch_event(start_event)
        yield start_event
        logger.info(
            "session_prompt_start session_id=%s mode=%s message_count=%d",
            self.session_id,
            self.plan_state.mode,
            len(self.messages),
        )
        try:
            turn_text = ""
            self._defer_custom_state = True
            async for event in run_turn(
                prompt=text,
                messages=self.messages,
                model=self.model,
                tools=self._active_tools(),
                options=options,
                cwd=self.cwd,
                workspace=self.workspace_policy,
                environment=self.environment,
                confirm=self.confirm,
                extensions=self.extensions,
                file_states=self.file_states,
                subagent_manager=self.subagent_manager if self.subagents_enabled else None,
                session_label=self.session_label,
                runtime_context=runtime_context,
                model_route=model_route,
                tools_settings=self.config.tools,
            ):
                if isinstance(event, TurnEndEvent):
                    turn_text = event.text
                    if event.stop_reason == "max_tool_rounds" and self.long_task_state.is_active:
                        self.long_task_state.record_continuation(stop_reason=event.stop_reason)
                        if not self.todo_state.items:
                            self.long_task_state.mark_needs_attention(reason="max_tool_rounds_without_todo")
                        self._persist_long_task_state()
                elif isinstance(event, ToolExecutionEndEvent) and event.tool_name == "memory" and not event.result.is_error:
                    self._refresh_memory_context()
                await self._dispatch_event(event)
                yield event
            if self.plan_state.is_plan_mode:
                self.plan_state.objective = self.plan_state.objective or user_text
                steps = extract_plan_steps(turn_text)
                if steps:
                    self.plan_state.steps = steps
                self._persist_plan_state()
        finally:
            self._defer_custom_state = False
            messages_persisted = False
            try:
                parent_id = self._persist_messages_from(persisted_from)
                messages_persisted = True
                self._flush_custom_state(parent_id=parent_id)
            except Exception:
                if not messages_persisted:
                    logger.exception(
                        "session_persist_messages_failed session_id=%s dirty_plan=%s dirty_todo=%s",
                        self.session_id,
                        self._dirty_plan_state,
                        self._dirty_todo_state,
                    )
                    self._discard_deferred_custom_state()
                raise
            end_event = AgentEndEvent(session_id=self.session_id)
            await self._dispatch_event(end_event)
            yield end_event
            logger.info("session_prompt_end session_id=%s", self.session_id)

    async def _dispatch_event(self, event: AgentEvent) -> None:
        await self.event_dispatcher.dispatch(event)

    def _persist_messages_from(self, index: int) -> str | None:
        if index >= len(self.messages):
            return self.session_store.info(self.session_id).leaf_id if self.session is not None else None
        self._ensure_session()
        parent_id = self.session_store.info(self.session_id).leaf_id
        for message in self.messages[index:]:
            if message.role == "assistant":
                message.provider = message.provider or self.model.provider
                message.model = message.model or self.model.id
            persisted_message = prepare_tool_message_for_session(message, cwd=self.cwd, session_id=self.session_id)
            parent_id = self.session_store.append_message(self.session_id, persisted_message, parent_id=parent_id)
        self.session = self.session_store.info(self.session_id)
        return parent_id

    def get_active_tools(self) -> list[str]:
        return [tool.name for tool in self._active_tools()]

    def start_long_task(self, objective: str, *, note: str = "", max_continuation_rounds: int | None = None) -> LongTaskState:
        objective = objective.strip()
        if not objective:
            raise ValueError("objective is required")
        self._ensure_session()
        self.long_task_state = self.long_task_store.create(
            objective=objective,
            session_id=self.session_id,
            note=note,
            max_continuation_rounds=max_continuation_rounds,
        )
        self._persist_long_task_state()
        self._refresh_system_message()
        return self.long_task_state

    def complete_long_task(self, *, note: str = "", force: bool = False) -> LongTaskState:
        ok, message = self._can_complete_long_task(force=force)
        if not ok:
            raise ValueError(message)
        if not self.long_task_state.is_active:
            raise ValueError("No sustained goal is active.")
        self.long_task_state.complete(note=note)
        self._persist_long_task_state()
        self._refresh_system_message()
        return self.long_task_state

    def cancel_long_task(self, *, note: str = "") -> LongTaskState:
        if not self.long_task_state.is_active:
            raise ValueError("No sustained goal is active.")
        self.long_task_state.cancel(note=note)
        self._persist_long_task_state()
        self._refresh_system_message()
        return self.long_task_state

    def long_task_status(self) -> LongTaskState:
        return self.long_task_state

    def set_subagents_enabled(self, enabled: bool) -> None:
        self.subagents_enabled = enabled
        self._refresh_system_message()

    def set_interaction_mode(self, mode: InteractionMode) -> None:
        self.plan_state.mode = mode
        self._persist_plan_state()
        self._refresh_system_message()

    def toggle_interaction_mode(self) -> InteractionMode:
        next_mode: InteractionMode = "edit" if self.plan_state.mode == "plan" else "plan"
        self.set_interaction_mode(next_mode)
        return next_mode

    def start_plan(self, objective: str = "") -> None:
        objective = objective.strip()
        if objective:
            self.plan_state.objective = objective
            self.plan_state.steps = []
        self.set_interaction_mode("plan")

    def cancel_plan(self) -> None:
        self.plan_state = PlanState()
        self._persist_plan_state()
        self._refresh_system_message()

    def approve_plan(self, approval_mode: ApprovalMode) -> str:
        self.plan_state.approval_mode = approval_mode
        self.plan_state.mode = "edit"
        self._persist_plan_state()
        if self.plan_state.steps:
            self._set_todo_state(todos_from_plan_steps(self.plan_state.steps))
        self._refresh_system_message()
        return execution_prompt(self.plan_state)

    def compaction_status(self, messages: list[Message] | None = None) -> CompactionCheck:
        return check_compaction_needed(messages or self.messages, self.model)

    async def compact(
        self,
        *,
        focus: str | None = None,
        reason: Literal["manual", "auto"] = "manual",
        force: bool = True,
        options: ModelRequestOptions | None = None,
    ) -> CompactionResult:
        self._ensure_session()
        path_entries = self.session_store.path_entries(self.session_id)
        plan = prepare_compaction(path_entries, CompactionSettings())
        if plan is None:
            raise CompactionError("Nothing to compact.")
        request_options = options or self._request_options()
        summary = await generate_summary(plan=plan, model=self.model, options=request_options, focus=focus)
        compacted_messages = [self._build_system_message()]
        compacted_messages.append(Message(role="system", content=f"Previous conversation summary:\n{summary}"))
        compacted_messages.extend(plan.kept_messages)
        tokens_after = estimate_messages_tokens(compacted_messages)
        self.session_store.append_compaction(
            self.session_id,
            summary=summary,
            first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=plan.tokens_before,
            details={
                "reason": reason,
                "focus": focus,
                "estimatedTokensAfter": tokens_after,
            },
        )
        if self.config.memory_enabled:
            try:
                self.memory_store.append_history(
                    summary=summary,
                    source="compaction",
                    session_id=self.session_id,
                    metadata={
                        "reason": reason,
                        "focus": focus,
                        "first_kept_entry_id": plan.first_kept_entry_id,
                        "tokens_before": plan.tokens_before,
                        "tokens_after": tokens_after,
                    },
                )
            except Exception:
                logger.warning("memory_history_append_failed session_id=%s", self.session_id, exc_info=True)
        self.session = self.session_store.info(self.session_id)
        self.plan_state = self._load_plan_state()
        self.todo_state = self._load_todo_state()
        self.long_task_state = self._load_long_task_state()
        session_context = self.session_store.build_context(self.session_id)
        self.messages = [self._build_system_message()]
        self.messages.extend(session_context.messages)
        return CompactionResult(
            summary=summary,
            first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=plan.tokens_before,
            tokens_after=tokens_after,
            reason=reason,
            focus=focus,
        )

    def set_model(self, model: Model) -> None:
        self.model = model
        self.subagent_manager.set_model(model)
        self._refresh_system_message()
        if self.session is None:
            return
        parent_id = self.session_store.info(self.session_id).leaf_id
        self.session_store.append_model_change(
            self.session_id,
            provider=model.provider,
            model=model.id,
            parent_id=parent_id,
        )
        self.session = self.session_store.info(self.session_id)

    def rewind(self, entry_id: str) -> None:
        if self.session is None:
            raise RuntimeError("Session is not initialized.")
        self.session_store.set_leaf(self.session_id, entry_id)
        self.session = self.session_store.info(self.session_id)
        self.plan_state = self._load_plan_state()
        self.todo_state = self._load_todo_state()
        self.long_task_state = self._load_long_task_state()
        session_context = self.session_store.build_context(self.session.id)
        self.messages = [self._build_system_message()]
        self.messages.extend(session_context.messages)

    def reset(self) -> None:
        if self.session is None:
            raise RuntimeError("Session is not initialized.")
        self.session = self.session_store.reset(self.session_id)
        cleanup_tool_results(self.cwd, self.session_id)
        self.plan_state = PlanState()
        self.todo_state = TodoState()
        self.long_task_state = LongTaskState()
        self.messages = [self._build_system_message()]

    @property
    def session_id(self) -> str:
        if self.session is None:
            raise RuntimeError("Session is not initialized.")
        return self.session.id

    @property
    def session_label(self) -> str:
        return self.session.id if self.session else "new"

    def _ensure_session(self) -> SessionInfo:
        if self.session is None:
            self.session = self.session_store.create(cwd=self.cwd, provider=self.model.provider, model=self.model.id)
        return self.session

    def _request_options(self) -> ModelRequestOptions:
        return self._request_options_for_model(self.model)

    def _request_options_for_model(self, model: Model) -> ModelRequestOptions:
        return ModelRequestOptions(
            api_key=get_api_key(model.provider, env_names=model.api_key_envs),
            temperature=model.temperature if model.temperature is not None else self.config.temperature,
            max_tokens=model.output_tokens,
            base_url=(self.config.base_url if model is self.model else None) or model.base_url,
        )

    def _build_system_message(self) -> Message:
        content = build_system_prompt(
            self.cwd,
            self._active_tools(),
            runtime_model=self.runtime_model_label,
            memory_context=self.memory_context,
        )
        return Message(role="system", content=content)

    def _refresh_system_message(self) -> None:
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = self._build_system_message()

    def _refresh_memory_context(self) -> None:
        self.memory_context = self.memory_store.context_snapshot() if self.config.memory_enabled else ""
        self._refresh_system_message()

    @property
    def runtime_model_label(self) -> str:
        return f"{self.model.provider}/{self.model.id}"

    def _active_tools(self) -> list:
        if self.plan_state.is_plan_mode:
            return self._plan_tools
        tools = self._edit_tools
        if self.long_task_state.is_active:
            tools = [tool for tool in tools if tool.name not in LONG_TASK_TOOL_NAMES or tool.name in LONG_TASK_COMPLETION_TOOL_NAMES]
        else:
            tools = [tool for tool in tools if tool.name not in LONG_TASK_TOOL_NAMES]
        if self.subagents_enabled:
            return tools
        return [tool for tool in tools if tool.name != "spawn_subagents"]

    def _subagent_tool_candidates(self) -> list:
        return [
            *[
                tool
                for tool in create_coding_tools(memory_enabled=self.config.memory_enabled, subagents_enabled=False)
                if tool.name not in LONG_TASK_TOOL_NAMES
            ],
            *self.extensions.tools(),
        ]

    def _persist_plan_state(self) -> None:
        if self._defer_custom_state:
            self._dirty_plan_state = True
            return
        self._ensure_session()
        self.session_store.append_custom(self.session_id, self.plan_state.to_dict())
        self.session = self.session_store.info(self.session_id)

    def _set_todo_state(self, state: TodoState) -> None:
        self.todo_state = state
        self._persist_todo_state()
        if self.long_task_state.is_active and self.todo_state.items and not self.todo_state.has_active_items():
            self.long_task_state.mark_completion_candidate()
            self._persist_long_task_state()

    def _set_long_task_state(self, state: LongTaskState) -> LongTaskState:
        self._ensure_session()
        previous_task_id = self.long_task_state.task_id
        if state.objective and not state.task_id:
            if previous_task_id and self.long_task_state.objective == state.objective:
                state.task_id = previous_task_id
                state.session_id = self.long_task_state.session_id or self.session_id
                state.created_at = self.long_task_state.created_at
            else:
                created = self.long_task_store.create(
                    objective=state.objective,
                    session_id=self.session_id,
                    note=state.notes[-1] if state.notes else "",
                    max_continuation_rounds=state.max_continuation_rounds,
                )
                self.long_task_state = created
                self._persist_long_task_state()
                return created
        if state.task_id:
            state.session_id = state.session_id or self.session_id
            self.long_task_store.save_state(state)
            self.long_task_store.append_event(state.task_id, state.status, {"status": state.status})
        self.long_task_state = state
        self._persist_long_task_state()
        return state

    def _can_complete_long_task(self, *, force: bool = False) -> tuple[bool, str]:
        if force:
            return True, ""
        if self.todo_state.has_active_items():
            return False, "Cannot complete sustained goal while todo items are still pending or in progress."
        return True, ""

    def _persist_todo_state(self) -> None:
        if self._defer_custom_state:
            self._dirty_todo_state = True
            return
        self._ensure_session()
        self.session_store.append_custom(self.session_id, self.todo_state.to_dict())
        self.session = self.session_store.info(self.session_id)

    def _persist_long_task_state(self) -> None:
        if self.long_task_state.task_id:
            self.long_task_store.save_state(self.long_task_state)
            self.long_task_store.write_checkpoint(
                self.long_task_state,
                todos=self.todo_state.read(),
                last_action="turn_state_persisted",
            )
        if self._defer_custom_state:
            self._dirty_long_task_state = True
            return
        self._ensure_session()
        if self.long_task_state.task_id:
            self.session_store.append_custom(self.session_id, self.long_task_state.to_ref().to_dict())
        else:
            self.session_store.append_custom(self.session_id, self.long_task_state.to_session_state_dict())
        self.session = self.session_store.info(self.session_id)

    def _flush_custom_state(self, parent_id: str | None = None) -> None:
        if not self._dirty_plan_state and not self._dirty_todo_state and not self._dirty_long_task_state:
            return
        logger.info(
            "custom_state_flush session_id=%s dirty_plan=%s dirty_todo=%s dirty_long_task=%s parent_id=%s",
            self.session_id,
            self._dirty_plan_state,
            self._dirty_todo_state,
            self._dirty_long_task_state,
            parent_id,
        )
        self._ensure_session()
        if parent_id is None:
            parent_id = self.session_store.info(self.session_id).leaf_id
        if self._dirty_plan_state:
            parent_id = self.session_store.append_custom(self.session_id, self.plan_state.to_dict(), parent_id=parent_id)
            self._dirty_plan_state = False
        if self._dirty_todo_state:
            parent_id = self.session_store.append_custom(self.session_id, self.todo_state.to_dict(), parent_id=parent_id)
            self._dirty_todo_state = False
        if self._dirty_long_task_state:
            payload = (
                self.long_task_state.to_ref().to_dict()
                if self.long_task_state.task_id
                else self.long_task_state.to_session_state_dict()
            )
            parent_id = self.session_store.append_custom(self.session_id, payload, parent_id=parent_id)
            self._dirty_long_task_state = False
        self.session = self.session_store.info(self.session_id)

    def _discard_deferred_custom_state(self) -> None:
        self._dirty_plan_state = False
        self._dirty_todo_state = False
        self._dirty_long_task_state = False

    def _load_plan_state(self) -> PlanState:
        if self.session is None:
            return PlanState()
        state = PlanState()
        try:
            entries = self.session_store.path_entries(self.session.id)
        except ValueError:
            return state
        for entry in entries:
            if entry.type == "custom" and entry.data.get("customType") == PLAN_STATE_ENTRY:
                state = PlanState.from_dict(entry.data)
        # Resume the stored plan content, but default the interaction mode back
        # to edit so a fresh CLI/TUI entry does not reopen in plan mode.
        state.mode = "edit"
        return state

    def _load_todo_state(self) -> TodoState:
        if self.session is None:
            return TodoState()
        state = TodoState()
        try:
            entries = self.session_store.entries(self.session.id)
        except ValueError:
            return state
        for entry in entries:
            if entry.type == "custom" and entry.data.get("customType") == TODO_STATE_ENTRY:
                state = TodoState.from_dict(entry.data)
        return state

    def _load_long_task_state(self) -> LongTaskState:
        if self.session is None:
            return LongTaskState()
        state = LongTaskState()
        try:
            entries = self.session_store.path_entries(self.session.id)
        except ValueError:
            return state
        for entry in entries:
            if entry.type == "custom" and entry.data.get("customType") == LONG_TASK_REF_ENTRY:
                ref = LongTaskRef.from_dict(entry.data)
                try:
                    state = self.long_task_store.load_state(ref.task_id)
                except (OSError, ValueError, json.JSONDecodeError):
                    logger.warning("long_task_state_load_failed task_id=%s", ref.task_id, exc_info=True)
                    state = LongTaskState(objective=ref.summary, status=ref.status, task_id=ref.task_id, updated_at=ref.updated_at)
            elif entry.type == "custom" and entry.data.get("customType") == LONG_TASK_STATE_ENTRY:
                state = LongTaskState.from_dict(entry.data)
        return state

    def _runtime_context(self) -> str | None:
        contexts = [context for context in [self.long_task_state.runtime_context(), self.todo_state.runtime_context()] if context]
        if self.long_task_state.is_active and not self.todo_state.items:
            contexts.append(
                "Active sustained goal has no todo list yet. For long-running work, create a todo list before substantial execution. "
                "If the next step is brief discovery needed to form the todo list, do that first, then call update_todo."
            )
        if not contexts:
            return None
        return "\n\n".join(contexts)
