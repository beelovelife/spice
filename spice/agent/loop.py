"""Agent loop: stream model output, run tools, continue until final text."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from spice.agent.events import (
    AgentErrorEvent,
    AgentEvent,
    AssistantMessageEvent,
    ModelFallbackEvent,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    RoundCompleteEvent,
    TextDeltaEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.agent.logging_config import get_logger
from spice.agent.tool_executor import ToolExecutionState, execute_tool_calls
from spice.extensions.manager import ExtensionManager
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy
from spice.llm.routing import ModelCandidate, ModelRoute
from spice.llm.stream import stream_model
from spice.llm.types import (
    Done,
    ModelFallbackNotice,
    ModelRequestOptions,
    ModelRetryNotice,
    ReasoningDelta,
    StreamError,
    TextDelta,
    ToolCallEvent,
)
from spice.llm.usage import TokenUsage, make_usage_record
from spice.tools.base import ConfirmFn, Tool
from spice.tools.file_state import FileStateStore
from spice.tools.tool_registry import ToolRegistry
from spice.sandbox.base import ExecutionEnvironment
from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.sandbox.policy import WorkspacePolicy

MAX_TOOL_ROUNDS = 30
logger = get_logger(__name__)


async def run_turn(
    *,
    prompt: str,
    messages: list[Message],
    model: Model,
    tools: list[Tool],
    options: ModelRequestOptions,
    cwd: Path,
    workspace: WorkspacePolicy | None = None,
    environment: ExecutionEnvironment | None = None,
    confirm: ConfirmFn | None,
    extensions: ExtensionManager | None = None,
    file_states: FileStateStore | None = None,
    subagent_manager=None,
    session_label: str | None = None,
    runtime_context: str | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    model_route: ModelRoute | None = None,
    tools_settings: dict | None = None,
) -> AsyncIterator[AgentEvent]:
    logger.info(
        "turn_start model=%s/%s message_count=%d tool_count=%d cwd=%s",
        model.provider,
        model.id,
        len(messages),
        len(tools),
        cwd,
    )
    workspace = workspace or create_workspace_policy(None, cwd=cwd)
    environment = environment or create_environment(None, cwd=cwd)
    logger.debug(
        "turn_context session=%s messages=%d tools=%d prompt_chars=%d",
        session_label or "<unknown>",
        len(messages),
        len(tools),
        len(prompt),
    )
    yield TurnStartEvent(prompt=prompt)
    messages.append(Message(role="user", content=prompt))
    model_messages = list(messages)
    if runtime_context and runtime_context.strip():
        model_messages.insert(
            -1, Message(role="system", content=runtime_context.strip())
        )
    tool_registry = ToolRegistry(tools)
    schemas = tool_registry.schemas()
    route = model_route or ModelRoute(
        [ModelCandidate(model.profile_key or model.id, model, options)],
        retry_policy=ModelRetryPolicy(enabled=False, max_attempts=1),
        fallback_enabled=False,
        stream_factory=stream_model,
    )
    tools_settings = tools_settings or {}
    max_tool_concurrency = min(
        max(int(tools_settings.get("max_concurrency", 4)), 1), 16
    )
    default_tool_timeout = max(
        float(tools_settings.get("default_timeout_seconds", 120)), 1.0
    )
    total_text_chars = 0
    total_tool_calls = 0

    for _round in range(max_tool_rounds):
        round_index = _round + 1
        round_started = time.perf_counter()
        logger.info(
            "model_stream_start round=%d message_count=%d",
            round_index,
            len(model_messages),
        )
        logger.debug(
            "model_round_start round=%d messages=%d schemas=%d",
            round_index,
            len(model_messages),
            len(schemas),
        )
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None
        token_usage: TokenUsage | None = None
        done_received = False

        async for event in route.stream(model_messages, schemas):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                yield TextDeltaEvent(event.text)
            elif isinstance(event, ReasoningDelta):
                if event.kind == "reasoning":
                    reasoning_parts.append(event.text)
                yield ReasoningDeltaEvent(event.text, kind=event.kind)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(
                    ToolCall(
                        id=event.id or uuid4().hex[:8],
                        name=event.name,
                        arguments=event.arguments,
                    )
                )
            elif isinstance(event, ModelRetryNotice):
                yield ModelRetryEvent(
                    provider=event.provider,
                    model=event.model,
                    failed_attempt=event.failed_attempt,
                    next_attempt=event.next_attempt,
                    max_attempts=event.max_attempts,
                    delay_seconds=event.delay_seconds,
                    error=event.error,
                )
            elif isinstance(event, ModelFallbackNotice):
                yield ModelFallbackEvent(
                    from_profile=event.from_profile,
                    from_provider=event.from_provider,
                    from_model=event.from_model,
                    to_profile=event.to_profile,
                    to_provider=event.to_provider,
                    to_model=event.to_model,
                    reason=event.reason,
                    fallback_index=event.fallback_index,
                    fallback_count=event.fallback_count,
                )
            elif isinstance(event, StreamError):
                logger.error(
                    "model_stream_error round=%d error=%s", round_index, event.error
                )
                assistant_text = "".join(text_parts)
                if assistant_text:
                    failed_model = route.actual.model
                    assistant_message = Message(
                        role="assistant",
                        content=assistant_text,
                        provider=failed_model.provider,
                        model=failed_model.id,
                    )
                    messages.append(assistant_message)
                    model_messages.append(assistant_message)
                    yield AssistantMessageEvent(text=assistant_text, tool_calls=[])
                yield AgentErrorEvent(event.error)
                yield TurnEndEvent(text=assistant_text, stop_reason="error")
                return
            elif isinstance(event, Done):
                finish_reason = event.finish_reason
                token_usage = event.usage
                done_received = True
                break

        assistant_text = "".join(text_parts)
        duration_ms = int((time.perf_counter() - round_started) * 1000)
        total_text_chars += len(assistant_text)
        total_tool_calls += len(tool_calls)
        logger.info(
            "model_stream_end round=%d duration_ms=%d text_chars=%d tool_calls=%d",
            round_index,
            duration_ms,
            len(assistant_text),
            len(tool_calls),
        )
        logger.debug(
            "model_round_result round=%d finish_reason=%s output_chars=%d tool_names=%s",
            round_index,
            finish_reason or "<unknown>",
            len(assistant_text),
            ",".join(call.name for call in tool_calls) or "<none>",
        )
        actual_model = route.actual.model
        usage_record = (
            make_usage_record(actual_model, token_usage, duration_ms=duration_ms)
            if done_received
            else None
        )
        assistant_message = Message(
            role="assistant",
            content=assistant_text,
            tool_calls=tool_calls,
            provider=actual_model.provider,
            model=actual_model.id,
            metadata={"usage": usage_record.to_dict()}
            if usage_record is not None
            else {},
        )
        messages.append(assistant_message)
        model_assistant_message = assistant_message
        if reasoning_parts and tool_calls:
            model_assistant_message = Message(
                role="assistant",
                content=assistant_text,
                tool_calls=tool_calls,
                provider=actual_model.provider,
                model=actual_model.id,
                metadata={
                    **assistant_message.metadata,
                    "_reasoning_content": "".join(reasoning_parts),
                },
            )
        model_messages.append(model_assistant_message)
        yield AssistantMessageEvent(
            text=assistant_text, tool_calls=tool_calls, usage=usage_record
        )

        if not tool_calls:
            logger.info(
                "turn_end text_chars=%d rounds=%d", len(assistant_text), round_index
            )
            yield TurnEndEvent(text=assistant_text, stop_reason=finish_reason or "stop")
            return

        execution_state = ToolExecutionState()
        async for tool_event in execute_tool_calls(
            round_index=round_index,
            calls=tool_calls,
            registry=tool_registry,
            messages=messages,
            model_messages=model_messages,
            cwd=cwd,
            workspace=workspace,
            environment=environment,
            confirm=confirm,
            extensions=extensions,
            file_states=file_states,
            subagent_manager=subagent_manager,
            max_concurrency=max_tool_concurrency,
            default_timeout_seconds=default_tool_timeout,
            state=execution_state,
        ):
            yield tool_event
        yield RoundCompleteEvent(round_index=round_index)
        if execution_state.fatal_result is not None:
            message = execution_state.fatal_result.content
            logger.error(
                "turn_stopped fatal_tool=%s code=%s",
                execution_state.fatal_tool_name,
                execution_state.fatal_result.error_code,
            )
            yield AgentErrorEvent(message, kind="fatal_tool")
            yield TurnEndEvent(text=assistant_text, stop_reason="fatal_tool_error")
            return

    logger.error("turn_stopped max_tool_rounds=%d", max_tool_rounds)
    yield TurnEndEvent(text="", stop_reason="max_tool_rounds")
    yield AgentErrorEvent(f"Stopped after {max_tool_rounds} tool rounds.")
