"""Agent loop: stream model output, run tools, continue until final text."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from spice.agent.debug_trace import (
    trace_round_end,
    trace_round_start,
    trace_tool_end,
    trace_tool_start,
    trace_turn_end,
    trace_turn_start,
)
from spice.agent.events import (
    AgentErrorEvent,
    AgentEvent,
    AssistantMessageEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.agent.logging_config import get_logger
from spice.agent.tool_results import build_tool_result_metadata
from spice.extensions.manager import ExtensionEvent, ExtensionManager
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.llm.error_safety import public_exception_message
from spice.llm.stream import stream_model
from spice.llm.types import Done, StreamError, ModelRequestOptions, TextDelta, ToolCallEvent
from spice.tools.base import ConfirmFn, Tool, ToolContext, tool_error
from spice.tools.file_state import FileStateStore
from spice.tools.tool_registry import ToolCallError, ToolRegistry
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
    trace_turn_start(session_label=session_label, prompt=prompt, message_count=len(messages), tool_count=len(tools))
    yield TurnStartEvent(prompt=prompt)
    messages.append(Message(role="user", content=prompt))
    model_messages = list(messages)
    if runtime_context and runtime_context.strip():
        model_messages.insert(-1, Message(role="system", content=runtime_context.strip()))
    tool_registry = ToolRegistry(tools)
    schemas = tool_registry.schemas()
    total_text_chars = 0
    total_tool_calls = 0
    rounds_run = 0

    for _round in range(max_tool_rounds):
        round_index = _round + 1
        rounds_run = round_index
        round_started = time.perf_counter()
        logger.info("model_stream_start round=%d message_count=%d", round_index, len(model_messages))
        trace_round_start(round_index, message_count=len(model_messages))
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None

        async for event in stream_model(model, model_messages, schemas, options):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                yield TextDeltaEvent(event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(ToolCall(id=event.id or uuid4().hex[:8], name=event.name, arguments=event.arguments))
            elif isinstance(event, StreamError):
                logger.error("model_stream_error round=%d error=%s", round_index, event.error)
                assistant_text = "".join(text_parts)
                if assistant_text:
                    assistant_message = Message(
                        role="assistant",
                        content=assistant_text,
                        provider=model.provider,
                        model=model.id,
                    )
                    messages.append(assistant_message)
                    model_messages.append(assistant_message)
                    yield AssistantMessageEvent(text=assistant_text, tool_calls=[])
                trace_turn_end(rounds=round_index, text_chars=total_text_chars + len(assistant_text), tool_calls=total_tool_calls)
                yield AgentErrorEvent(event.error)
                yield TurnEndEvent(text=assistant_text, stop_reason="error")
                return
            elif isinstance(event, Done):
                finish_reason = event.finish_reason
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
        trace_round_end(
            round_index,
            duration_ms=duration_ms,
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        assistant_message = Message(
            role="assistant",
            content=assistant_text,
            tool_calls=tool_calls,
            provider=model.provider,
            model=model.id,
        )
        messages.append(assistant_message)
        model_messages.append(assistant_message)
        yield AssistantMessageEvent(text=assistant_text, tool_calls=tool_calls)

        if not tool_calls:
            logger.info("turn_end text_chars=%d rounds=%d", len(assistant_text), round_index)
            trace_turn_end(rounds=round_index, text_chars=total_text_chars, tool_calls=total_tool_calls)
            yield TurnEndEvent(text=assistant_text, stop_reason=finish_reason or "stop")
            return

        for call in tool_calls:
            if extensions:
                extension_event = await extensions.emit(
                    "tool_call_start",
                    ExtensionEvent(
                        type="tool_call_start",
                        data={"tool_name": call.name, "arguments": dict(call.arguments), "tool_call_id": call.id},
                    ),
                )
                call.arguments = dict(extension_event.data.get("arguments") or call.arguments)
                if extension_event.blocked:
                    result = tool_error(extension_event.block_reason or f"Tool blocked by extension: {call.name}")
                    trace_tool_start(round_index, call)
                    trace_tool_end(round_index, call, result, duration_ms=0)
                    blocked_message = Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=call.id,
                        name=call.name,
                        is_error=result.is_error,
                        metadata=build_tool_result_metadata(call.name, call.arguments, result),
                    )
                    messages.append(blocked_message)
                    model_messages.append(blocked_message)
                    yield ToolExecutionStartEvent(tool_call_id=call.id, tool_name=call.name, args=call.arguments)
                    yield ToolExecutionEndEvent(tool_call_id=call.id, tool_name=call.name, result=result)
                    continue
            yield ToolExecutionStartEvent(tool_call_id=call.id, tool_name=call.name, args=call.arguments)
            tool_started = time.perf_counter()
            logger.info("tool_start round=%d name=%s id=%s", round_index, call.name, call.id)
            trace_tool_start(round_index, call)
            plan = tool_registry.prepare_call(call.name, call.arguments)
            if isinstance(plan, ToolCallError):
                result = tool_error(plan.message, {"errors": plan.errors})
            else:
                tool = plan.tool
                call.arguments = plan.arguments
                if tool.requires_confirmation and confirm is None:
                    result = tool_error(f"Tool requires confirmation but no confirmation callback is configured: {call.name}")
                elif tool.requires_confirmation and not await confirm(tool.name, call.arguments):
                    result = tool_error(f"Tool denied by user: {call.name}")
                else:
                    try:
                        result = await tool.execute(
                            call.arguments,
                            ToolContext(
                                cwd=cwd,
                                workspace=workspace,
                                environment=environment,
                                confirm=confirm,
                                file_states=file_states,
                                subagent_manager=subagent_manager,
                            ),
                        )
                    except Exception as exc:
                        logger.exception("tool_exception round=%d name=%s id=%s", round_index, call.name, call.id)
                        result = tool_error(public_exception_message(exc, prefix="Tool failed"))
            tool_duration_ms = int((time.perf_counter() - tool_started) * 1000)
            logger.info(
                "tool_end round=%d name=%s id=%s is_error=%s duration_ms=%d content_chars=%d",
                round_index,
                call.name,
                call.id,
                result.is_error,
                tool_duration_ms,
                len(result.content),
            )
            trace_tool_end(round_index, call, result, duration_ms=tool_duration_ms)
            tool_message = Message(
                role="tool",
                content=result.content,
                tool_call_id=call.id,
                name=call.name,
                is_error=result.is_error,
                metadata=build_tool_result_metadata(call.name, call.arguments, result),
            )
            messages.append(tool_message)
            model_messages.append(tool_message)
            yield ToolExecutionEndEvent(tool_call_id=call.id, tool_name=call.name, result=result)
            if extensions:
                await extensions.emit(
                    "tool_call_end",
                    ExtensionEvent(
                        type="tool_call_end",
                        data={
                            "tool_name": call.name,
                            "arguments": dict(call.arguments),
                            "tool_call_id": call.id,
                            "result": result,
                        },
                    ),
                )

    logger.error("turn_stopped max_tool_rounds=%d", max_tool_rounds)
    trace_turn_end(rounds=rounds_run, text_chars=total_text_chars, tool_calls=total_tool_calls)
    yield TurnEndEvent(text="", stop_reason="max_tool_rounds")
    yield AgentErrorEvent(f"Stopped after {max_tool_rounds} tool rounds.")
