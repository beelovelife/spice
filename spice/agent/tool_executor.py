"""Tool preflight, ordered concurrency, timeout, and failure handling."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from spice.agent.events import AgentEvent, ToolExecutionEndEvent, ToolExecutionStartEvent
from spice.agent.logging_config import get_logger
from spice.agent.tool_results import build_tool_result_metadata
from spice.extensions.manager import ExtensionEvent, ExtensionManager
from spice.llm.error_safety import public_exception_message
from spice.llm.messages import Message, ToolCall
from spice.sandbox.base import ExecutionEnvironment
from spice.sandbox.policy import WorkspacePolicy
from spice.tools.base import (
    ConfirmFn,
    FatalToolError,
    RecoverableToolError,
    Tool,
    ToolContext,
    ToolResult,
    fatal_tool_error,
    tool_error,
)
from spice.tools.file_state import FileStateStore
from spice.tools.tool_registry import ToolCallError, ToolRegistry

logger = get_logger(__name__)


@dataclass
class PreparedToolCall:
    index: int
    call: ToolCall
    tool: Tool | None = None
    immediate_result: ToolResult | None = None


@dataclass
class ToolCallOutcome:
    prepared: PreparedToolCall
    result: ToolResult
    duration_ms: int


@dataclass
class ToolExecutionState:
    fatal_result: ToolResult | None = None
    fatal_tool_name: str | None = None


async def execute_tool_calls(
    *,
    round_index: int,
    calls: list[ToolCall],
    registry: ToolRegistry,
    messages: list[Message],
    model_messages: list[Message],
    cwd: Path,
    workspace: WorkspacePolicy,
    environment: ExecutionEnvironment,
    confirm: ConfirmFn | None,
    extensions: ExtensionManager | None,
    file_states: FileStateStore | None,
    subagent_manager,
    max_concurrency: int,
    default_timeout_seconds: float,
    state: ToolExecutionState,
) -> AsyncIterator[AgentEvent]:
    prepared = await _preflight(calls, registry=registry, confirm=confirm, extensions=extensions)
    fatal_preflight = next(
        (item for item in prepared if item.immediate_result and item.immediate_result.disposition == "fatal"),
        None,
    )
    if fatal_preflight is not None:
        outcomes = _preflight_fatal_outcomes(prepared, fatal_preflight)
        async for event in _emit_and_persist(outcomes, messages, model_messages, extensions):
            yield event
        state.fatal_result = fatal_preflight.immediate_result
        state.fatal_tool_name = fatal_preflight.call.name
        return

    batches = _build_batches(prepared)
    for batch_index, batch in enumerate(batches):
        parallel = len(batch) > 1 and all(item.tool and item.tool.concurrency == "parallel" for item in batch)
        logger.info(
            "tool_batch_start round=%d batch=%d mode=%s calls=%d limit=%d",
            round_index,
            batch_index + 1,
            "parallel" if parallel else "serial",
            len(batch),
            max_concurrency,
        )
        if parallel:
            outcomes = []
            queue: asyncio.Queue[tuple[str, ToolExecutionStartEvent | ToolCallOutcome]] = asyncio.Queue()
            semaphore = asyncio.Semaphore(max(max_concurrency, 1))
            started_ids: set[str] = set()
            batch_fatal = False

            async def worker(item: PreparedToolCall) -> None:
                async with semaphore:
                    await queue.put(("event", ToolExecutionStartEvent(item.call.id, item.call.name, item.call.arguments)))
                    outcome = await _execute_one(
                        item,
                        round_index=round_index,
                        cwd=cwd,
                        workspace=workspace,
                        environment=environment,
                        confirm=confirm,
                        file_states=file_states,
                        subagent_manager=subagent_manager,
                        default_timeout_seconds=default_timeout_seconds,
                    )
                    await queue.put(("done", outcome))

            tasks = [asyncio.create_task(worker(item)) for item in batch]
            try:
                while len(outcomes) < len(batch):
                    _kind, payload = await queue.get()
                    if isinstance(payload, ToolExecutionStartEvent):
                        started_ids.add(payload.tool_call_id)
                        yield payload
                    else:
                        outcome = payload
                        outcomes.append(outcome)
                        yield ToolExecutionEndEvent(
                            outcome.prepared.call.id,
                            outcome.prepared.call.name,
                            outcome.result,
                        )
                        if outcome.result.disposition == "fatal":
                            batch_fatal = True
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                            completed_ids = {item.prepared.call.id for item in outcomes}
                            while not queue.empty():
                                _queued_kind, queued_payload = queue.get_nowait()
                                if isinstance(queued_payload, ToolExecutionStartEvent):
                                    started_ids.add(queued_payload.tool_call_id)
                                    yield queued_payload
                                    continue
                                queued_outcome = queued_payload
                                queued_id = queued_outcome.prepared.call.id
                                if queued_id in completed_ids:
                                    continue
                                outcomes.append(queued_outcome)
                                completed_ids.add(queued_id)
                                yield ToolExecutionEndEvent(
                                    queued_id,
                                    queued_outcome.prepared.call.name,
                                    queued_outcome.result,
                                )
                            for item in batch:
                                if item.call.id in completed_ids:
                                    continue
                                skipped = _skipped_outcome(item)
                                if item.call.id not in started_ids:
                                    yield ToolExecutionStartEvent(item.call.id, item.call.name, item.call.arguments)
                                yield ToolExecutionEndEvent(item.call.id, item.call.name, skipped.result)
                                outcomes.append(skipped)
                            break
                if not batch_fatal:
                    await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            outcomes.sort(key=lambda item: item.prepared.index)
            await _persist_outcomes(outcomes, messages, model_messages, extensions)
        else:
            outcomes = []
            for item in batch:
                yield ToolExecutionStartEvent(item.call.id, item.call.name, item.call.arguments)
                outcome = await _execute_one(
                    item,
                    round_index=round_index,
                    cwd=cwd,
                    workspace=workspace,
                    environment=environment,
                    confirm=confirm,
                    file_states=file_states,
                    subagent_manager=subagent_manager,
                    default_timeout_seconds=default_timeout_seconds,
                )
                outcomes.append(outcome)
                yield ToolExecutionEndEvent(item.call.id, item.call.name, outcome.result)
            await _persist_outcomes(outcomes, messages, model_messages, extensions)

        fatal = next((outcome for outcome in outcomes if outcome.result.disposition == "fatal"), None)
        if fatal is not None:
            state.fatal_result = fatal.result
            state.fatal_tool_name = fatal.prepared.call.name
            remaining = [item for later in batches[batch_index + 1 :] for item in later]
            skipped = [_skipped_outcome(item) for item in remaining]
            async for event in _emit_and_persist(skipped, messages, model_messages, extensions):
                yield event
            return


async def _preflight(
    calls: list[ToolCall],
    *,
    registry: ToolRegistry,
    confirm: ConfirmFn | None,
    extensions: ExtensionManager | None,
) -> list[PreparedToolCall]:
    prepared: list[PreparedToolCall] = []
    fatal_seen = False
    for index, call in enumerate(calls):
        if fatal_seen:
            prepared.append(PreparedToolCall(index, call, immediate_result=_skipped_result()))
            continue
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
                result = fatal_tool_error(
                    extension_event.block_reason or f"Tool blocked by extension: {call.name}",
                    code="extension_blocked",
                )
                prepared.append(PreparedToolCall(index, call, immediate_result=result))
                fatal_seen = True
                continue
        plan = registry.prepare_call(call.name, call.arguments)
        if isinstance(plan, ToolCallError):
            prepared.append(
                PreparedToolCall(
                    index,
                    call,
                    immediate_result=tool_error(plan.message, {"errors": plan.errors}, code="invalid_tool_call"),
                )
            )
            continue
        call.arguments = plan.arguments
        if plan.tool.requires_confirmation:
            if confirm is None:
                result = fatal_tool_error(
                    f"Tool requires confirmation but no confirmation callback is configured: {call.name}",
                    code="confirmation_unavailable",
                )
                prepared.append(PreparedToolCall(index, call, tool=plan.tool, immediate_result=result))
                fatal_seen = True
                continue
            if not await confirm(plan.tool.name, call.arguments):
                result = fatal_tool_error(f"Tool denied by user: {call.name}", code="user_denied")
                prepared.append(PreparedToolCall(index, call, tool=plan.tool, immediate_result=result))
                fatal_seen = True
                continue
        prepared.append(PreparedToolCall(index, call, tool=plan.tool))
    return prepared


def _build_batches(prepared: list[PreparedToolCall]) -> list[list[PreparedToolCall]]:
    batches: list[list[PreparedToolCall]] = []
    parallel_batch: list[PreparedToolCall] = []
    for item in prepared:
        is_parallel = item.immediate_result is None and item.tool is not None and item.tool.concurrency == "parallel"
        if is_parallel:
            parallel_batch.append(item)
            continue
        if parallel_batch:
            batches.append(parallel_batch)
            parallel_batch = []
        batches.append([item])
    if parallel_batch:
        batches.append(parallel_batch)
    return batches


async def _execute_one(
    prepared: PreparedToolCall,
    *,
    round_index: int,
    cwd: Path,
    workspace: WorkspacePolicy,
    environment: ExecutionEnvironment,
    confirm: ConfirmFn | None,
    file_states: FileStateStore | None,
    subagent_manager,
    default_timeout_seconds: float,
) -> ToolCallOutcome:
    call = prepared.call
    started = time.perf_counter()
    logger.debug(
        "tool_start round=%d name=%s id=%s argument_keys=%s",
        round_index,
        call.name,
        call.id,
        ",".join(sorted(call.arguments)) or "<none>",
    )
    if prepared.immediate_result is not None:
        result = prepared.immediate_result
    else:
        assert prepared.tool is not None
        timeout = prepared.tool.timeout_seconds or default_timeout_seconds
        if call.name == "bash" and call.arguments.get("timeout") is not None:
            timeout = min(timeout, float(call.arguments["timeout"]))
        timeout = min(timeout if timeout > 0 else default_timeout_seconds, 3600.0)
        try:
            async with asyncio.timeout(timeout):
                result = await prepared.tool.execute(
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
        except TimeoutError:
            result = tool_error(f"Tool timed out after {timeout:g}s: {call.name}", code="tool_timeout")
        except RecoverableToolError as exc:
            result = tool_error(str(exc), exc.details, code=exc.code)
        except FatalToolError as exc:
            result = fatal_tool_error(str(exc), exc.details, code=exc.code)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("tool_exception round=%d name=%s id=%s", round_index, call.name, call.id)
            result = fatal_tool_error(
                public_exception_message(exc, prefix="Tool failed"),
                code="unexpected_tool_error",
            )
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.debug(
        "tool_result round=%d name=%s id=%s content_chars=%d detail_keys=%s",
        round_index,
        call.name,
        call.id,
        len(result.content),
        ",".join(sorted(result.details)) or "<none>",
    )
    logger.info(
        "tool_end round=%d name=%s id=%s is_error=%s disposition=%s duration_ms=%d",
        round_index,
        call.name,
        call.id,
        result.is_error,
        result.disposition,
        duration_ms,
    )
    return ToolCallOutcome(prepared, result, duration_ms)


async def _emit_and_persist(
    outcomes: list[ToolCallOutcome],
    messages: list[Message],
    model_messages: list[Message],
    extensions: ExtensionManager | None,
) -> AsyncIterator[AgentEvent]:
    for outcome in outcomes:
        call = outcome.prepared.call
        yield ToolExecutionStartEvent(call.id, call.name, call.arguments)
        yield ToolExecutionEndEvent(call.id, call.name, outcome.result)
    await _persist_outcomes(outcomes, messages, model_messages, extensions)


async def _persist_outcomes(
    outcomes: list[ToolCallOutcome],
    messages: list[Message],
    model_messages: list[Message],
    extensions: ExtensionManager | None,
) -> None:
    for outcome in sorted(outcomes, key=lambda item: item.prepared.index):
        call = outcome.prepared.call
        result = outcome.result
        metadata = build_tool_result_metadata(call.name, call.arguments, result)
        if result.full_content is not None:
            metadata["_full_tool_output"] = result.full_content
        message = Message(
            role="tool",
            content=result.content,
            tool_call_id=call.id,
            name=call.name,
            is_error=result.is_error,
            metadata=metadata,
        )
        messages.append(message)
        model_messages.append(message)
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


def _preflight_fatal_outcomes(
    prepared: list[PreparedToolCall],
    fatal: PreparedToolCall,
) -> list[ToolCallOutcome]:
    outcomes: list[ToolCallOutcome] = []
    for item in prepared:
        if item is fatal:
            result = item.immediate_result or fatal_tool_error("Tool preflight failed.", code="preflight_failed")
        elif item.immediate_result is not None and item.immediate_result.error_code == "invalid_tool_call":
            result = item.immediate_result
        else:
            result = _skipped_result()
        outcomes.append(ToolCallOutcome(item, result, 0))
    return outcomes


def _skipped_outcome(item: PreparedToolCall) -> ToolCallOutcome:
    return ToolCallOutcome(item, _skipped_result(), 0)


def _skipped_result() -> ToolResult:
    return tool_error(
        "Skipped because another tool call ended the current turn.",
        code="tool_batch_cancelled",
    )
