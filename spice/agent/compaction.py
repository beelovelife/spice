"""Context compaction helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.stream import stream_model
from spice.llm.types import ModelRequestOptions, StreamError, TextDelta

DEFAULT_RESERVE_TOKENS = 16_384
DEFAULT_KEEP_RECENT_TOKENS = 20_000
MIN_CONTEXT_FRACTION_AFTER_RESERVE = 0.35
MIN_MESSAGES_TO_COMPACT = 8


@dataclass(frozen=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    min_messages: int = MIN_MESSAGES_TO_COMPACT


@dataclass(frozen=True)
class CompactionCheck:
    estimated_tokens: int
    threshold_tokens: int | None
    reserve_tokens: int
    should_compact: bool
    reason: str = ""


@dataclass(frozen=True)
class CompactionPlan:
    first_kept_entry_id: str
    messages_to_summarize: list[Message]
    kept_messages: list[Message]
    previous_summary: str | None
    tokens_before: int


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    tokens_after: int
    reason: Literal["manual", "auto"]
    focus: str | None = None


class CompactionError(RuntimeError):
    """Raised when compaction cannot safely run."""


def effective_reserve_tokens(model: Model, settings: CompactionSettings = CompactionSettings()) -> int:
    """Return a reserve that leaves enough space for the retained context."""
    reserve = max(settings.reserve_tokens, model.output_tokens * 2)
    if model.context_window <= 0:
        return reserve
    max_reserve = int(model.context_window * (1 - MIN_CONTEXT_FRACTION_AFTER_RESERVE))
    return max(0, min(reserve, max_reserve))


def check_compaction_needed(
    messages: list[Message],
    model: Model,
    settings: CompactionSettings = CompactionSettings(),
) -> CompactionCheck:
    reserve = effective_reserve_tokens(model, settings)
    estimated = estimate_messages_tokens(messages)
    if not settings.enabled:
        return CompactionCheck(estimated, None, reserve, False, "disabled")
    if model.context_window <= 0:
        return CompactionCheck(estimated, None, reserve, False, "missing context_window")
    threshold = max(0, model.context_window - reserve)
    return CompactionCheck(
        estimated_tokens=estimated,
        threshold_tokens=threshold,
        reserve_tokens=reserve,
        should_compact=estimated > threshold,
        reason="over threshold" if estimated > threshold else "under threshold",
    )


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_message_tokens(message: Message) -> int:
    chars = len(message.role) + len(message.content or "")
    if message.name:
        chars += len(message.name)
    if message.tool_call_id:
        chars += len(message.tool_call_id)
    for call in message.tool_calls:
        chars += len(call.id) + len(call.name) + len(_safe_json(call.arguments))
    return max(1, (chars + 3) // 4)


def prepare_compaction(
    path_entries: list[Any],
    settings: CompactionSettings = CompactionSettings(),
) -> CompactionPlan | None:
    message_entries = [entry for entry in path_entries if entry.type == "message"]
    if len(message_entries) < settings.min_messages:
        return None
    if path_entries and path_entries[-1].type == "compaction":
        return None

    previous_summary: str | None = None
    boundary_start = 0
    for index in range(len(path_entries) - 1, -1, -1):
        entry = path_entries[index]
        if entry.type == "compaction":
            previous_summary = str(entry.data.get("summary") or "")
            first_kept_id = entry.data.get("first_kept_entry_id")
            if first_kept_id:
                for kept_index, candidate in enumerate(path_entries):
                    if candidate.id == first_kept_id:
                        boundary_start = kept_index
                        break
            else:
                boundary_start = index + 1
            break

    compactable = path_entries[boundary_start:]
    compactable_messages = [
        (index + boundary_start, _message_from_entry(entry))
        for index, entry in enumerate(compactable)
        if entry.type == "message"
    ]
    compactable_messages = [(index, message) for index, message in compactable_messages if message is not None]
    if len(compactable_messages) < settings.min_messages:
        return None

    recent_tokens = 0
    first_kept_path_index = compactable_messages[-1][0]
    for path_index, message in reversed(compactable_messages):
        recent_tokens += estimate_message_tokens(message)
        first_kept_path_index = path_index
        if recent_tokens >= settings.keep_recent_tokens:
            break
    first_kept_path_index = _align_first_kept_to_tool_round(path_entries, first_kept_path_index, boundary_start)

    summarize_messages: list[Message] = []
    kept_messages: list[Message] = []
    first_kept_entry_id = path_entries[first_kept_path_index].id
    for path_index, message in compactable_messages:
        if path_index < first_kept_path_index:
            summarize_messages.append(message)
        else:
            kept_messages.append(message)

    if not summarize_messages or not kept_messages:
        return None

    all_messages = [
        message
        for _path_index, message in compactable_messages
    ]
    return CompactionPlan(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=summarize_messages,
        kept_messages=kept_messages,
        previous_summary=previous_summary or None,
        tokens_before=estimate_messages_tokens(all_messages),
    )


async def generate_summary(
    *,
    plan: CompactionPlan,
    model: Model,
    options: ModelRequestOptions,
    focus: str | None = None,
) -> str:
    prompt = _summary_prompt(plan, focus=focus)
    request_messages = [
        Message(
            role="system",
            content=(
                "You summarize prior conversation context for another coding agent. "
                "Do not continue the conversation. Only produce the requested summary."
            ),
        ),
        Message(role="user", content=prompt),
    ]
    summary_parts: list[str] = []
    async for event in stream_model(model, request_messages, [], options):
        if isinstance(event, TextDelta):
            summary_parts.append(event.text)
        elif isinstance(event, StreamError):
            raise CompactionError(event.error)
    summary = "".join(summary_parts).strip()
    if not summary:
        raise CompactionError("Compaction summary was empty.")
    return summary


def _summary_prompt(plan: CompactionPlan, *, focus: str | None) -> str:
    previous = f"\n<previous-summary>\n{plan.previous_summary}\n</previous-summary>\n" if plan.previous_summary else ""
    focus_text = f"\nFocus especially on: {focus}\n" if focus else ""
    return (
        "Summarize the conversation messages below into a compact checkpoint. "
        "Preserve exact file paths, function names, commands, errors, user constraints, "
        "decisions, current progress, and next steps. If a tool result contains JSON or "
        "structured data, summarize the structure and important values; do not emit partial JSON."
        f"{focus_text}{previous}\n"
        "Use this format:\n\n"
        "## Goal\n"
        "## Constraints & Preferences\n"
        "## Progress\n"
        "## Key Decisions\n"
        "## Files & Artifacts\n"
        "## Next Steps\n"
        "## Critical Context\n\n"
        "<conversation>\n"
        f"{_serialize_messages(plan.messages_to_summarize)}\n"
        "</conversation>"
    )


def _serialize_messages(messages: list[Message]) -> str:
    chunks: list[str] = []
    for message in messages:
        label = message.role if not message.name else f"{message.role}:{message.name}"
        chunks.append(f"[{label}]\n{message.content or ''}")
        if message.tool_calls:
            calls = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ]
            chunks.append(f"[assistant tool_calls]\n{_safe_json(calls)}")
    return "\n\n".join(chunks)


def _message_from_entry(entry: Any) -> Message | None:
    from spice.agent.sessions import message_from_dict

    raw = entry.data.get("message")
    if not isinstance(raw, dict):
        return None
    return message_from_dict(raw)


def _align_first_kept_to_tool_round(path_entries: list[Any], index: int, boundary_start: int) -> int:
    while index > boundary_start:
        message = _message_from_entry(path_entries[index])
        if message is None or message.role != "tool" or not message.tool_call_id:
            return index
        assistant_index = _find_assistant_for_tool(path_entries, index, message.tool_call_id, boundary_start)
        if assistant_index is None:
            return index
        index = assistant_index
    return index


def _find_assistant_for_tool(path_entries: list[Any], tool_index: int, tool_call_id: str, boundary_start: int) -> int | None:
    for index in range(tool_index - 1, boundary_start - 1, -1):
        message = _message_from_entry(path_entries[index])
        if message is None:
            continue
        if message.role == "tool":
            continue
        if message.role == "assistant" and any(call.id == tool_call_id for call in message.tool_calls):
            return index
        return None
    return None


def _safe_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({"_unserializable": repr(value)}, ensure_ascii=False, sort_keys=True)
