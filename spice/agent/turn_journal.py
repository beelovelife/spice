"""Persist execution facts separately from ordered model conversation history.

Assistant calls are written before execution. Completed results are checkpointed
as they arrive; ordered tool messages are committed at the round boundary. A
restart can recover known results without replaying tools. An absent result means
unknown side effects, never permission to run the call again.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from spice.agent.events import AgentErrorEvent, ToolExecutionEndEvent, TurnEndEvent
from spice.agent.session_history import AGENT_ERROR_ENTRY
from spice.agent.turn_outcome import TURN_OUTCOME_ENTRY
from spice.agent.sessions import SessionStoreProtocol, message_from_dict, message_to_dict
from spice.agent.tool_results import prepare_tool_message_for_session, tool_result_message
from spice.llm.messages import Message, ToolCall
from spice.llm.error_safety import sanitize_error_text
from spice.tools.base import tool_error

TOOL_CHECKPOINT_ENTRY = "tool_execution_checkpoint"


class TurnJournal:
    """One writer and an incrementally advanced cursor for a single prompt."""

    def __init__(
        self, store: SessionStoreProtocol, session_id: str, *, cwd: Path,
        cursor: int, provider: str, model: str,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.cwd = cwd
        self.cursor = cursor
        self.provider = provider
        self.model = model
        self.assistant_entry_id: str | None = None
        self.calls: dict[str, ToolCall] = {}
        self._outcome_written = False

    def flush(self, messages: list[Message]) -> str | None:
        parent_id = self.store.info(self.session_id).leaf_id
        while self.cursor < len(messages):
            message = messages[self.cursor]
            if message.role == "assistant":
                message.provider = message.provider or self.provider
                message.model = message.model or self.model
            persisted = prepare_tool_message_for_session(
                message, cwd=self.cwd, session_id=self.session_id,
            )
            parent_id = self.store.append_message(self.session_id, persisted, parent_id=parent_id)
            # Advance after every successful append, including a partial flush.
            # A later write error must not duplicate the successful prefix.
            self.cursor += 1
            if message.role == "assistant":
                self.assistant_entry_id = parent_id
                self.calls = {call.id: call for call in message.tool_calls}
        return parent_id

    def checkpoint(self, event: ToolExecutionEndEvent) -> None:
        call = self.calls.get(event.tool_call_id)
        if call is None or self.assistant_entry_id is None:
            raise RuntimeError("Cannot checkpoint a result without its persisted assistant call.")
        message = prepare_tool_message_for_session(
            tool_result_message(call, event.result), cwd=self.cwd, session_id=self.session_id,
        )
        self.store.append_custom(self.session_id, {
            "customType": TOOL_CHECKPOINT_ENTRY,
            "assistant_entry_id": self.assistant_entry_id,
            "message": message_to_dict(message),
        })

    def rebase(self, messages: list[Message]) -> None:
        """Compaction replaced an already persisted conversation projection."""
        self.cursor = len(messages)
        self.assistant_entry_id = None
        self.calls.clear()

    def record_error(self, event: AgentErrorEvent, messages: list[Message]) -> None:
        """Persist the error after its preceding messages, never as an LLM message."""
        self.flush(messages)
        self.store.append_custom(self.session_id, {
            "customType": AGENT_ERROR_ENTRY,
            "message": sanitize_error_text(event.message),
            "kind": event.kind,
        })

    def finish(self, event: TurnEndEvent) -> None:
        """Keep the outcome on the active branch without making it an LLM message."""
        if self._outcome_written:
            return
        self.store.append_custom(self.session_id, {
            "customType": TURN_OUTCOME_ENTRY, **asdict(event),
        })
        self._outcome_written = True


def interrupted_tool_message(call: ToolCall, *, code: str = "execution_interrupted") -> Message:
    return tool_result_message(call, tool_error(
        "Tool execution was interrupted before its result was recorded. Side effects may "
        "have occurred. It was not retried automatically; inspect the workspace or external "
        "system before deciding whether to retry.",
        code=code,
    ))


def append_interrupted_results(messages: list[Message], *, code: str = "execution_interrupted") -> None:
    """Close only the outstanding tail round; call IDs are scoped to a round."""
    pending: dict[str, ToolCall] = {}
    for message in messages:
        if message.role == "tool":
            if message.tool_call_id is not None:
                pending.pop(message.tool_call_id, None)
        else:
            pending = {call.id: call for call in message.tool_calls}
    messages.extend(interrupted_tool_message(call, code=code) for call in pending.values())


def recover_interrupted_tools(store: SessionStoreProtocol, session_id: str) -> int:
    """Recover the active branch's unfinished tail, idempotently, without execution.

    Checkpoints are associated with the assistant *entry*, not just a provider's
    tool-call ID (which can repeat across rounds or branches).
    """
    entries = store.path_entries(session_id)
    assistant_id: str | None = None
    pending: dict[str, ToolCall] = {}
    completed: dict[str, Message] = {}
    for entry in entries:
        if entry.type == "message":
            message = message_from_dict(entry.data.get("message") or {})
            if message.role == "tool":
                if message.tool_call_id is not None:
                    pending.pop(message.tool_call_id, None)
            else:
                assistant_id = entry.id if message.role == "assistant" else None
                pending = {call.id: call for call in message.tool_calls}
                completed.clear()
        elif (
            entry.type == "custom"
            and entry.data.get("customType") == TOOL_CHECKPOINT_ENTRY
            and entry.data.get("assistant_entry_id") == assistant_id
        ):
            message = message_from_dict(entry.data.get("message") or {})
            call = pending.get(message.tool_call_id or "")
            if call is not None and message.role == "tool" and message.name == call.name:
                completed[call.id] = message
    for call in pending.values():
        store.append_message(session_id, completed.get(call.id) or interrupted_tool_message(call))
    return len(pending)
