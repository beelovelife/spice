"""Display history from active-branch facts, separate from model context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spice.agent.sessions import SessionEntry, message_from_dict
from spice.agent.turn_outcome import TURN_OUTCOME_ENTRY
from spice.llm.error_safety import sanitize_error_text
from spice.llm.messages import Message

AGENT_ERROR_ENTRY = "agent_error"


@dataclass(frozen=True)
class HistoryError:
    message: str
    kind: str
    timestamp: str

    @property
    def display_text(self) -> str:
        return f"历史错误：{self.message}"


def history_from_entries(entries: list[SessionEntry]) -> list[Message | HistoryError]:
    """Replay recorded errors in place, including before a later successful turn."""
    history: list[Message | HistoryError] = []
    error_recorded = False
    for entry in entries:
        if entry.type == "message":
            message = message_from_dict(entry.data.get("message") or {})
            if message.role != "system":
                history.append(message)
        elif entry.type == "custom":
            custom_type = entry.data.get("customType")
            if custom_type == AGENT_ERROR_ENTRY:
                history.append(HistoryError(
                    sanitize_error_text(str(entry.data.get("message") or "执行失败")),
                    str(entry.data.get("kind") or "error"), entry.timestamp,
                ))
                error_recorded = True
            elif custom_type == TURN_OUTCOME_ENTRY:
                if entry.data.get("status") == "failed" and not error_recorded:
                    reason = sanitize_error_text(str(entry.data.get("stop_reason") or "error"))
                    history.append(HistoryError(
                        f"该轮执行失败（{reason}）；旧会话记录未保存具体错误，请查看运行日志。",
                        "error", entry.timestamp,
                    ))
                error_recorded = False
    return history


def session_history(session: Any) -> list[Message | HistoryError]:
    """Shared history source for CLI/TUI, with an in-memory fallback for adapters."""
    store = getattr(session, "session_store", None)
    info = getattr(session, "session", None)
    if store is not None and info is not None:
        return history_from_entries(store.path_entries(info.id))
    return [message for message in getattr(session, "messages", ()) if message.role != "system"]
