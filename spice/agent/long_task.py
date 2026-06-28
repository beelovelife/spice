"""Durable state for sustained tasks/goals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from spice.llm.config import CONFIG_DIR

LONG_TASK_STATE_ENTRY = "long_task_state"
LONG_TASK_REF_ENTRY = "long_task_ref"
DEFAULT_TASKS_DIR = CONFIG_DIR / "tasks"


@dataclass
class LongTaskRef:
    task_id: str = ""
    status: str = "idle"
    summary: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "customType": LONG_TASK_REF_ENTRY,
            "taskId": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongTaskRef":
        return cls(
            task_id=str(data.get("taskId") or ""),
            status=str(data.get("status") or "idle"),
            summary=str(data.get("summary") or ""),
            updated_at=str(data.get("updatedAt") or ""),
        )


@dataclass
class LongTaskState:
    objective: str = ""
    status: str = "idle"
    notes: list[str] = field(default_factory=list)
    continuation_rounds: int = 0
    max_continuation_rounds: int = 12
    needs_user_attention: bool = False
    last_stop_reason: str = ""
    completion_candidate: bool = False
    task_id: str = ""
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == "active" and bool(self.objective.strip())

    @property
    def remaining_continuations(self) -> int:
        return max(0, self.max_continuation_rounds - self.continuation_rounds)

    @property
    def can_continue(self) -> bool:
        return self.is_active and not self.needs_user_attention and self.remaining_continuations > 0

    def start(
        self,
        objective: str,
        *,
        note: str = "",
        max_continuation_rounds: int | None = None,
        session_id: str = "",
    ) -> None:
        now = _now()
        self.objective = objective.strip()
        self.status = "active"
        self.session_id = session_id or self.session_id
        self.continuation_rounds = 0
        self.max_continuation_rounds = _clamp_positive_int(max_continuation_rounds, default=self.max_continuation_rounds)
        self.needs_user_attention = False
        self.last_stop_reason = ""
        self.completion_candidate = False
        self.created_at = self.created_at or now
        self.updated_at = now
        if note.strip():
            self.notes.append(note.strip())

    def complete(self, *, note: str = "") -> None:
        if note.strip():
            self.notes.append(note.strip())
        self.status = "completed"
        self.needs_user_attention = False
        self.completion_candidate = False
        self.updated_at = _now()

    def cancel(self, *, note: str = "") -> None:
        if note.strip():
            self.notes.append(note.strip())
        self.status = "cancelled"
        self.needs_user_attention = False
        self.updated_at = _now()

    def record_continuation(self, *, stop_reason: str = "") -> None:
        self.continuation_rounds += 1
        self.last_stop_reason = stop_reason.strip()
        if self.remaining_continuations <= 0:
            self.needs_user_attention = True
        self.updated_at = _now()

    def mark_needs_attention(self, *, reason: str) -> None:
        self.needs_user_attention = True
        self.last_stop_reason = reason.strip()
        self.updated_at = _now()

    def mark_completion_candidate(self) -> None:
        self.completion_candidate = True
        self.updated_at = _now()

    def runtime_context(self) -> str | None:
        if not self.is_active:
            return None
        lines = [
            "Active sustained goal:",
            self.objective,
            "",
            f"Task id: {self.task_id or '(session-local legacy task)'}",
            f"Continuation budget: {self.continuation_rounds}/{self.max_continuation_rounds} used, {self.remaining_continuations} remaining.",
            "Do not continue indefinitely. If the continuation budget is exhausted, or further progress needs a user decision, stop and ask the user before proceeding.",
            "",
            "Continue working toward this objective using available tools. If the objective is fully done and verified, call complete_long_task. If it is replaced or cancelled, update the sustained goal explicitly.",
        ]
        if self.needs_user_attention:
            lines.extend(
                [
                    "",
                    "This goal is marked as needing user attention. Explain the blocker or ask for confirmation before doing more work.",
                ]
            )
        if self.completion_candidate:
            lines.extend(
                [
                    "",
                    "All immediate work appears complete. Verify the objective, summarize evidence, then call complete_long_task if done.",
                ]
            )
        if self.last_stop_reason:
            lines.extend(["", f"Last stop reason: {self.last_stop_reason}"])
        if self.notes:
            lines.append("")
            lines.append("Goal notes:")
            lines.extend(f"- {note}" for note in self.notes[-5:])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "objective": self.objective,
            "sessionId": self.session_id,
            "status": self.status,
            "notes": list(self.notes),
            "continuationRounds": self.continuation_rounds,
            "maxContinuationRounds": self.max_continuation_rounds,
            "needsUserAttention": self.needs_user_attention,
            "lastStopReason": self.last_stop_reason,
            "completionCandidate": self.completion_candidate,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def to_session_state_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["customType"] = LONG_TASK_STATE_ENTRY
        return payload

    def to_ref(self) -> LongTaskRef:
        return LongTaskRef(
            task_id=self.task_id,
            status=self.status,
            summary=_summary(self.objective),
            updated_at=self.updated_at or _now(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongTaskState":
        raw_notes = data.get("notes")
        notes = [str(item) for item in raw_notes if str(item).strip()] if isinstance(raw_notes, list) else []
        status = str(data.get("status") or "idle")
        if status not in {"idle", "active", "paused", "completed", "cancelled"}:
            status = "idle"
        return cls(
            task_id=str(data.get("taskId") or ""),
            objective=str(data.get("objective") or ""),
            session_id=str(data.get("sessionId") or ""),
            status=status,
            notes=notes,
            continuation_rounds=_clamp_nonnegative_int(data.get("continuationRounds"), default=0),
            max_continuation_rounds=_clamp_positive_int(data.get("maxContinuationRounds"), default=12),
            needs_user_attention=bool(data.get("needsUserAttention")),
            last_stop_reason=str(data.get("lastStopReason") or ""),
            completion_candidate=bool(data.get("completionCandidate")),
            created_at=str(data.get("createdAt") or ""),
            updated_at=str(data.get("updatedAt") or ""),
        )


class LongTaskStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or DEFAULT_TASKS_DIR

    def create(
        self,
        *,
        objective: str,
        session_id: str,
        note: str = "",
        max_continuation_rounds: int | None = None,
    ) -> LongTaskState:
        task_id = _new_task_id()
        state = LongTaskState(task_id=task_id)
        state.start(
            objective,
            note=note,
            max_continuation_rounds=max_continuation_rounds,
            session_id=session_id,
        )
        self.save_state(state)
        self.write_checkpoint(state, todos=[], last_action="created")
        self.append_event(task_id, "created", {"objective": state.objective, "sessionId": session_id})
        return state

    def load_state(self, task_id: str) -> LongTaskState:
        path = self.task_dir(task_id) / "state.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid long task state: {path}")
        return LongTaskState.from_dict(data)

    def save_state(self, state: LongTaskState) -> None:
        state.updated_at = state.updated_at or _now()
        self._write_json(self.task_dir(state.task_id) / "state.json", state.to_dict())

    def write_checkpoint(
        self,
        state: LongTaskState,
        *,
        todos: list[dict[str, Any]],
        last_action: str,
        next_action: str = "",
        last_error: str = "",
    ) -> None:
        payload = {
            "taskId": state.task_id,
            "objective": state.objective,
            "status": state.status,
            "todoSummary": _todo_summary(todos),
            "todos": todos,
            "lastAction": last_action,
            "nextAction": next_action,
            "lastError": last_error,
            "lastStopReason": state.last_stop_reason,
            "updatedAt": _now(),
        }
        self._write_json(self.task_dir(state.task_id) / "checkpoint.json", payload)

    def append_event(self, task_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
        directory = self.task_dir(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": _now(),
            "taskId": task_id,
            "type": event_type,
            "data": data or {},
        }
        with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def task_dir(self, task_id: str) -> Path:
        if not task_id:
            raise ValueError("task_id is required")
        return self.base_dir / task_id

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(path)


def _todo_summary(todos: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [str(item.get("status") or "pending") for item in todos if isinstance(item, dict)]
    return {
        "total": len(statuses),
        "pending": statuses.count("pending"),
        "in_progress": statuses.count("in_progress"),
        "completed": statuses.count("completed"),
        "cancelled": statuses.count("cancelled"),
    }


def _new_task_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"lt_{stamp}_{uuid4().hex[:8]}"


def _summary(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:160]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_nonnegative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _clamp_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)
