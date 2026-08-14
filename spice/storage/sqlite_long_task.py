"""SQLite-backed sustained task storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spice.agent.long_task import LongTaskState, _new_task_id, _now, _todo_summary
from spice.llm.config import CONFIG_DIR
from spice.storage.sqlite import init_sqlite_database, open_sqlite

DEFAULT_SQLITE_PATH = CONFIG_DIR / "spice.db"


class SqliteLongTaskStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or DEFAULT_SQLITE_PATH).expanduser()
        self.base_dir = self.db_path.parent / "tasks"
        init_sqlite_database(self.db_path)

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
        with open_sqlite(self.db_path) as conn:
            row = conn.execute("select data_json from long_tasks where task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise ValueError(f"Long task not found: {task_id}")
        data = json.loads(row["data_json"])
        if not isinstance(data, dict):
            raise ValueError(f"Invalid long task state: {task_id}")
        return LongTaskState.from_dict(data)

    def save_state(self, state: LongTaskState) -> None:
        state.updated_at = state.updated_at or _now()
        payload = state.to_dict()
        with open_sqlite(self.db_path) as conn:
            conn.execute(
                """
                insert into long_tasks (
                    task_id, session_id, status, objective, created_at, updated_at, data_json
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(task_id) do update set
                    session_id = excluded.session_id,
                    status = excluded.status,
                    objective = excluded.objective,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                (
                    state.task_id,
                    state.session_id,
                    state.status,
                    state.objective,
                    state.created_at,
                    state.updated_at,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

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
        with open_sqlite(self.db_path) as conn:
            conn.execute(
                """
                insert into long_task_checkpoints (task_id, updated_at, data_json)
                values (?, ?, ?)
                on conflict(task_id) do update set
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                (state.task_id, payload["updatedAt"], json.dumps(payload, ensure_ascii=False)),
            )

    def append_event(self, task_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
        event = {
            "timestamp": _now(),
            "taskId": task_id,
            "type": event_type,
            "data": data or {},
        }
        with open_sqlite(self.db_path) as conn:
            conn.execute(
                """
                insert into long_task_events (task_id, timestamp, type, data_json)
                values (?, ?, ?, ?)
                """,
                (task_id, event["timestamp"], event_type, json.dumps(event, ensure_ascii=False)),
            )

    def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        with open_sqlite(self.db_path) as conn:
            row = conn.execute("select data_json from long_task_checkpoints where task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        data = json.loads(row["data_json"])
        return data if isinstance(data, dict) else None

    def read_events(self, task_id: str) -> list[dict[str, Any]]:
        with open_sqlite(self.db_path) as conn:
            rows = conn.execute(
                "select data_json from long_task_events where task_id = ? order by id",
                (task_id,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            data = json.loads(row["data_json"])
            if isinstance(data, dict):
                events.append(data)
        return events
