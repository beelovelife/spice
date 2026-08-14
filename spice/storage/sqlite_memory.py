"""SQLite-backed memory history and distill log storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spice.llm.config import CONFIG_DIR
from spice.storage.sqlite import init_sqlite_database, open_sqlite

DEFAULT_SQLITE_PATH = CONFIG_DIR / "spice.db"


class SqliteMemoryHistoryBackend:
    def __init__(self, db_path: Path | None = None, *, workspace_id: str | None = None) -> None:
        self.db_path = (db_path or DEFAULT_SQLITE_PATH).expanduser()
        self.workspace_id = workspace_id
        init_sqlite_database(self.db_path)
        self._backfill_legacy_workspace_ids()

    def append_history(
        self,
        *,
        summary: str,
        source: str,
        session_id: str,
        metadata: dict[str, Any],
        history_limit: int,
    ) -> dict[str, Any]:
        timestamp = datetime.now(UTC).isoformat()
        with open_sqlite(self.db_path) as conn:
            result = conn.execute(
                """
                insert into memory_history (timestamp, summary, source, session_id, metadata_json, workspace_id)
                values (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, summary, source, session_id, json.dumps(metadata, ensure_ascii=False), self.workspace_id),
            )
            cursor = int(result.lastrowid or 0)
        entry = {
            "cursor": int(cursor),
            "timestamp": timestamp,
            "summary": summary,
            "source": source,
            "session_id": session_id,
            "metadata": metadata,
        }
        cleanup = self.compact_history(max_entries=history_limit)
        if cleanup["removed"]:
            entry["cleanup"] = cleanup
        return entry

    def read_history(self) -> list[dict[str, Any]]:
        with open_sqlite(self.db_path) as conn:
            rows = conn.execute(
                """
                select cursor, timestamp, summary, source, session_id, metadata_json
                from memory_history
                where workspace_id is ?
                order by cursor
                """,
                (self.workspace_id,),
            ).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                metadata = {}
            history.append(
                {
                    "cursor": int(row["cursor"]),
                    "timestamp": row["timestamp"],
                    "summary": row["summary"],
                    "source": row["source"],
                    "session_id": row["session_id"],
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
        return history

    def set_distill_cursor(self, cursor: int) -> None:
        key = self._cursor_key()
        with open_sqlite(self.db_path) as conn:
            conn.execute(
                """
                insert into memory_distill_state (key, value)
                values (?, ?)
                on conflict(key) do update set value = excluded.value
                """,
                (key, str(max(0, cursor))),
            )

    def distill_cursor(self) -> int:
        key = self._cursor_key()
        with open_sqlite(self.db_path) as conn:
            row = conn.execute("select value from memory_distill_state where key = ?", (key,)).fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def compact_history(self, *, max_entries: int) -> dict[str, Any]:
        history = self.read_history()
        if len(history) <= max_entries:
            return {"removed": 0, "dropped_unprocessed": []}
        cursor = self.distill_cursor()
        removed = history[:-max_entries]
        kept_min_cursor = int(history[-max_entries]["cursor"])
        dropped = [int(item["cursor"]) for item in removed if int(item.get("cursor", 0)) > cursor]
        with open_sqlite(self.db_path) as conn:
            conn.execute(
                "delete from memory_history where workspace_id is ? and cursor < ?",
                (self.workspace_id, kept_min_cursor),
            )
        if dropped:
            self.append_distill_log({"event": "history_unprocessed_dropped", "cursors": dropped})
        return {"removed": len(removed), "dropped_unprocessed": dropped}

    def append_distill_log(self, payload: dict[str, Any]) -> None:
        timestamp = str(payload.get("timestamp") or datetime.now(UTC).isoformat())
        event = str(payload.get("event") or "distill")
        with open_sqlite(self.db_path) as conn:
            conn.execute(
                """
                insert into memory_distill_log (timestamp, event, data_json)
                values (?, ?, ?)
                """,
                (timestamp, event, json.dumps({"timestamp": timestamp, **payload}, ensure_ascii=False)),
            )

    def read_distill_log(self) -> list[dict[str, Any]]:
        with open_sqlite(self.db_path) as conn:
            rows = conn.execute("select data_json from memory_distill_log order by id").fetchall()
        logs: list[dict[str, Any]] = []
        for row in rows:
            data = json.loads(row["data_json"])
            if isinstance(data, dict):
                logs.append(data)
        return logs

    def _cursor_key(self) -> str:
        return "cursor" if self.workspace_id is None else f"cursor:{self.workspace_id}"

    def _backfill_legacy_workspace_ids(self) -> None:
        with open_sqlite(self.db_path) as conn:
            rows = conn.execute(
                "select cursor, metadata_json from memory_history where workspace_id is null"
            ).fetchall()
            for row in rows:
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    continue
                workspace_id = metadata.get("workspace_id") if isinstance(metadata, dict) else None
                if workspace_id:
                    conn.execute(
                        "update memory_history set workspace_id = ? where cursor = ?",
                        (str(workspace_id), int(row["cursor"])),
                    )
