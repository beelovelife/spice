"""SQLite-backed append-only tree session storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from spice.agent.sessions import (
    SESSION_VERSION,
    TREE_ENTRY_TYPES,
    SessionContext,
    SessionEntry,
    SessionInfo,
    _current_leaf,
    _current_model,
    _now,
    _stub_older_tool_results,
    message_from_dict,
    message_to_dict,
    workspace_key,
)
from spice.llm.config import CONFIG_DIR
from spice.llm.messages import Message
from spice.storage.sqlite import connect_sqlite, init_sqlite_database

DEFAULT_SQLITE_PATH = CONFIG_DIR / "spice.db"


class SqliteSessionStore:
    """SessionStore-compatible SQLite implementation."""

    def __init__(self, db_path: Path | None = None, *, cwd: Path | None = None) -> None:
        self.db_path = (db_path or DEFAULT_SQLITE_PATH).expanduser()
        self.cwd = cwd.resolve() if cwd else None
        self.workspace_key = workspace_key(self.cwd) if self.cwd else None
        self.base_dir = self.db_path.parent / "sessions"
        self._init_db()

    @property
    def base_root(self) -> Path:
        return self.base_dir

    def create(self, *, cwd: Path, provider: str, model: str, parent_session_id: str | None = None) -> SessionInfo:
        session_id = uuid4().hex[:12]
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into sessions (
                    id, cwd, workspace_key, provider, model, created_at, updated_at,
                    parent_session_id, leaf_id
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(cwd.resolve()),
                    workspace_key(cwd),
                    provider,
                    model,
                    now,
                    now,
                    parent_session_id,
                    None,
                ),
            )
        return self.info(session_id)

    def path_for(self, session_id: str) -> Path:
        return self.db_path

    def info(self, session_id: str) -> SessionInfo:
        header, entries = self._read_session(session_id)
        return self._info_from_entries(header, entries, fallback_id=session_id)

    def list(self, *, cwd: Path | None = None, limit: int | None = None, include_empty: bool = False) -> list[SessionInfo]:
        query = "select * from sessions"
        params: list[Any] = []
        if cwd is not None:
            query += " where cwd = ?"
            params.append(str(cwd.resolve()))
        query += " order by updated_at desc, created_at desc"
        if limit is not None:
            query += " limit ?"
            params.append(limit)

        rows: list[SessionInfo] = []
        with self._connect() as conn:
            for row in conn.execute(query, params).fetchall():
                header = self._header_from_row(row)
                entries = self._entries_for(row["id"], conn=conn)
                info = self._info_from_entries(header, entries, fallback_id=row["id"])
                if info.message_count == 0 and not include_empty:
                    continue
                rows.append(info)
        return rows

    def latest(self, *, cwd: Path | None = None) -> SessionInfo:
        rows = self.list(cwd=cwd, limit=1)
        if not rows:
            raise ValueError("No sessions found.")
        return rows[0]

    def resolve(self, session_id: str, *, cwd: Path | None = None) -> SessionInfo:
        try:
            info = self.info(session_id)
        except ValueError:
            info = None
        if info is not None and (cwd is None or Path(info.cwd).resolve() == cwd.resolve()):
            return info

        matches = []
        with self._connect() as conn:
            for row in conn.execute("select id from sessions where id like ? order by updated_at desc", (f"{session_id}%",)):
                candidate = self.info(str(row["id"]))
                if cwd is not None and Path(candidate.cwd).resolve() != cwd.resolve():
                    continue
                matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Session not found: {session_id}")
        raise ValueError(f"Session id prefix is ambiguous: {session_id}")

    def entries(self, session_id: str) -> list[SessionEntry]:
        return self._read_session(session_id)[1]

    def path_entries(self, session_id: str, leaf_id: str | None = None) -> list[SessionEntry]:
        header, entries = self._read_session(session_id)
        if not entries:
            return []
        return self._path_entries_from_entries(entries, leaf_id or _current_leaf(header, entries))

    def build_context(self, session_id: str, leaf_id: str | None = None) -> SessionContext:
        entries = self.path_entries(session_id, leaf_id=leaf_id)
        messages: list[Message] = []
        first_kept_id: str | None = None
        for entry in entries:
            if entry.type == "compaction":
                summary = str(entry.data.get("summary") or "")
                first_kept_id = entry.data.get("first_kept_entry_id")
                kept_messages = []
                for candidate in entries:
                    if candidate.id == first_kept_id and candidate.type == "message":
                        kept_messages.append(message_from_dict(candidate.data.get("message") or {}))
                        break
                messages = [Message(role="system", content=f"Previous conversation summary:\n{summary}"), *kept_messages]
                continue
            if first_kept_id and entry.id != first_kept_id and not messages:
                continue
            if entry.type == "message":
                messages.append(message_from_dict(entry.data.get("message") or {}))
        messages = _stub_older_tool_results(messages)
        return SessionContext(messages=messages, entries=entries, leaf_id=entries[-1].id if entries else None)

    def load_messages(self, session_id: str, leaf_id: str | None = None) -> list[Message]:
        return self.build_context(session_id, leaf_id=leaf_id).messages

    def append_message(self, session_id: str, message: Message, parent_id: str | None = None) -> str:
        return self._append_entry(session_id, "message", {"message": message_to_dict(message)}, parent_id=parent_id)

    def append_custom(self, session_id: str, data: dict[str, Any], parent_id: str | None = None) -> str:
        return self._append_entry(session_id, "custom", dict(data), parent_id=parent_id)

    def append_model_change(self, session_id: str, *, provider: str, model: str, parent_id: str | None = None) -> str:
        return self._append_entry(session_id, "model_change", {"provider": provider, "model": model}, parent_id=parent_id)

    def append_compaction(
        self,
        session_id: str,
        *,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        return self._append_entry(
            session_id,
            "compaction",
            {
                "summary": summary,
                "first_kept_entry_id": first_kept_entry_id,
                "tokens_before": tokens_before,
                "details": details or {},
            },
            parent_id=first_kept_entry_id,
        )

    def set_leaf(self, session_id: str, entry_id: str) -> str:
        if entry_id not in {entry.id for entry in self.entries(session_id)}:
            raise ValueError(f"Entry not found: {entry_id}")
        return self._append_entry(session_id, "leaf", {"leaf_id": entry_id}, parent_id=entry_id)

    def reset(self, session_id: str) -> SessionInfo:
        with self._connect() as conn:
            row = conn.execute("select id from sessions where id = ?", (session_id,)).fetchone()
            if row is None:
                raise ValueError(f"Session not found: {session_id}")
            now = _now()
            conn.execute("delete from session_entries where session_id = ?", (session_id,))
            conn.execute("update sessions set updated_at = ?, leaf_id = null where id = ?", (now, session_id))
        return self.info(session_id)

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            result = conn.execute("delete from sessions where id = ?", (session_id,))
            if result.rowcount == 0:
                raise ValueError(f"Session not found: {session_id}")

    def _append_entry(self, session_id: str, entry_type: str, data: dict[str, Any], parent_id: str | None = None) -> str:
        if entry_type not in TREE_ENTRY_TYPES:
            raise ValueError(f"Invalid session entry type: {entry_type}")
        with self._connect() as conn:
            header = self._header_for(session_id, conn=conn)
            entries = self._entries_for(session_id, conn=conn)
            if parent_id is None:
                parent_id = _current_leaf(header, entries)
            entry_id = uuid4().hex[:12]
            timestamp = _now()
            ordinal = self._next_ordinal(session_id, conn=conn)
            conn.execute(
                """
                insert into session_entries (
                    id, session_id, type, timestamp, parent_id, data_json, ordinal
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, session_id, entry_type, timestamp, parent_id, json.dumps(data, ensure_ascii=False), ordinal),
            )
            leaf_id = data.get("leaf_id") if entry_type == "leaf" else entry_id
            conn.execute(
                "update sessions set updated_at = ?, leaf_id = ? where id = ?",
                (timestamp, leaf_id, session_id),
            )
            return entry_id

    def _info_from_entries(self, header: dict[str, Any], entries: list[SessionEntry], *, fallback_id: str) -> SessionInfo:
        leaf_id = _current_leaf(header, entries)
        active_entries = self._path_entries_from_entries(entries, leaf_id)
        messages = [entry for entry in active_entries if entry.type == "message"]
        provider, model = _current_model(header, active_entries)
        preview = ""
        for entry in reversed(messages):
            message = entry.data.get("message") or {}
            content = str(message.get("content") or "").strip()
            if content:
                preview = content.splitlines()[0][:120]
                break
        updated_at = entries[-1].timestamp if entries else str(header.get("updated_at") or header.get("created_at") or "")
        return SessionInfo(
            id=str(header.get("id") or fallback_id),
            path=self.db_path,
            cwd=str(header.get("cwd") or ""),
            provider=provider,
            model=model,
            created_at=str(header.get("created_at") or ""),
            updated_at=updated_at,
            preview=preview,
            leaf_id=leaf_id,
            message_count=len(messages),
            parent_session_id=header.get("parent_session_id"),
        )

    def _path_entries_from_entries(self, entries: list[SessionEntry], leaf_id: str | None) -> list[SessionEntry]:
        if not entries or not leaf_id:
            return []
        by_id = {entry.id: entry for entry in entries}
        current = leaf_id
        path: list[SessionEntry] = []
        seen: set[str] = set()
        while current and current in by_id and current not in seen:
            seen.add(current)
            entry = by_id[current]
            if entry.type != "leaf":
                path.append(entry)
            current = entry.parent_id
        path.reverse()
        return path

    def _read_session(self, session_id: str) -> tuple[dict[str, Any], list[SessionEntry]]:
        with self._connect() as conn:
            header = self._header_for(session_id, conn=conn)
            entries = self._entries_for(session_id, conn=conn)
        return header, entries

    def _header_for(self, session_id: str, *, conn: sqlite3.Connection) -> dict[str, Any]:
        row = conn.execute("select * from sessions where id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"Session not found: {session_id}")
        return self._header_from_row(row)

    def _header_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "type": "session",
            "version": SESSION_VERSION,
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "cwd": row["cwd"],
            "provider": row["provider"],
            "model": row["model"],
            "parent_session_id": row["parent_session_id"],
            "leaf_id": row["leaf_id"],
        }

    def _entries_for(self, session_id: str, *, conn: sqlite3.Connection) -> list[SessionEntry]:
        rows = conn.execute(
            """
            select id, type, timestamp, parent_id, data_json
            from session_entries
            where session_id = ?
            order by ordinal asc
            """,
            (session_id,),
        ).fetchall()
        entries: list[SessionEntry] = []
        previous_id: str | None = None
        for row in rows:
            entry_type = str(row["type"] or "message")
            parent_id = row["parent_id"]
            if parent_id is None and entry_type == "message" and previous_id is not None:
                parent_id = previous_id
            try:
                data = json.loads(row["data_json"])
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            entries.append(
                SessionEntry(
                    id=str(row["id"]),
                    type=entry_type,
                    timestamp=str(row["timestamp"] or ""),
                    parent_id=parent_id,
                    data=data,
                )
            )
            if entry_type != "leaf":
                previous_id = str(row["id"])
        return entries

    def _next_ordinal(self, session_id: str, *, conn: sqlite3.Connection) -> int:
        row = conn.execute("select coalesce(max(ordinal), 0) + 1 as next_ordinal from session_entries where session_id = ?", (session_id,)).fetchone()
        return int(row["next_ordinal"])

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path)

    def _init_db(self) -> None:
        init_sqlite_database(self.db_path)
