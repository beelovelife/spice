"""Append-only tree JSONL session storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from spice.agent.logging_config import get_logger
from spice.llm.config import CONFIG_DIR
from spice.llm.messages import Message, ToolCall

SESSION_VERSION = 1
DEFAULT_SESSIONS_DIR = CONFIG_DIR / "sessions"
WORKSPACE_HASH_LENGTH = 10
TREE_ENTRY_TYPES = {"message", "model_change", "compaction", "custom", "leaf"}
logger = get_logger(__name__)


@dataclass
class SessionInfo:
    id: str
    path: Path
    cwd: str
    provider: str
    model: str
    created_at: str
    updated_at: str
    preview: str = ""
    leaf_id: str | None = None
    message_count: int = 0
    parent_session_id: str | None = None


@dataclass
class SessionEntry:
    id: str
    type: str
    timestamp: str
    parent_id: str | None
    data: dict[str, Any]


@dataclass
class SessionContext:
    messages: list[Message]
    entries: list[SessionEntry]
    leaf_id: str | None


def workspace_key(cwd: Path) -> str:
    resolved = str(cwd.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:WORKSPACE_HASH_LENGTH]
    return f"workspace-{digest}"


def message_to_dict(message: Message) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ]
    for attr in ("tool_call_id", "name", "provider", "model"):
        value = getattr(message, attr)
        if value is not None:
            data[attr] = value
    if message.is_error:
        data["is_error"] = True
    if message.metadata:
        data["metadata"] = message.metadata
    return data


def message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        role=data.get("role", "user"),
        content=data.get("content", ""),
        tool_calls=[
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(item.get("name") or ""),
                arguments=dict(item.get("arguments") or {}),
            )
            for item in data.get("tool_calls", []) or []
            if isinstance(item, dict)
        ],
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        is_error=bool(data.get("is_error", False)),
        provider=data.get("provider"),
        model=data.get("model"),
        metadata=dict(data.get("metadata") or {}),
    )


class SessionStore:
    def __init__(self, base_dir: Path | None = None, *, cwd: Path | None = None) -> None:
        self.base_dir = base_dir or DEFAULT_SESSIONS_DIR
        self.cwd = cwd.resolve() if cwd else None
        self.workspace_key = workspace_key(self.cwd) if self.cwd else None

    @property
    def base_root(self) -> Path:
        return self.base_dir

    def create(self, *, cwd: Path, provider: str, model: str, parent_session_id: str | None = None) -> SessionInfo:
        session_id = uuid4().hex[:12]
        path = self._workspace_dir(cwd) / f"{session_id}.jsonl"
        header = {
            "type": "session",
            "version": SESSION_VERSION,
            "id": session_id,
            "created_at": _now(),
            "updated_at": _now(),
            "cwd": str(cwd.resolve()),
            "provider": provider,
            "model": model,
            "parent_session_id": parent_session_id,
            "leaf_id": None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return self.info(session_id)

    def path_for(self, session_id: str) -> Path:
        direct = self.base_dir / f"{session_id}.jsonl"
        if direct.exists():
            return direct
        if self.workspace_key:
            scoped = self.base_dir / self.workspace_key / f"{session_id}.jsonl"
            if scoped.exists() or not any(self.base_dir.rglob(f"{session_id}.jsonl")):
                return scoped
        matches = list(self.base_dir.rglob(f"{session_id}.jsonl"))
        if matches:
            return matches[0]
        return direct

    def info(self, session_id: str) -> SessionInfo:
        path = self.path_for(session_id)
        if not path.exists():
            raise ValueError(f"Session not found: {session_id}")
        header, entries = self._read(path)
        return self._info_from_entries(path, header, entries, fallback_id=session_id)

    def list(self, *, cwd: Path | None = None, limit: int | None = None, include_empty: bool = False) -> list[SessionInfo]:
        rows: list[SessionInfo] = []
        for path in self._session_files():
            try:
                header, entries = self._read(path)
                info = self._info_from_entries(path, header, entries, fallback_id=path.stem)
            except ValueError:
                continue
            if cwd is not None and Path(info.cwd).resolve() != cwd.resolve():
                continue
            if info.message_count == 0 and not include_empty:
                continue
            rows.append(info)
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows

    def latest(self, *, cwd: Path | None = None) -> SessionInfo:
        rows = self.list(cwd=cwd)
        if not rows:
            raise ValueError("No sessions found.")
        return rows[0]

    def resolve(self, session_id: str, *, cwd: Path | None = None) -> SessionInfo:
        exact = self.path_for(session_id)
        if exact.exists():
            info = self.info(session_id)
            if cwd is None or Path(info.cwd).resolve() == cwd.resolve():
                return info
        matches = []
        for path in self._session_files():
            try:
                header, entries = self._read(path)
                info = self._info_from_entries(path, header, entries, fallback_id=path.stem)
            except ValueError:
                continue
            if not info.id.startswith(session_id):
                continue
            if cwd is not None and Path(info.cwd).resolve() != cwd.resolve():
                continue
            matches.append(info)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Session not found: {session_id}")
        raise ValueError(f"Session id prefix is ambiguous: {session_id}")

    def entries(self, session_id: str) -> list[SessionEntry]:
        return self._read(self.path_for(session_id))[1]

    def path_entries(self, session_id: str, leaf_id: str | None = None) -> list[SessionEntry]:
        header, entries = self._read(self.path_for(session_id))
        if not entries:
            return []
        return self._path_entries_from_entries(entries, leaf_id or _current_leaf(header, entries))

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

    def _info_from_entries(
        self,
        path: Path,
        header: dict[str, Any],
        entries: list[SessionEntry],
        *,
        fallback_id: str,
    ) -> SessionInfo:
        if header.get("type") != "session":
            raise ValueError(f"Invalid session file: {path}")
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
            path=path,
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
        path = self.path_for(session_id)
        header, _entries = self._read(path)
        header["leaf_id"] = None
        header["updated_at"] = _now()
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return self.info(session_id)

    def delete(self, session_id: str) -> None:
        self.path_for(session_id).unlink()

    def _append_entry(self, session_id: str, entry_type: str, data: dict[str, Any], parent_id: str | None = None) -> str:
        if entry_type not in TREE_ENTRY_TYPES:
            raise ValueError(f"Invalid session entry type: {entry_type}")
        path = self.path_for(session_id)
        header, entries = self._read(path)
        if parent_id is None:
            parent_id = _current_leaf(header, entries)
        entry_id = uuid4().hex[:12]
        row = {"type": entry_type, "id": entry_id, "timestamp": _now(), "parent_id": parent_id, **data}
        self._append_json(path, row)
        return entry_id

    def _append_json(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read(self, path: Path) -> tuple[dict[str, Any], list[SessionEntry]]:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if not rows:
                    raise ValueError(f"Invalid session header in {path}: line {line_number}") from exc
                logger.warning("Skipping invalid session JSONL row in %s at line %s", path, line_number)
                continue
        if not rows:
            raise ValueError(f"Empty session file: {path}")
        header = rows[0]
        entries: list[SessionEntry] = []
        previous_id: str | None = None
        for row in rows[1:]:
            entry_type = str(row.get("type") or "message")
            entry_id = str(row.get("id") or uuid4().hex[:12])
            parent_id = row.get("parent_id")
            if parent_id is None and entry_type == "message" and previous_id is not None:
                parent_id = previous_id
            data = dict(row)
            for key in ("id", "type", "timestamp", "parent_id"):
                data.pop(key, None)
            entries.append(
                SessionEntry(
                    id=entry_id,
                    type=entry_type,
                    timestamp=str(row.get("timestamp") or ""),
                    parent_id=parent_id,
                    data=data,
                )
            )
            if entry_type != "leaf":
                previous_id = entry_id
        return header, entries

    def _session_files(self) -> list[Path]:
        if not self.base_dir.exists():
            return []
        return sorted(self.base_dir.rglob("*.jsonl"))

    def _workspace_dir(self, cwd: Path) -> Path:
        return self.base_dir / workspace_key(cwd)


def _current_leaf(header: dict[str, Any], entries: list[SessionEntry]) -> str | None:
    leaf = header.get("leaf_id")
    for entry in entries:
        if entry.type == "leaf":
            leaf = entry.data.get("leaf_id") or entry.parent_id
        elif entry.type in TREE_ENTRY_TYPES:
            leaf = entry.id
    return leaf


def _current_model(header: dict[str, Any], entries: list[SessionEntry]) -> tuple[str, str]:
    provider = str(header.get("provider") or "")
    model = str(header.get("model") or "")
    for entry in entries:
        if entry.type == "model_change":
            provider = str(entry.data.get("provider") or provider)
            model = str(entry.data.get("model") or model)
    return provider, model


def _stub_older_tool_results(messages: list[Message]) -> list[Message]:
    tool_indexes = [index for index, message in enumerate(messages) if message.role == "tool"]
    keep = set(tool_indexes[-3:])
    updated: list[Message] = []
    for index, message in enumerate(messages):
        if message.role == "tool" and index not in keep and message.metadata.get("tool_result"):
            meta = message.metadata["tool_result"]
            display = meta.get("display") or meta.get("tool_name") or message.name or "tool"
            updated.append(
                Message(
                    role="tool",
                    content=f"[tool output omitted from older context: {display}]",
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    is_error=message.is_error,
                    metadata=message.metadata,
                )
            )
        else:
            updated.append(message)
    return updated


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
