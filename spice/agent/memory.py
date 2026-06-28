"""Simple persistent memory store."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from spice.llm.config import CONFIG_DIR
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.stream import stream_model
from spice.llm.types import ModelRequestOptions, StreamError, TextDelta

DEFAULT_MEMORY_DIR = CONFIG_DIR / "memory"
ENTRY_DELIMITER = "\n§\n"

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps atomic replace but no cross-process lock.
    fcntl = None


class MemoryHistoryBackend(Protocol):
    def append_history(self, *, summary: str, source: str, session_id: str, metadata: dict[str, Any], history_limit: int) -> dict[str, Any]:
        ...

    def read_history(self) -> list[dict[str, Any]]:
        ...

    def set_distill_cursor(self, cursor: int) -> None:
        ...

    def distill_cursor(self) -> int:
        ...

    def compact_history(self, *, max_entries: int) -> dict[str, Any]:
        ...

    def append_distill_log(self, payload: dict[str, Any]) -> None:
        ...


class MemoryStore:
    def __init__(
        self,
        root: Path | None = None,
        *,
        user_char_limit: int = 1_500,
        memory_char_limit: int = 3_000,
        history_limit: int = 1_000,
        distill_batch_size: int = 50,
        history_backend: MemoryHistoryBackend | None = None,
    ) -> None:
        self.root = root or DEFAULT_MEMORY_DIR
        self.user_char_limit = user_char_limit
        self.memory_char_limit = memory_char_limit
        self.history_limit = history_limit
        self.distill_batch_size = distill_batch_size
        self.user_file = self.root / "USER.md"
        self.memory_file = self.root / "MEMORY.md"
        self.history_file = self.root / "history.jsonl"
        self.cursor_file = self.root / ".distill_cursor"
        self.distill_log_file = self.root / "distill.log.jsonl"
        self.lock_file = self.root / ".lock"
        self.legacy_user_file = self.root / "user.json"
        self.legacy_memory_file = self.root / "memory.json"
        self.legacy_history_file = self.root / "history.json"
        self.legacy_cursor_file = self.root / "distill_cursor.json"
        self.history_backend = history_backend

    def add(self, target: str, content: str) -> dict[str, Any]:
        target = self._normalize_target(target)
        content = content.strip()
        error = _content_error(content)
        if error:
            return self._error_result(target, error)
        with self._lock():
            limit = self._limit_for(target)
            entries = self.read_entries(target)
            current_usage = _entries_usage(entries)
            if current_usage + len(content) > limit:
                return self._error_result(target, f"{target} memory limit exceeded.", entries=entries)
            if content not in entries:
                entries.append(content)
                self._write_entries(target, entries)
            return self._success_result(target, entries)

    def read(self, target: str) -> dict[str, Any]:
        return self._success_result(target, self.read_entries(target))

    def read_entries(self, target: str) -> list[str]:
        path = self._target_file(target)
        if path.exists():
            return _read_entry_file(path)
        data = _read_json(self._legacy_target_file(target), [])
        return [str(item) for item in data] if isinstance(data, list) else []

    def replace(self, target: str, old: str, new: str) -> dict[str, Any]:
        target = self._normalize_target(target)
        old = old.strip()
        new = new.strip()
        if not old:
            return self._error_result(target, "old text cannot be empty")
        error = _content_error(new)
        if error:
            return self._error_result(target, error)
        with self._lock():
            entries = self.read_entries(target)
            matches = [index for index, item in enumerate(entries) if old in item]
            if not matches:
                return self._error_result(target, "No entry matched.", entries=entries)
            if len(matches) > 1:
                return self._error_result(target, "Multiple entries matched; provide a unique substring.", entries=entries)
            candidate = list(entries)
            candidate[matches[0]] = new
            usage = _entries_usage(candidate)
            if usage > self._limit_for(target):
                return self._error_result(target, f"{target} memory limit exceeded.", entries=entries)
            entries[matches[0]] = new
            self._write_entries(target, entries)
            return self._success_result(target, entries)

    def remove(self, target: str, needle: str) -> dict[str, Any]:
        target = self._normalize_target(target)
        needle = needle.strip()
        if not needle:
            return self._error_result(target, "old text cannot be empty")
        with self._lock():
            entries = self.read_entries(target)
            matches = [index for index, item in enumerate(entries) if needle in item]
            if not matches:
                return self._error_result(target, "No entry matched.", entries=entries)
            if len(matches) > 1:
                return self._error_result(target, "Multiple entries matched; provide a unique substring.", entries=entries)
            del entries[matches[0]]
            self._write_entries(target, entries)
            return self._success_result(target, entries)

    def context_snapshot(self) -> str:
        sections = []
        for target, title in (("user", "User memory"), ("memory", "Project memory")):
            entries = self.read_entries(target)
            if entries:
                sections.append(title + ":\n" + "\n".join(f"- {item}" for item in entries))
        if not sections:
            return ""
        return (
            "Persistent memory snapshot:\n"
            "The following memory is persistent background context, not new user input.\n\n"
            + "\n\n".join(sections)
        )

    def append_history(self, *, summary: str, source: str, session_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        summary = summary.strip()
        if not summary:
            raise ValueError("history summary cannot be empty")
        if self.history_backend is not None:
            return self.history_backend.append_history(
                summary=summary,
                source=source,
                session_id=session_id,
                metadata=metadata,
                history_limit=self.history_limit,
            )
        with self._lock():
            history = self.read_history()
            cursor = max([int(item.get("cursor", 0)) for item in history] + [0]) + 1
            entry = {
                "cursor": cursor,
                "timestamp": datetime.now(UTC).isoformat(),
                "summary": summary,
                "source": source,
                "session_id": session_id,
                "metadata": metadata,
            }
            history.append(entry)
            self._write_history(history)
            cleanup = self._compact_history_locked(max_entries=self.history_limit)
            if cleanup["removed"]:
                entry["cleanup"] = cleanup
            return entry

    def read_history(self) -> list[dict[str, Any]]:
        if self.history_backend is not None:
            return self.history_backend.read_history()
        if self.history_file.exists():
            history = []
            try:
                lines = self.history_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    history.append(item)
            return history
        data = _read_json(self.legacy_history_file, [])
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def set_distill_cursor(self, cursor: int) -> None:
        if self.history_backend is not None:
            self.history_backend.set_distill_cursor(cursor)
            return
        with self._lock():
            self.root.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(self.cursor_file, f"{cursor}\n")

    def distill_cursor(self) -> int:
        if self.history_backend is not None:
            return self.history_backend.distill_cursor()
        try:
            return int(self.cursor_file.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            data = _read_json(self.legacy_cursor_file, {})
            return int(data.get("cursor", 0)) if isinstance(data, dict) else 0

    def compact_history(self, *, max_entries: int | None = None) -> dict[str, Any]:
        if self.history_backend is not None:
            return self.history_backend.compact_history(max_entries=max_entries or self.history_limit)
        with self._lock():
            return self._compact_history_locked(max_entries=max_entries or self.history_limit)

    def append_distill_log(self, payload: dict[str, Any]) -> None:
        if self.history_backend is not None:
            self.history_backend.append_distill_log(payload)
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self.distill_log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), **payload}, ensure_ascii=False) + "\n")

    def _compact_history_locked(self, *, max_entries: int) -> dict[str, Any]:
        max_entries = max_entries or self.history_limit
        history = self.read_history()
        if len(history) <= max_entries:
            return {"removed": 0, "dropped_unprocessed": []}
        cursor = self.distill_cursor()
        removed = history[:-max_entries]
        kept = history[-max_entries:]
        dropped = [int(item["cursor"]) for item in removed if int(item.get("cursor", 0)) > cursor]
        self._write_history(kept)
        if dropped:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.distill_log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "history_unprocessed_dropped", "cursors": dropped}) + "\n")
        return {"removed": len(removed), "dropped_unprocessed": dropped}

    def status(self) -> dict[str, Any]:
        history = self.read_history()
        cursor = self.distill_cursor()
        unprocessed = [item for item in history if int(item.get("cursor", 0)) > cursor]
        return {
            "history_count": len(history),
            "history_limit": self.history_limit,
            "processed_cursor": cursor,
            "unprocessed_count": len(unprocessed),
            "next_distill_batch": min(len(unprocessed), self.distill_batch_size),
            "user_usage": f"{sum(len(item) for item in self.read_entries('user'))}/{self.user_char_limit}",
            "memory_usage": f"{sum(len(item) for item in self.read_entries('memory'))}/{self.memory_char_limit}",
        }

    def _target_file(self, target: str) -> Path:
        target = self._normalize_target(target)
        if target == "user":
            return self.user_file
        if target == "memory":
            return self.memory_file
        raise ValueError("target must be user or memory")

    def _write_entries(self, target: str, entries: list[str]) -> None:
        path = self._target_file(target)
        _write_text_atomic(path, ENTRY_DELIMITER.join(entries).strip() + ("\n" if entries else ""))

    def _write_history(self, entries: list[dict[str, Any]]) -> None:
        lines = "".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n" for entry in entries)
        _write_text_atomic(self.history_file, lines)

    def _success_result(self, target: str, entries: list[str]) -> dict[str, Any]:
        target = self._normalize_target(target)
        limit = self._limit_for(target)
        return {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{_entries_usage(entries)}/{limit}",
            "entry_count": len(entries),
        }

    def _error_result(self, target: str, error: str, *, entries: list[str] | None = None) -> dict[str, Any]:
        target = self._normalize_target(target)
        current = self.read_entries(target) if entries is None else entries
        return {
            "success": False,
            "target": target,
            "error": error,
            "entries": current,
            "usage": f"{_entries_usage(current)}/{self._limit_for(target)}",
            "entry_count": len(current),
        }

    def _legacy_target_file(self, target: str) -> Path:
        target = self._normalize_target(target)
        if target == "user":
            return self.legacy_user_file
        if target == "memory":
            return self.legacy_memory_file
        raise ValueError("target must be user or memory")

    def _limit_for(self, target: str) -> int:
        target = self._normalize_target(target)
        return self.user_char_limit if target == "user" else self.memory_char_limit

    @staticmethod
    def _normalize_target(target: str) -> str:
        normalized = str(target or "").strip().lower()
        if normalized not in {"user", "memory"}:
            raise ValueError("target must be user or memory")
        return normalized

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class MemoryDistiller:
    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        model: Model | None = None,
        options: ModelRequestOptions | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.model = model
        self.options = options

    def distill(self) -> dict[str, Any]:
        history = self._unprocessed_history()
        if not history:
            return {"success": True, "processed": 0, "message": "No unprocessed memory history."}
        return {
            "success": False,
            "processed": 0,
            "message": "Memory distillation requires an async model request. Use run().",
        }

    async def run(self) -> dict[str, Any]:
        history = self._unprocessed_history()
        if not history:
            return {"success": True, "processed": 0, "message": "No unprocessed memory history."}
        if self.model is None or self.options is None:
            return {
                "success": False,
                "processed": 0,
                "message": "Memory distillation requires a model and request options.",
            }

        from_cursor = int(history[0].get("cursor", 0))
        to_cursor = int(history[-1].get("cursor", 0))
        plan = _parse_distill_plan(await self._request_plan(history))
        result = self._apply_plan(plan)
        if result["skipped"]:
            payload = {
                "success": False,
                "processed": 0,
                "from_cursor": from_cursor,
                "to_cursor": to_cursor,
                "adds": result["adds"],
                "replacements": result["replacements"],
                "removals": result["removals"],
                "skipped": result["skipped"],
                "message": "Memory distillation skipped one or more operations; cursor was not advanced.",
            }
            self._append_log({"event": "distill_skipped", **payload})
            return payload
        self.store.set_distill_cursor(to_cursor)
        cleanup = self.store.compact_history()
        payload = {
            "success": True,
            "processed": len(history),
            "from_cursor": from_cursor,
            "to_cursor": to_cursor,
            "adds": result["adds"],
            "replacements": result["replacements"],
            "removals": result["removals"],
            "skipped": result["skipped"],
            "cleanup": cleanup,
        }
        self._append_log({"event": "distill", **payload})
        return payload

    def _unprocessed_history(self) -> list[dict[str, Any]]:
        cursor = self.store.distill_cursor()
        history = [item for item in self.store.read_history() if int(item.get("cursor", 0)) > cursor]
        return history[: self.store.distill_batch_size]

    async def _request_plan(self, history: list[dict[str, Any]]) -> str:
        assert self.model is not None
        assert self.options is not None
        messages = [
            Message(
                role="system",
                content=(
                    "You distill coding-agent history summaries into durable long-term memory. "
                    "Return only valid JSON. Do not include markdown fences."
                ),
            ),
            Message(role="user", content=_distill_prompt(self.store, history)),
        ]
        parts: list[str] = []
        async for event in stream_model(self.model, messages, [], self.options):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, StreamError):
                raise RuntimeError(event.error)
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("Memory distillation returned an empty plan.")
        return text

    def _apply_plan(self, plan: dict[str, Any]) -> dict[str, int]:
        counts = {"adds": 0, "replacements": 0, "removals": 0, "skipped": 0}
        for item in _list_value(plan.get("adds")):
            target = _target_value(item.get("target"))
            content = str(item.get("content") or "").strip()
            if not target or not content:
                counts["skipped"] += 1
                continue
            result = self.store.add(target, content)
            counts["adds" if result.get("success") else "skipped"] += 1
        for item in _list_value(plan.get("replacements")):
            target = _target_value(item.get("target"))
            old = str(item.get("old") or item.get("old_text") or "").strip()
            content = str(item.get("content") or item.get("new") or "").strip()
            if not target or not old or not content:
                counts["skipped"] += 1
                continue
            result = self.store.replace(target, old, content)
            counts["replacements" if result.get("success") else "skipped"] += 1
        for item in _list_value(plan.get("removals")):
            target = _target_value(item.get("target"))
            old = str(item.get("old") or item.get("old_text") or item.get("content") or "").strip()
            if not target or not old:
                counts["skipped"] += 1
                continue
            result = self.store.remove(target, old)
            counts["removals" if result.get("success") else "skipped"] += 1
        return counts

    def _append_log(self, payload: dict[str, Any]) -> None:
        self.store.append_distill_log(payload)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _distill_prompt(store: MemoryStore, history: list[dict[str, Any]]) -> str:
    payload = {
        "existing_memory": {
            "user": store.read_entries("user"),
            "memory": store.read_entries("memory"),
        },
        "history": history,
    }
    return (
        "Extract only durable facts worth remembering across future sessions.\n"
        "Use target='user' for stable user preferences/profile. Use target='memory' for stable project or environment facts.\n"
        "Do not store secrets, raw logs, temporary progress, todo state, guesses, stack traces, or facts easily rediscovered from files.\n"
        "Return JSON with exactly these top-level keys: adds, replacements, removals.\n"
        "Schema:\n"
        "{\n"
        '  "adds": [{"target": "user|memory", "content": "..."}],\n'
        '  "replacements": [{"target": "user|memory", "old": "unique existing substring", "content": "new full entry"}],\n'
        '  "removals": [{"target": "user|memory", "old": "unique existing substring"}]\n'
        "}\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_distill_plan(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Memory distillation returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Memory distillation plan must be a JSON object.")
    return data


def _list_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _target_value(value: Any) -> str | None:
    target = str(value or "").strip()
    return target if target in {"user", "memory"} else None


def _read_entry_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [entry.strip() for entry in text.split(ENTRY_DELIMITER) if entry.strip()]


def _entries_usage(entries: list[str]) -> int:
    return sum(len(item) for item in entries)


def _content_error(content: str) -> str | None:
    if not content.strip():
        return "content cannot be empty"
    lowered = content.lower()
    blocked = [
        "<memory-context",
        "</memory-context",
        "ignore previous instructions",
        "ignore all previous instructions",
        "reveal your system prompt",
        "developer message",
    ]
    for pattern in blocked:
        if pattern in lowered:
            return f"memory content rejected because it contains unsafe pattern: {pattern}"
    secret_patterns = [
        r"\b(?:sk|sk-proj|sk-ant|AIza)[A-Za-z0-9_-]{16,}\b",
        r"\b[A-Z0-9_]*(?:API|TOKEN|SECRET|KEY)[A-Z0-9_]*\s*=\s*['\"]?[^'\"\s]{8,}",
    ]
    for pattern in secret_patterns:
        if re.search(pattern, content):
            return "memory content rejected because it appears to contain a secret"
    return None


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
