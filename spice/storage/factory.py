"""Factories for configured application-state stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spice.agent.long_task import LongTaskStore
from spice.agent.memory import MemoryStore
from spice.agent.sessions import SessionStore
from spice.llm.config import SpiceConfig
from spice.storage.sqlite_long_task import SqliteLongTaskStore
from spice.storage.sqlite_memory import SqliteMemoryHistoryBackend
from spice.storage.sqlite_sessions import SqliteSessionStore


def create_session_store(config: SpiceConfig, *, cwd: Path | None = None) -> SessionStore | SqliteSessionStore:
    if storage_backend(config) == "sqlite":
        return SqliteSessionStore(sqlite_path(config), cwd=cwd)
    return SessionStore(cwd=cwd)


def create_long_task_store(config: SpiceConfig, *, file_base_dir: Path | None = None) -> LongTaskStore | SqliteLongTaskStore:
    if storage_backend(config) == "sqlite":
        return SqliteLongTaskStore(sqlite_path(config))
    return LongTaskStore(file_base_dir)


def create_memory_store(config: SpiceConfig) -> MemoryStore:
    if storage_backend(config) == "sqlite":
        return MemoryStore(history_backend=SqliteMemoryHistoryBackend(sqlite_path(config)))
    return MemoryStore()


def sqlite_path(config: SpiceConfig) -> Path:
    storage = config.storage if isinstance(config.storage, dict) else {}
    raw_path = storage.get("sqlitePath") or storage.get("sqlite_path") or "~/.spice/spice.db"
    return Path(str(raw_path)).expanduser()


def storage_backend(config: SpiceConfig) -> str:
    storage: dict[str, Any] = config.storage if isinstance(config.storage, dict) else {}
    backend = str(storage.get("backend") or "file").strip().lower()
    return backend if backend in {"file", "sqlite"} else "file"
