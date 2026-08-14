"""Factories for configured application-state stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spice.agent.long_task import LongTaskStore
from spice.agent.memory import MemoryStore, resolve_workspace_root, workspace_memory_id
from spice.agent.sessions import SessionStore, SessionStoreProtocol
from spice.llm.config import SpiceConfig
from spice.storage.sqlite_long_task import SqliteLongTaskStore
from spice.storage.sqlite_memory import SqliteMemoryHistoryBackend
from spice.storage.sqlite_sessions import SqliteSessionStore


def create_session_store(config: SpiceConfig, *, cwd: Path | None = None) -> SessionStoreProtocol:
    if storage_backend(config) == "sqlite":
        return SqliteSessionStore(sqlite_path(config), cwd=cwd)
    return SessionStore(cwd=cwd)


def create_long_task_store(config: SpiceConfig, *, file_base_dir: Path | None = None) -> LongTaskStore | SqliteLongTaskStore:
    if storage_backend(config) == "sqlite":
        return SqliteLongTaskStore(sqlite_path(config))
    return LongTaskStore(file_base_dir)


def create_memory_store(config: SpiceConfig, *, workspace: Path | None = None) -> MemoryStore:
    kwargs = {
        "user_char_limit": config.memory_user_char_limit,
        "memory_char_limit": config.memory_global_char_limit,
        "project_memory_char_limit": config.memory_project_char_limit,
        "workspace": workspace,
    }
    if storage_backend(config) == "sqlite":
        workspace_id = None
        if workspace is not None:
            root = resolve_workspace_root(workspace)
            workspace_id = workspace_memory_id(root) if root is not None else None
        return MemoryStore(history_backend=SqliteMemoryHistoryBackend(sqlite_path(config), workspace_id=workspace_id), **kwargs)
    return MemoryStore(**kwargs)


def sqlite_path(config: SpiceConfig) -> Path:
    storage = config.storage if isinstance(config.storage, dict) else {}
    raw_path = storage.get("sqlitePath") or storage.get("sqlite_path") or "~/.spice/spice.db"
    return Path(str(raw_path)).expanduser()


def storage_backend(config: SpiceConfig) -> str:
    storage: dict[str, Any] = config.storage if isinstance(config.storage, dict) else {}
    backend = str(storage.get("backend") or "file").strip().lower()
    return backend if backend in {"file", "sqlite"} else "file"
