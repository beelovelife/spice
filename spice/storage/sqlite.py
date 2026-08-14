"""SQLite database initialization helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Single source of truth for the application-state schema. Keep this constant
# in sync with the SQLite storage implementations under spice/storage/.
SCHEMA = """
pragma foreign_keys = on;

create table if not exists sessions (
    id text primary key,
    cwd text not null,
    workspace_key text,
    provider text not null,
    model text not null,
    created_at text not null,
    updated_at text not null,
    parent_session_id text,
    leaf_id text
);

create table if not exists session_entries (
    id text primary key,
    session_id text not null,
    type text not null,
    timestamp text not null,
    parent_id text,
    data_json text not null,
    ordinal integer not null,
    foreign key(session_id) references sessions(id) on delete cascade
);

create index if not exists session_entries_session_ordinal
    on session_entries(session_id, ordinal);

create index if not exists session_entries_session_parent
    on session_entries(session_id, parent_id);

create index if not exists sessions_workspace_updated
    on sessions(workspace_key, updated_at);

create index if not exists sessions_cwd_updated
    on sessions(cwd, updated_at);

create table if not exists long_tasks (
    task_id text primary key,
    session_id text,
    status text not null,
    objective text not null,
    created_at text not null,
    updated_at text not null,
    data_json text not null
);

create table if not exists long_task_checkpoints (
    task_id text primary key,
    updated_at text not null,
    data_json text not null,
    foreign key(task_id) references long_tasks(task_id) on delete cascade
);

create table if not exists long_task_events (
    id integer primary key autoincrement,
    task_id text not null,
    timestamp text not null,
    type text not null,
    data_json text not null,
    foreign key(task_id) references long_tasks(task_id) on delete cascade
);

create index if not exists long_tasks_session_updated
    on long_tasks(session_id, updated_at);

create index if not exists long_tasks_status_updated
    on long_tasks(status, updated_at);

create index if not exists long_task_events_task_id
    on long_task_events(task_id, id);

create table if not exists memory_history (
    cursor integer primary key autoincrement,
    timestamp text not null,
    summary text not null,
    source text not null,
    session_id text not null,
    metadata_json text not null,
    workspace_id text
);

create table if not exists memory_distill_state (
    key text primary key,
    value text not null
);

create table if not exists memory_distill_log (
    id integer primary key autoincrement,
    timestamp text not null,
    event text not null,
    data_json text not null
);

create index if not exists memory_history_session_cursor
    on memory_history(session_id, cursor);

create index if not exists memory_history_workspace_cursor
    on memory_history(workspace_id, cursor);

create table if not exists artifacts (
    artifact_id text primary key,
    session_id text,
    entry_id text,
    tool_name text,
    created_at text not null,
    file_path text not null,
    bytes integer not null,
    original_chars integer,
    sha256 text,
    preview text,
    pinned integer not null default 0
);

create index if not exists artifacts_session_created
    on artifacts(session_id, created_at);

create index if not exists artifacts_created
    on artifacts(created_at);
"""


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path.expanduser())
    conn.row_factory = sqlite3.Row
    conn.execute("pragma foreign_keys=on")
    return conn


@contextmanager
def open_sqlite(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection that commits on success, rolls back on error, and always closes.

    sqlite3's own connection context manager only manages the transaction, so
    `with connect_sqlite(...)` leaks the connection (and WAL file handles).
    """
    conn = connect_sqlite(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_sqlite_database(db_path: Path) -> None:
    with open_sqlite(db_path) as conn:
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
        table_exists = conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'memory_history'"
        ).fetchone()
        if table_exists is not None:
            columns = {str(row[1]) for row in conn.execute("pragma table_info(memory_history)")}
            if "workspace_id" not in columns:
                conn.execute("alter table memory_history add column workspace_id text")
        conn.executescript(sqlite_schema())


def sqlite_schema() -> str:
    return SCHEMA
