"""验证命令的执行证据与工作区指纹；不依赖会话或前端。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from spice.llm.error_safety import sanitize_error_text
from spice.sandbox.base import ExecResult, ExecutionEnvironment

VERIFICATION_ENTRY = "verification"
MAX_FILES = 20_000
MAX_BYTES = 128 * 1024 * 1024
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
EXCLUDED_RUNTIME = {"tool-results", "sessions", "tasks"}
_observer: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("verification_observer", default=None)


@contextmanager
def verification_scope(observer: Callable[[dict[str, Any]], None]):
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def _excluded(path: Path) -> bool:
    parts = path.parts
    return any(part in EXCLUDED_DIRS for part in parts) or any(
        parts[i] == ".spice" and parts[i + 1] in EXCLUDED_RUNTIME for i in range(len(parts) - 1)
    )


def workspace_fingerprint(root: Path) -> dict[str, Any]:
    """有界地散列 Git 跟踪和未忽略文件；无法完整读取时不声称已验证。"""
    root = root.resolve()
    started = time.monotonic()
    scope = "filesystem"
    try:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "."],
                capture_output=True, timeout=5, check=False,
            )
        except FileNotFoundError:
            result = None
        if result is not None and result.returncode == 0:
            scope = "git-tracked-and-unignored"
            paths = {Path(os.fsdecode(name)) for name in result.stdout.split(b"\0") if name}
        else:
            paths: set[Path] = set()
            for directory, dirs, files in os.walk(root, followlinks=False):
                relative = Path(directory).relative_to(root)
                dirs[:] = [name for name in dirs if not _excluded(relative / name)]
                for name in [*files, *(name for name in dirs if (Path(directory) / name).is_symlink())]:
                    paths.add(relative / name)
                if len(paths) > MAX_FILES or time.monotonic() - started > 5:
                    raise ValueError("工作区超过指纹扫描上限")
        paths = {path for path in paths if not _excluded(path)}
        if len(paths) > MAX_FILES:
            raise ValueError("工作区文件数量超过上限")
        digest = hashlib.sha256()
        total_bytes = 0
        for relative in sorted(paths):
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("工作区文件路径越界")
            path = root / relative
            digest.update(os.fsencode(str(relative)) + b"\0")
            try:
                before = path.lstat()
            except FileNotFoundError:
                digest.update(b"missing\0")
                continue
            if not stat.S_ISREG(before.st_mode) or path.resolve().is_relative_to(root) is False:
                raise ValueError("工作区包含无法完整验证的符号链接、子模块或特殊文件")
            total_bytes += before.st_size
            if total_bytes > MAX_BYTES or time.monotonic() - started > 5:
                raise ValueError("工作区超过指纹扫描上限")
            digest.update(str(stat.S_IMODE(before.st_mode)).encode() + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("指纹扫描期间文件发生变化")
        return {"root": str(root), "digest": digest.hexdigest(), "scope": scope, "complete": True}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"root": str(root), "digest": None, "scope": scope, "complete": False,
                "reason": sanitize_error_text(str(exc))}


async def run_verified_command(
    command: str, *, root: Path, cwd: Path, environment: ExecutionEnvironment, timeout: float,
) -> ExecResult:
    return await run_verified_operation(
        command, root=root, cwd=cwd, environment=environment,
        operation=lambda: environment.run(command, cwd=cwd, timeout=timeout),
    )


async def run_verified_operation(
    command: str, *, root: Path, cwd: Path, environment: ExecutionEnvironment,
    operation: Callable[[], Awaitable[ExecResult]],
) -> ExecResult:
    observer = _observer.get()
    before = await asyncio.to_thread(workspace_fingerprint, root)
    record: dict[str, Any] = {
        "id": uuid4().hex,
        "command": sanitize_error_text(command),
        "cwd": str(cwd),
        "environment": environment.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "before": before,
        "after": None,
        "exit_code": None,
        "status": "unverified",
        "reason": "命令未完成",
    }
    started = time.perf_counter()
    try:
        result = await operation()
        after = await asyncio.to_thread(workspace_fingerprint, root)
        record.update(after=after, exit_code=result.exit_code)
        if result.timed_out:
            record.update(status="failed", reason="验证超时")
        elif result.exit_code != 0:
            record.update(status="failed", reason="验证命令退出码非零")
        elif not before["complete"] or not after["complete"]:
            record.update(status="unverified", reason="工作区指纹不完整")
        elif before["digest"] != after["digest"]:
            record.update(status="stale", reason="验证期间代码发生变化，需要对最终代码重新验证")
        else:
            record.update(status="verified", reason="指定命令通过，且验证前后工作区指纹一致")
        result.details = {**result.details, "verification": record}
        return result
    except asyncio.CancelledError:
        record.update(status="unverified", reason="验证被取消或执行超时")
        raise
    except Exception:
        record.update(status="failed", reason="验证命令执行异常")
        raise
    finally:
        record["duration_ms"] = int((time.perf_counter() - started) * 1000)
        if observer is not None:
            observer(record)


def verification_summary(
    records: Iterable[dict[str, Any]], *, root: Path, task_id: str = "", environment: str | None = None,
) -> dict[str, Any]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if str(record.get("task_id") or "") == task_id:
            latest[(str(record.get("command")), str(record.get("cwd")))] = record
    if not latest:
        return {"status": "unverified", "reason": "尚无明确执行的验证命令", "records": []}
    current = workspace_fingerprint(root)
    rows = []
    for record in latest.values():
        row = dict(record)
        after = row.get("after") or {}
        if row.get("status") not in {"verified", "failed", "stale", "unverified"}:
            row.update(status="unverified", reason="验证记录状态无效")
        if row.get("status") in {"verified", "failed"} and (
            (environment is not None and row.get("environment") != environment)
            or not after.get("complete") or not current["complete"] or after.get("digest") != current["digest"]
            or after.get("root") != current["root"] or after.get("scope") != current["scope"]
        ):
            row.update(
                recorded_status=row["status"], status="stale",
                reason="代码或执行环境已变化，或无法确认当前状态；该检查需要重新验证",
            )
        rows.append(row)
    statuses = {row.get("status") for row in rows}
    state = next((value for value in ("failed", "stale", "unverified") if value in statuses), "verified")
    reasons = {
        "verified": "已记录的指定验证命令均通过；不代表覆盖了所有正确性要求",
        "failed": "存在失败的验证命令",
        "stale": "已有验证证据不适用于当前代码",
        "unverified": "验证尚未完成或无法确认代码状态",
    }
    return {"status": state, "reason": reasons[state], "records": rows}


def verification_label(status: str) -> str:
    return {"verified": "已验证（指定命令）", "failed": "验证失败", "stale": "未验证（证据已过期）"}.get(status, "未验证")
