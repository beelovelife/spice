"""Workspace access policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


DEFAULT_PROTECTED_WRITE = [".spice/**", ".git/**"]
DEFAULT_SECRET_PATHS = [
    ".env",
    ".env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_ed25519",
]


@dataclass
class WorkspacePolicy:
    root: Path
    restrict: bool = True
    max_write_bytes: int = 1_000_000
    protected_write: list[str] = field(default_factory=lambda: list(DEFAULT_PROTECTED_WRITE))
    secret_paths: list[str] = field(default_factory=lambda: list(DEFAULT_SECRET_PATHS))

    @classmethod
    def from_settings(cls, settings: dict[str, Any] | None, *, cwd: Path) -> "WorkspacePolicy":
        data = settings if isinstance(settings, dict) else {}
        raw_root = data.get("root", ".")
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = cwd / root
        return cls(
            root=root.resolve(),
            restrict=bool(data.get("restrict", True)),
            max_write_bytes=_coerce_positive_int(data.get("max_write_bytes"), 1_000_000),
            protected_write=_string_list(data.get("protected_write"), DEFAULT_PROTECTED_WRITE),
            secret_paths=_string_list(data.get("secret_paths"), DEFAULT_SECRET_PATHS),
        )

    def resolve_read(self, raw_path: str) -> Path:
        path = self._resolve(raw_path)
        self._check_inside(path, raw_path)
        self._check_secret(path, raw_path, operation="read")
        return path

    def resolve_read_dir(self, raw_path: str) -> Path:
        return self.resolve_read(raw_path)

    def resolve_write(self, raw_path: str, *, content_size: int = 0) -> Path:
        path = self._resolve(raw_path)
        self._check_inside(path, raw_path)
        self._check_write_size(content_size)
        self._check_protected_write(path, raw_path)
        self._check_secret(path, raw_path, operation="write")
        return path

    def resolve_exec_cwd(self, raw_path: str | None = None) -> Path:
        path = self._resolve(raw_path or ".")
        self._check_inside(path, raw_path or ".")
        return path

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def _resolve(self, raw_path: str) -> Path:
        if not raw_path:
            raw_path = "."
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _check_inside(self, path: Path, raw_path: str) -> None:
        if not self.restrict:
            return
        root = self.root.resolve()
        if path != root and root not in path.parents:
            raise PermissionError(f"Path is outside workspace: {raw_path}")

    def _check_write_size(self, content_size: int) -> None:
        if content_size > self.max_write_bytes:
            raise PermissionError(
                f"Write exceeds workspace limit: {content_size} bytes > {self.max_write_bytes} bytes"
            )

    def _check_protected_write(self, path: Path, raw_path: str) -> None:
        rel = self.relative(path)
        for pattern in self.protected_write:
            if _matches(rel, pattern):
                raise PermissionError(f"Path is protected from writes: {raw_path}")

    def _check_secret(self, path: Path, raw_path: str, *, operation: str) -> None:
        rel = self.relative(path)
        for pattern in self.secret_paths:
            if _matches(rel, pattern):
                raise PermissionError(f"Path is blocked by secret policy for {operation}: {raw_path}")


def _matches(rel: str, pattern: str) -> bool:
    rel = rel.strip("/")
    pattern = pattern.strip("/")
    if fnmatch(rel, pattern) or fnmatch("/" + rel, pattern):
        return True
    if pattern.startswith("**/") and fnmatch(rel, pattern[3:]):
        return True
    if pattern.endswith("/**") and rel == pattern[:-3]:
        return True
    return False


def _string_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    return [item for item in value if isinstance(item, str) and item]


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, 1)
