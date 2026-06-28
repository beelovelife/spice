"""Track file reads so editing tools can detect stale context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileState:
    digest: str | None
    exists: bool
    partial: bool = False


class FileStateStore:
    def __init__(self) -> None:
        self._states: dict[Path, FileState] = {}

    def record_read(self, path: Path, *, partial: bool = False) -> None:
        self._states[path.resolve()] = FileState(_digest(path), path.exists(), partial)

    def note_write(self, path: Path) -> None:
        self._states[path.resolve()] = FileState(_digest(path), path.exists(), False)

    def check_before_edit(self, path: Path) -> str | None:
        resolved = path.resolve()
        state = self._states.get(resolved)
        if state is None:
            return f"read_file must be called first before editing {path}."
        if state.partial:
            return (
                f"Full read_file must be called before editing {path}; the last read was partial. "
                "Read the remaining file range or call read_file with a larger limit until partial=false."
            )
        current = FileState(_digest(path), path.exists(), False)
        if current.digest != state.digest or current.exists != state.exists:
            return f"Cannot edit {path}: file changed since it was last read."
        return None

    def check_before_overwrite(self, path: Path) -> str | None:
        resolved = path.resolve()
        state = self._states.get(resolved)
        if path.exists() and state is None:
            return f"read_file must be called first before overwriting {path}."
        if state is None:
            return None
        current = FileState(_digest(path), path.exists(), False)
        if current.digest != state.digest or current.exists != state.exists:
            return f"Cannot overwrite {path}: file changed since it was last read."
        return None


def _digest(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()
