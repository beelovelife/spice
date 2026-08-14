"""Execution environment interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ExecResult:
    output: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class ExecutionEnvironment(Protocol):
    name: str

    async def ensure_ready(self) -> None:
        """Prepare the environment before command execution."""

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run *command* inside the environment."""
        ...

    async def cleanup(self) -> None:
        """Release environment resources."""
