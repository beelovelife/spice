"""Local host execution environment."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from spice.sandbox.base import ExecResult


class LocalEnvironment:
    name = "local"

    async def ensure_ready(self) -> None:
        return None

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                env={**os.environ, **(env or {})},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                await _kill_process_group(proc)
            return ExecResult(
                output="",
                stdout="",
                stderr="",
                exit_code=124,
                timed_out=True,
                details={"environment": self.name},
            )
        except asyncio.CancelledError:
            if proc is not None:
                await asyncio.shield(_kill_process_group(proc))
            raise
        output = stdout.decode(errors="replace")
        error = stderr.decode(errors="replace")
        return ExecResult(
            output=output if not error else output + ("\n" if output else "") + error,
            stdout=output,
            stderr=error,
            exit_code=proc.returncode if proc is not None and proc.returncode is not None else 1,
            details={"environment": self.name},
        )

    async def cleanup(self) -> None:
        return None


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await proc.wait()
