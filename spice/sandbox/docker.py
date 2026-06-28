"""Docker execution environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spice.sandbox.base import ExecResult
from spice.sandbox.policy import WorkspacePolicy


@dataclass
class DockerEnvironment:
    workspace: WorkspacePolicy
    image: str = "spice-sandbox:latest"
    container_name: str = ""
    container_workspace: str = "/workspace"
    network: bool = False
    memory: str = "2g"
    cpus: float = 2
    pids_limit: int = 256
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    no_new_privileges: bool = True
    persist: bool = True
    fallback_to_local: bool = False
    executable: str = "docker"

    name = "docker"

    @classmethod
    def from_settings(cls, settings: dict[str, Any], *, workspace: WorkspacePolicy) -> "DockerEnvironment":
        return cls(
            workspace=workspace,
            image=str(settings.get("image") or "spice-sandbox:latest"),
            container_name=str(settings.get("container_name") or ""),
            container_workspace=str(settings.get("container_workspace") or "/workspace"),
            network=bool(settings.get("network", False)),
            memory=str(settings.get("memory") or "2g"),
            cpus=_coerce_float(settings.get("cpus"), 2),
            pids_limit=_coerce_int(settings.get("pids_limit"), 256),
            cap_drop=_string_list(settings.get("cap_drop"), ["ALL"]),
            no_new_privileges=bool(settings.get("no_new_privileges", True)),
            persist=bool(settings.get("persist", True)),
            fallback_to_local=bool(settings.get("fallback_to_local", False)),
            executable=str(settings.get("executable") or "docker"),
        )

    @property
    def resolved_container_name(self) -> str:
        if self.container_name:
            return self.container_name
        digest = hashlib.sha256(str(self.workspace.root).encode("utf-8")).hexdigest()[:8]
        return f"spice-sandbox-{digest}"

    async def ensure_ready(self) -> None:
        await self._docker(["version", "--format", "{{.Server.Version}}"], timeout=10)
        state = await self._container_state()
        if state is None:
            await self._create_container()
        elif state != "running":
            await self._docker(["start", self.resolved_container_name], timeout=30)
        await self._validate_container()

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        await self.ensure_ready()
        container_cwd = self._container_path(cwd)
        args = ["exec", "-w", container_cwd]
        for key, value in (env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        args.extend([self.resolved_container_name, "bash", "-lc", command])
        result = await self._docker(args, timeout=timeout, check=False)
        stdout = result["stdout"]
        stderr = result["stderr"]
        output = stdout if not stderr else stdout + ("\n" if stdout else "") + stderr
        return ExecResult(
            output=output,
            stdout=stdout,
            stderr=stderr,
            exit_code=result["returncode"],
            timed_out=result.get("timed_out", False),
            details={
                "environment": self.name,
                "container": self.resolved_container_name,
                "cwd": container_cwd,
            },
        )

    async def cleanup(self) -> None:
        if self.persist:
            return
        await self._docker(["rm", "-f", self.resolved_container_name], timeout=30, check=False)

    async def stop(self) -> None:
        await self._docker(["stop", self.resolved_container_name], timeout=30, check=False)

    async def status(self) -> dict[str, Any]:
        state = await self._container_state()
        return {
            "mode": self.name,
            "container": self.resolved_container_name,
            "state": state or "missing",
            "workspace": str(self.workspace.root),
            "container_workspace": self.container_workspace,
            "network": self.network,
            "image": self.image,
            "fallback_to_local": self.fallback_to_local,
        }

    async def _create_container(self) -> None:
        args = [
            "run",
            "-d",
            "--name",
            self.resolved_container_name,
            "--label",
            "spice.managed=true",
            "--label",
            f"spice.workspace={self.workspace.root}",
            "-w",
            self.container_workspace,
            "-v",
            f"{self.workspace.root}:{self.container_workspace}",
        ]
        if not self.network:
            args.extend(["--network", "none"])
        for cap in self.cap_drop:
            args.extend(["--cap-drop", cap])
        if self.no_new_privileges:
            args.extend(["--security-opt", "no-new-privileges"])
        if self.pids_limit > 0:
            args.extend(["--pids-limit", str(self.pids_limit)])
        if self.memory:
            args.extend(["--memory", self.memory])
        if self.cpus > 0:
            args.extend(["--cpus", str(self.cpus)])
        args.extend([self.image, "sleep", "infinity"])
        await self._docker(args, timeout=120)

    async def _container_state(self) -> str | None:
        result = await self._docker(
            ["inspect", "-f", "{{.State.Status}}", self.resolved_container_name],
            timeout=10,
            check=False,
        )
        if result["returncode"] != 0:
            return None
        state = result["stdout"].strip()
        return state or None

    async def _validate_container(self) -> None:
        result = await self._docker(["inspect", self.resolved_container_name], timeout=10)
        try:
            payload = json.loads(result["stdout"])[0]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("Docker inspect returned an unexpected payload.") from exc
        labels = payload.get("Config", {}).get("Labels") or {}
        if labels.get("spice.managed") != "true":
            raise RuntimeError(f"Docker container is not managed by Spice: {self.resolved_container_name}")
        if labels.get("spice.workspace") != str(self.workspace.root):
            raise RuntimeError(f"Docker container workspace label does not match: {self.resolved_container_name}")
        host_config = payload.get("HostConfig") or {}
        if host_config.get("Privileged"):
            raise RuntimeError("Docker sandbox must not run privileged.")
        mounts = payload.get("Mounts") or []
        expected_source = str(self.workspace.root)
        expected_dest = self.container_workspace
        if not any(mount.get("Source") == expected_source and mount.get("Destination") == expected_dest for mount in mounts):
            raise RuntimeError("Docker sandbox mount does not match current workspace.")

    def _container_path(self, host_path: Path) -> str:
        resolved = host_path.resolve()
        try:
            rel = resolved.relative_to(self.workspace.root)
        except ValueError as exc:
            raise PermissionError(f"Path is outside workspace: {host_path}") from exc
        if str(rel) == ".":
            return self.container_workspace
        return f"{self.container_workspace.rstrip('/')}/{rel.as_posix()}"

    async def _docker(
        self,
        args: list[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            self.executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"stdout": "", "stderr": "", "returncode": 124, "timed_out": True}
        result = {
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "returncode": proc.returncode or 0,
            "timed_out": False,
        }
        if check and result["returncode"] != 0:
            message = result["stderr"].strip() or result["stdout"].strip() or "docker command failed"
            raise RuntimeError(message)
        return result


def _string_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    return [item for item in value if isinstance(item, str) and item]


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
