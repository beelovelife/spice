"""Sandbox factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spice.sandbox.local import LocalEnvironment
from spice.sandbox.policy import WorkspacePolicy


def create_workspace_policy(sandbox_settings: dict[str, Any] | None, *, cwd: Path) -> WorkspacePolicy:
    settings = sandbox_settings if isinstance(sandbox_settings, dict) else {}
    raw_workspace = settings.get("workspace")
    workspace_settings = raw_workspace if isinstance(raw_workspace, dict) else {}
    return WorkspacePolicy.from_settings(workspace_settings, cwd=cwd)


def create_environment(sandbox_settings: dict[str, Any] | None, *, cwd: Path):
    settings = sandbox_settings if isinstance(sandbox_settings, dict) else {}
    mode = str(settings.get("mode") or "workspace")
    if mode in {"local", "workspace"}:
        return LocalEnvironment()
    if mode == "docker":
        from spice.sandbox.docker import DockerEnvironment

        raw_docker = settings.get("docker")
        docker_settings = raw_docker if isinstance(raw_docker, dict) else {}
        workspace = create_workspace_policy(settings, cwd=cwd)
        return DockerEnvironment.from_settings(docker_settings, workspace=workspace)
    raise ValueError(f"Unknown sandbox mode: {mode}")
