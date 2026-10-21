"""Read-only local health checks used by ``spice doctor``."""

from __future__ import annotations

import json
from importlib import metadata, util
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from spice.agent.logging_config import log_path
from spice.extensions.manager import ExtensionManager
from spice.llm.config import (
    CONFIG_DIR,
    SETTINGS_PATH,
    SpiceConfig,
    get_api_key,
    load_config,
)
from spice.llm.model_registry import ModelRegistry, find_initial_model
from spice.mcp.config import load_mcp_config
from spice.skills.loader import load_skills
from spice.storage.factory import sqlite_path, storage_backend
from spice.tools.catalog import build_tool_catalog
from spice.version import __version__

DoctorStatus = Literal["pass", "warning", "error"]


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: DoctorStatus
    summary: str
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def error_count(self) -> int:
        return sum(check.status == "error" for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(check.status == "warning" for check in self.checks)

    @property
    def ready(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "checks": [check.to_dict() for check in self.checks],
        }


def build_doctor_report(
    *,
    cwd: Path | None = None,
    config: SpiceConfig | None = None,
    settings_path: Path | None = None,
) -> DoctorReport:
    """Run diagnostics without contacting a model provider or MCP server."""
    resolved_cwd = (cwd or Path.cwd()).resolve()
    resolved_settings = settings_path or SETTINGS_PATH
    effective_config = config or load_config()
    checks = [
        _runtime_check(),
        _tui_check(),
        _config_check(resolved_settings),
        _model_check(effective_config, resolved_settings),
        _storage_check(effective_config),
        _sandbox_check(effective_config),
        _tools_check(effective_config),
        _skills_check(resolved_cwd),
        _extensions_check(resolved_cwd),
        _mcp_check(resolved_cwd),
        _logs_check(),
    ]
    return DoctorReport(tuple(checks))


def _runtime_check() -> DoctorCheck:
    uv_path = shutil.which("uv")
    summary = f"Spice {__version__} · Python {platform.python_version()}"
    if uv_path:
        return DoctorCheck(
            "Runtime", "pass", f"{summary} · uv available", (sys.executable, uv_path)
        )
    return DoctorCheck(
        "Runtime", "warning", f"{summary} · uv not found", (sys.executable,)
    )


def _tui_check() -> DoctorCheck:
    if util.find_spec("textual") is None:
        return DoctorCheck(
            "TUI",
            "error",
            "Textual is not installed; `spice tui` is unavailable",
            (
                "uv tool install: uv tool upgrade --reinstall spice",
                "project environment: uv sync",
            ),
        )
    try:
        version = metadata.version("textual")
    except metadata.PackageNotFoundError:
        version = "installed"
    return DoctorCheck("TUI", "pass", f"Textual {version}")


def _config_check(path: Path) -> DoctorCheck:
    if not path.exists():
        return DoctorCheck(
            "Config", "warning", "settings.json not found; using defaults", (str(path),)
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return DoctorCheck(
            "Config",
            "error",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}",
            (str(path),),
        )
    except OSError as exc:
        return DoctorCheck(
            "Config", "error", f"cannot read settings: {exc}", (str(path),)
        )
    if not isinstance(payload, dict):
        return DoctorCheck(
            "Config", "error", "top-level value must be an object", (str(path),)
        )
    return DoctorCheck("Config", "pass", "settings.json loaded", (str(path),))


def _model_check(config: SpiceConfig, settings_path: Path) -> DoctorCheck:
    result = find_initial_model(ModelRegistry(settings_path), config)
    if result.model is None:
        return DoctorCheck(
            "Model", "error", result.message or "model is not configured"
        )
    model = result.model
    protocol = model.protocol or config.protocol or "provider default"
    label = f"{model.provider}/{model.id} · {protocol}"
    if not get_api_key(model.provider, model.api_key_envs):
        envs = ", ".join(model.api_key_envs or []) or f"a {model.provider} API key"
        return DoctorCheck(
            "Model", "error", f"{label} · API key missing", (f"Expected: {envs}",)
        )
    return DoctorCheck("Model", "pass", f"{label} · API key configured")


def _storage_check(config: SpiceConfig) -> DoctorCheck:
    backend = storage_backend(config)
    target = sqlite_path(config) if backend == "sqlite" else CONFIG_DIR / "sessions"
    writable, reason = _path_is_writable(target)
    if not writable:
        return DoctorCheck(
            "Storage", "error", f"{backend} · not writable", (str(target), reason)
        )
    return DoctorCheck("Storage", "pass", f"{backend} · writable", (str(target),))


def _sandbox_check(config: SpiceConfig) -> DoctorCheck:
    settings = config.sandbox if isinstance(config.sandbox, dict) else {}
    mode = str(settings.get("mode") or "workspace")
    if mode not in {"local", "workspace", "docker"}:
        return DoctorCheck("Sandbox", "error", f"unknown mode: {mode}")
    if mode == "docker" and not shutil.which("docker"):
        return DoctorCheck(
            "Sandbox", "error", "docker mode configured, but docker is not available"
        )
    return DoctorCheck("Sandbox", "pass", f"{mode} mode")


def _tools_check(config: SpiceConfig) -> DoctorCheck:
    entries = build_tool_catalog(config)
    enabled = sum(entry.status == "enabled" for entry in entries)
    unavailable = [entry for entry in entries if entry.status == "unavailable"]
    disabled = sum(entry.status == "disabled" for entry in entries)
    confirmations = sum(
        entry.status == "enabled" and entry.confirmation == "required"
        for entry in entries
    )
    summary = f"{enabled} enabled · {disabled} disabled · {confirmations} require confirmation"
    if unavailable:
        details = tuple(f"{entry.name}: {entry.note}" for entry in unavailable)
        return DoctorCheck(
            "Tools", "warning", f"{summary} · {len(unavailable)} unavailable", details
        )
    return DoctorCheck("Tools", "pass", summary)


def _skills_check(cwd: Path) -> DoctorCheck:
    result = load_skills(cwd=cwd)
    if result.diagnostics:
        details = tuple(f"{item.type}: {item.message}" for item in result.diagnostics)
        return DoctorCheck(
            "Skills",
            "warning",
            f"{len(result.skills)} loaded · {len(result.diagnostics)} diagnostics",
            details,
        )
    return DoctorCheck(
        "Skills", "pass", f"{len(result.skills)} loaded · no diagnostics"
    )


def _extensions_check(cwd: Path) -> DoctorCheck:
    try:
        paths = ExtensionManager(cwd=cwd).discover()
    except OSError as exc:
        return DoctorCheck("Extensions", "warning", f"discovery failed: {exc}")
    return DoctorCheck("Extensions", "pass", f"{len(paths)} discovered · not executed")


def _mcp_check(cwd: Path) -> DoctorCheck:
    result = load_mcp_config(cwd)
    enabled = sum(server.enabled for server in result.servers.values())
    if result.errors:
        return DoctorCheck(
            "MCP",
            "warning",
            f"{enabled} enabled · {len(result.errors)} configuration errors",
            result.errors,
        )
    return DoctorCheck("MCP", "pass", f"{enabled} enabled · configuration valid")


def _logs_check() -> DoctorCheck:
    target = log_path()
    writable, reason = _path_is_writable(target)
    if not writable:
        return DoctorCheck(
            "Logs", "error", "log path is not writable", (str(target), reason)
        )
    return DoctorCheck("Logs", "pass", "log path is writable", (str(target),))


def _path_is_writable(path: Path) -> tuple[bool, str]:
    target = path.expanduser()
    if target.exists():
        return os.access(target, os.W_OK), "existing path is not writable"
    parent = target.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.exists():
        return False, "no existing parent directory"
    return os.access(parent, os.W_OK), f"parent is not writable: {parent}"
