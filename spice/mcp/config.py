"""MCP server configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from spice.llm.config import SECRETS_PATH, SETTINGS_PATH

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class McpToolFilter:
    include: frozenset[str] | None = None
    exclude: frozenset[str] = frozenset()

    def allows(self, name: str) -> bool:
        if self.include is not None:
            return name in self.include
        return name not in self.exclude


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: Literal["stdio", "http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    connect_timeout: float = 15.0
    tool_timeout: float = 120.0
    tool_filter: McpToolFilter = field(default_factory=McpToolFilter)
    source: Literal["global", "project"] = "global"
    source_path: Path | None = None

    def trust_digest(self) -> str:
        payload = {
            "command": self.command,
            "args": list(self.args),
            "env_keys": sorted(self.env),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class McpConfigLoadResult:
    servers: dict[str, McpServerConfig]
    errors: tuple[str, ...] = ()
    project_file: Path | None = None


def load_mcp_config(cwd: Path) -> McpConfigLoadResult:
    secrets = _read_object(SECRETS_PATH)
    merged: dict[str, tuple[dict[str, Any], str, Path]] = {}
    errors: list[str] = []
    settings = _read_object(SETTINGS_PATH)
    _merge_servers(merged, settings, source="global", path=SETTINGS_PATH, errors=errors)
    project_file = find_project_mcp_file(cwd)
    if project_file is not None:
        _merge_servers(
            merged,
            _read_object(project_file),
            source="project",
            path=project_file,
            errors=errors,
        )

    servers: dict[str, McpServerConfig] = {}
    for name, (raw, source, path) in merged.items():
        try:
            servers[name] = _parse_server(name, raw, source=source, path=path, secrets=secrets)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
    return McpConfigLoadResult(servers, tuple(errors), project_file)


def find_project_mcp_file(cwd: Path) -> Path | None:
    current = cwd.resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        candidate = parent / ".mcp.json"
        if candidate.is_file():
            return candidate
    return None


def _merge_servers(
    target: dict[str, tuple[dict[str, Any], str, Path]],
    document: dict[str, Any],
    *,
    source: str,
    path: Path,
    errors: list[str],
) -> None:
    raw_servers = document.get("mcpServers")
    if raw_servers is None:
        return
    if not isinstance(raw_servers, dict):
        errors.append(f"{path}: mcpServers must be an object")
        return
    for raw_name, raw_config in raw_servers.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_config, dict):
            errors.append(f"{path}: invalid MCP server entry {raw_name!r}")
            continue
        target[name] = (raw_config, source, path)


def _parse_server(
    name: str,
    raw: dict[str, Any],
    *,
    source: str,
    path: Path,
    secrets: dict[str, Any],
) -> McpServerConfig:
    command = raw.get("command")
    url = raw.get("url")
    if bool(command) == bool(url):
        raise ValueError("configure exactly one of command or url")
    args = raw.get("args") or []
    env = raw.get("env") or {}
    headers = raw.get("headers") or {}
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("args must be an array of strings")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError("env must be an object of strings")
    if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ValueError("headers must be an object of strings")
    tools = raw.get("tools") or {}
    if not isinstance(tools, dict):
        raise ValueError("tools must be an object")
    include = _name_set(tools.get("include"), present="include" in tools)
    exclude = _name_set(tools.get("exclude"), present=True) or frozenset()
    values = {**{k: str(v) for k, v in secrets.items()}, **os.environ}
    return McpServerConfig(
        name=name,
        transport="stdio" if command else "http",
        command=_expand(str(command), values) if command else None,
        args=tuple(_expand(item, values) for item in args),
        env={key: _expand(value, values) for key, value in env.items()},
        url=_expand(str(url), values) if url else None,
        headers={key: _expand(value, values) for key, value in headers.items()},
        enabled=bool(raw.get("enabled", True)),
        connect_timeout=_positive_number(raw.get("connectTimeout", raw.get("connect_timeout")), 15.0),
        tool_timeout=_positive_number(raw.get("toolTimeout", raw.get("tool_timeout")), 120.0),
        tool_filter=McpToolFilter(include=include, exclude=exclude),
        source="project" if source == "project" else "global",
        source_path=path,
    )


def _expand(value: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"missing environment variable {name}")
        return values[name]

    return _ENV_PATTERN.sub(replace, value)


def _name_set(value: Any, *, present: bool) -> frozenset[str] | None:
    if not present:
        return None
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return frozenset(value)
    raise ValueError("tool include/exclude must be a string or array of strings")


def _positive_number(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeouts must be numbers") from exc
    if parsed <= 0:
        raise ValueError("timeouts must be positive")
    return parsed


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
