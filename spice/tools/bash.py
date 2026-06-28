"""Shell execution tool."""

from __future__ import annotations

import asyncio  # Re-exported for existing tests that monkeypatch subprocess behavior.
import os  # Re-exported for existing tests that monkeypatch process-group cleanup.
import signal  # Re-exported with os for process-group cleanup tests.
from typing import Any

from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result, truncate_head


async def bash(args: dict[str, Any], context: ToolContext) -> ToolResult:
    command = str(args.get("command") or "").strip()
    timeout = float(args.get("timeout") or 30)
    if not command:
        return tool_error("command is required.")
    try:
        workspace = context.workspace or create_workspace_policy(None, cwd=context.cwd)
        environment = context.environment or create_environment(None, cwd=context.cwd)
        cwd = workspace.resolve_exec_cwd(str(args.get("cwd") or "."))
        result = await environment.run(command, cwd=cwd, timeout=timeout)
    except (PermissionError, RuntimeError) as exc:
        return tool_error(str(exc))
    if result.timed_out:
        return tool_error(f"Command timed out after {timeout:g}s: {command}", result.details)
    content = result.output
    details = {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr, **result.details}
    if result.exit_code:
        return tool_error(truncate_head(content, 12000), details)
    return tool_result(truncate_head(content, 12000) or "(no output)", details)


def create_bash_tools() -> list[Tool]:
    return [
        Tool(
            name="bash",
            description=(
                "Run a shell command in the workspace. Use for builds, tests, package managers, git, "
                "scripts, processes, and complex shell pipelines. For simple file reading, listing, "
                "or searching, prefer read_file, list_dir, or search_files. For workspace file edits, "
                "prefer edit_file or apply_patch over shell scripts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "cwd": {
                        "type": "string",
                        "description": "Working directory relative to the current workspace. Defaults to '.'.",
                    },
                    "timeout": {"type": "number", "minimum": 1, "maximum": 600},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            execute=bash,
            requires_confirmation=True,
        )
    ]
