"""Shell execution tool."""

from __future__ import annotations

from typing import Any

from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.tools.base import Tool, ToolContext, ToolResult, fatal_tool_error, tool_error, tool_result, truncate_head_tail


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
        return fatal_tool_error(str(exc), code="execution_policy_denied")
    if result.timed_out:
        return tool_error(f"Command timed out after {timeout:g}s: {command}", result.details)
    content = result.output
    preview = truncate_head_tail(content, 12000)
    details = {
        "exit_code": result.exit_code,
        "stdout_lines": len(result.stdout.splitlines()),
        "stderr_lines": len(result.stderr.splitlines()),
        "stdout_chars": len(result.stdout),
        "stderr_chars": len(result.stderr),
        "output_truncated": preview != content,
        **result.details,
    }
    if result.exit_code:
        return tool_error(preview, details, full_content=content if preview != content else None)
    return tool_result(preview or "(no output)", details, full_content=content if preview != content else None)


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
