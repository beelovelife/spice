"""Adapt discovered MCP tools and results to Spice tools."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from spice.mcp.config import McpServerConfig
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result

McpCall = Callable[[str, str, dict[str, Any]], Awaitable[Any]]


def public_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp__{_component(server_name)}__{_component(tool_name)}"


def adapt_mcp_tool(config: McpServerConfig, remote_tool: Any, call: McpCall) -> tuple[Tool, bool]:
    remote_name = str(remote_tool.name)
    exposed_name = public_tool_name(config.name, remote_name)
    annotations = getattr(remote_tool, "annotations", None)
    read_only = bool(getattr(annotations, "readOnlyHint", False))
    destructive = getattr(annotations, "destructiveHint", None)
    safe_read_only = read_only and destructive is not True
    schema = _normalize_schema(getattr(remote_tool, "inputSchema", None))

    async def execute(args: dict[str, Any], _context: ToolContext) -> ToolResult:
        try:
            result = await call(config.name, remote_name, args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from spice.mcp.connection import sanitize_mcp_error

            return tool_error(
                f"MCP tool {exposed_name} failed: {sanitize_mcp_error(exc)}",
                {"server": config.name, "remote_tool": remote_name},
                code="mcp_tool_error",
            )
        return normalize_mcp_result(result, server=config.name, tool=remote_name)

    # Local import above is intentional for cheap startup, but CancelledError is needed here.
    import asyncio

    description = str(getattr(remote_tool, "description", "") or f"MCP tool {remote_name} from {config.name}")
    tool = Tool(
        name=exposed_name,
        description=f"[MCP server: {config.name}] {description}",
        parameters=schema,
        execute=execute,
        requires_confirmation=not safe_read_only,
        concurrency="serial",
        timeout_seconds=config.tool_timeout,
    )
    return tool, safe_read_only


def normalize_mcp_result(result: Any, *, server: str, tool: str) -> ToolResult:
    parts: list[str] = []
    unsupported: list[str] = []
    for block in list(getattr(result, "content", None) or []):
        block_type = str(getattr(block, "type", type(block).__name__))
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            unsupported.append(block_type)
            parts.append(f"[unsupported MCP content type: {block_type}]")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    content = "\n".join(part for part in parts if part).strip() or "(empty MCP tool result)"
    details: dict[str, Any] = {"server": server, "remote_tool": tool}
    if unsupported:
        details["unsupported_content_types"] = unsupported
    if bool(getattr(result, "isError", False)):
        return tool_error(content, details, code="mcp_tool_returned_error")
    return tool_result(content, details)


def _normalize_schema(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}
    schema = dict(value)
    if schema.get("type") not in {None, "object"}:
        return {"type": "object", "properties": {}}
    schema["type"] = "object"
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    return schema


def _component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return normalized or "unnamed"
