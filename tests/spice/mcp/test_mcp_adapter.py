from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace


from spice.mcp.adapter import adapt_mcp_tool, normalize_mcp_result
from spice.mcp.config import McpServerConfig
from spice.tools.base import ToolContext


def test_read_only_annotation_controls_confirmation_and_routes_native_names() -> None:
    calls = []

    async def call(server, tool, arguments):
        calls.append((server, tool, arguments))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], isError=False)

    remote = SimpleNamespace(
        name="read-item",
        description="Read an item",
        inputSchema={"type": "object", "properties": {"id": {"type": "string"}}},
        annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
    )

    tool, read_only = adapt_mcp_tool(McpServerConfig("my server", "http", url="https://example.com"), remote, call)
    result = asyncio.run(tool.execute({"id": "1"}, ToolContext(cwd=Path.cwd())))

    assert tool.name == "mcp__my_server__read-item"
    assert read_only is True
    assert tool.requires_confirmation is False
    assert result.content == "ok"
    assert calls == [("my server", "read-item", {"id": "1"})]


def test_structured_and_unsupported_content_are_not_silently_lost() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(type="image", data="...")],
        structuredContent={"answer": 42},
        isError=False,
    )

    normalized = normalize_mcp_result(result, server="s", tool="t")

    assert "unsupported MCP content type: image" in normalized.content
    assert '"answer": 42' in normalized.content
    assert normalized.details["unsupported_content_types"] == ["image"]
