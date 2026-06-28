"""Web search tool."""

from __future__ import annotations

import json
from typing import Any

from tavily import AsyncTavilyClient

from spice.llm.config import get_api_key
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result, truncate_head


async def web_search(args: dict[str, Any], context: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    max_results = int(args.get("max_results") or 10)
    if not query:
        return tool_error("query is required.")
    api_key = get_api_key("tavily")
    if not api_key:
        return tool_error("TAVILY_API_KEY is not configured.")
    client = AsyncTavilyClient(api_key=api_key)
    try:
        response = await client.search(query=query, max_results=max_results)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list):
        return tool_error("Unexpected Tavily response.", {"response": response})
    blocks = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or "(untitled)"
        url = item.get("url") or ""
        content = item.get("content") or ""
        blocks.append(f"{title}\n{url}\n{content}".strip())
    return tool_result(truncate_head("\n\n".join(blocks) or "(no results)", 12000), {"raw": response})


def create_web_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the web for current information using Tavily and return result snippets with URLs.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            execute=web_search,
        )
    ]
