"""Web search tool."""

from __future__ import annotations

from typing import Any

from tavily import AsyncTavilyClient

from spice.llm.config import get_api_key
from spice.llm.error_safety import public_exception_message
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result, truncate_head_tail


async def web_search(args: dict[str, Any], context: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    max_results = int(args.get("max_results") or 10)
    if not query:
        return tool_error("query is required.")
    api_key = get_api_key("tavily")
    if not api_key:
        return tool_error("TAVILY_API_KEY is not configured.", code="tool_configuration_missing")
    client = AsyncTavilyClient(api_key=api_key)
    try:
        try:
            response = await client.search(query=query, max_results=max_results)
        except Exception as exc:
            return tool_error(
                public_exception_message(exc, prefix="Web search failed"),
                code="web_search_failed",
            )
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
    content = "\n\n".join(blocks) or "(no results)"
    preview = truncate_head_tail(content, 12000)
    return tool_result(
        preview,
        {"result_count": len(blocks), "output_truncated": preview != content},
        full_content=content if preview != content else None,
    )


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
            concurrency="parallel",
        )
    ]
