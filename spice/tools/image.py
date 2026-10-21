"""Read workspace images as model-visible attachments."""

from __future__ import annotations

import asyncio
from typing import Any

from spice.llm.images import load_image
from spice.sandbox.factory import create_workspace_policy
from spice.tools.base import Tool, ToolContext, ToolResult, fatal_tool_error, tool_error


async def view_image(args: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.supports_vision:
        return tool_error(
            "The current model does not support image input. Select a vision-capable model; "
            "a file path or an ASCII approximation is not a visual inspection.",
            code="vision_unsupported",
        )
    workspace = context.workspace or create_workspace_policy(None, cwd=context.cwd)
    try:
        path = workspace.resolve_read(str(args.get("path") or ""))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")
    try:
        image = await asyncio.to_thread(load_image, path)
    except (OSError, ValueError) as exc:
        return tool_error(str(exc), code="invalid_image")
    return ToolResult(
        content=f"Image attached: {path} ({image.width}×{image.height}, {image.media_type}; animated images use the first frame).",
        details={"path": str(path), "width": image.width, "height": image.height, "media_type": image.media_type},
        images=[image],
    )


def create_image_tools() -> list[Tool]:
    return [Tool(
        name="view_image",
        description=(
            "View a local PNG, JPEG, WebP, or GIF as an actual image using a vision-capable model. "
            "Use for screenshots, visual layout, colors, and image details instead of pixel-to-ASCII scripts. "
            "Images are resized to at most 2048 pixels per edge; GIF uses the first frame."
        ),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=view_image,
        concurrency="parallel",
    )]
