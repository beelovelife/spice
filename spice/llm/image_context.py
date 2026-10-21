"""Project image snapshots into a request without changing session history."""

from __future__ import annotations

from dataclasses import replace

from spice.llm.messages import Message


def project_image_context(messages: list[Message]) -> list[Message]:
    """Send images only until the next assistant response.

    All images after the latest assistant message belong to the pending input,
    including multiple tool results from one batch. Retrying that request keeps
    those images. Once an assistant responds, later requests retain the recorded
    text and tool IDs, but omit the old image payloads. Deriving this boundary
    from messages also handles resumed sessions without a separate image cursor.
    """
    boundary = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if messages[index].role == "assistant"),
        -1,
    )
    projected = []
    for index, message in enumerate(messages):
        if index <= boundary and message.images:
            message = replace(
                message,
                images=[],
                content=message.content or (
                    "[Previous image attachment omitted from this request. "
                    "Use view_image or browser_check if another look is needed.]"
                ),
            )
        projected.append(message)
    return projected
