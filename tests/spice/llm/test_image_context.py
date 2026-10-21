from __future__ import annotations

from copy import deepcopy

from spice.llm.image_context import project_image_context
from spice.llm.messages import ImageContent, Message, ToolCall


def test_only_pending_images_are_sent_without_changing_history():
    image = ImageContent("image/png", "image-bytes", 1400, 900, "page")
    messages = [
        Message(role="user", content="reference", images=[image]),
        Message(role="assistant", tool_calls=[ToolCall("old", "browser_check", {})]),
        Message(role="tool", name="browser_check", tool_call_id="old",
                content='{"passed": true}', images=[image]),
        Message(role="assistant", content="Compare day and night", tool_calls=[
            ToolCall("day", "browser_check", {}), ToolCall("night", "browser_check", {}),
        ]),
        Message(role="tool", name="browser_check", tool_call_id="day",
                content='{"passed": false}', is_error=True, images=[image]),
        Message(role="tool", name="browser_check", tool_call_id="night",
                content='{"passed": true}', images=[image]),
        Message(role="system", content="runtime context"),
    ]
    original = deepcopy(messages)

    projected = project_image_context(messages)

    assert [len(message.images) for message in projected] == [0, 0, 0, 0, 1, 1, 0]
    assert [message.content for message in projected] == [message.content for message in messages]
    assert [message.tool_call_id for message in projected] == [message.tool_call_id for message in messages]
    assert projected[4].is_error
    assert messages == original
    assert project_image_context(projected) == projected


def test_images_without_an_assistant_response_remain_pending():
    image = ImageContent("image/png", "image-bytes", 40, 20)
    messages = [Message(role="user", images=[image]), Message(role="user", content="continue")]

    assert project_image_context(messages) == messages


def test_image_only_history_retains_a_nonempty_text_placeholder():
    image = ImageContent("image/png", "image-bytes", 40, 20)
    messages = [Message(role="user", images=[image]), Message(role="assistant", content="seen")]

    projected = project_image_context(messages)

    assert projected[0].content and "omitted" in projected[0].content
    assert projected[0].images == []
    assert messages[0].images == [image] and messages[0].content == ""


def test_requesting_the_same_image_again_is_allowed():
    image = ImageContent("image/png", "same-image-bytes", 40, 20, "/tmp/ui.png")
    messages = [
        Message(role="user", images=[image]),
        Message(role="assistant", tool_calls=[ToolCall("again", "view_image", {"path": "/tmp/ui.png"})]),
        Message(role="tool", tool_call_id="again", name="view_image", images=[image]),
    ]

    projected = project_image_context(messages)

    assert projected[0].images == []
    assert projected[-1].images == [image]
