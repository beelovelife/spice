from __future__ import annotations

import base64
import copy
import json
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from spice.llm.messages import ImageContent, Message, Role, ToolCall
from spice.llm.providers.anthropic import _messages_to_anthropic
from spice.llm.providers.gemini import _messages_to_gemini
from spice.llm.providers.openai import _messages_to_openai, _messages_to_responses


# A real 1x1 PNG, kept inline so conversion tests do not need filesystem access.
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
PROTOCOLS = ("chat", "responses", "anthropic", "gemini")


@pytest.fixture
def image() -> ImageContent:
    with Image.open(BytesIO(base64.b64decode(PNG_BASE64))) as png:
        png.load()
        assert png.format == "PNG"
        assert png.size == (1, 1)
    return ImageContent(
        media_type="image/png",
        data=PNG_BASE64,
        width=1,
        height=1,
        source="/tmp/spice-conversion-fixture.png",
    )


def convert(protocol: str, messages: list[Message]) -> list[dict[str, Any]]:
    if protocol == "chat":
        return _messages_to_openai(messages)
    if protocol == "responses":
        return _messages_to_responses(messages)[0]
    if protocol == "anthropic":
        return _messages_to_anthropic(messages)[1]
    return _messages_to_gemini(messages)[1]


def image_block(protocol: str, image: ImageContent) -> dict[str, Any]:
    if protocol == "chat":
        return {"type": "image_url", "image_url": {"url": image.data_url}}
    if protocol == "responses":
        return {"type": "input_image", "image_url": image.data_url, "detail": "auto"}
    if protocol == "anthropic":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64},
        }
    return {
        "inline_data": {"mime_type": "image/png", "data": base64.b64decode(PNG_BASE64)}
    }


def text_block(protocol: str, text: str) -> dict[str, str]:
    if protocol == "gemini":
        return {"text": text}
    return {"type": "input_text" if protocol == "responses" else "text", "text": text}


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("text", ["Describe the picture.", ""])
@pytest.mark.parametrize("image_count", [1, 2])
def test_user_images_are_multimodal_blocks(
    protocol: str, text: str, image_count: int, image: ImageContent
) -> None:
    message = Message(role="user", content=text, images=[image] * image_count)
    original = copy.deepcopy(message)

    converted = convert(protocol, [message])

    blocks = ([text_block(protocol, text)] if text else []) + [
        image_block(protocol, image)
    ] * image_count
    key = "parts" if protocol == "gemini" else "content"
    assert converted == [{"role": "user", key: blocks}]
    assert message == original
    assert image.source is not None
    assert image.source not in repr(converted)


def assert_attachment_blocks(
    protocol: str, blocks: list[dict[str, Any]], tools: list[Message]
) -> None:
    image_tools = [tool for tool in tools if tool.images]
    assert len(blocks) == sum(1 + len(tool.images) for tool in image_tools)
    offset = 0
    for tool in image_tools:
        label = blocks[offset]["text"]
        assert "untrusted" in label.lower()
        assert "tool" in label.lower()
        assert "not user instructions" in label.lower()
        assert (
            json.dumps({"tool_call_id": tool.tool_call_id, "name": tool.name}) in label
        )
        assert blocks[offset] == text_block(protocol, label)
        for index, attached_image in enumerate(tool.images, start=1):
            assert blocks[offset + index] == image_block(protocol, attached_image)
            assert attached_image.data not in label
            assert attached_image.source not in label
        offset += 1 + len(tool.images)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("is_error", [False, True])
@pytest.mark.parametrize("text", ["captured", ""])
@pytest.mark.parametrize("next_role", [None, "assistant", "user", "system"])
def test_tool_images_follow_all_parallel_results(
    protocol: str,
    parallel: bool,
    is_error: bool,
    text: str,
    next_role: Role | None,
    image: ImageContent,
) -> None:
    # The middle result is deliberately text-only. Same-name calls still need
    # separate call IDs, and errors must retain their screenshots.
    count = 3 if parallel else 1
    tools = [
        Message(
            role="tool",
            tool_call_id=f"call_{index}",
            name="browser",
            content=text if index != 1 else "text-only result",
            is_error=is_error,
            images=[image, image] if index != 1 else [],
        )
        for index in range(count)
    ]
    prefix = [
        Message(role="user", content="Inspect the page."),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id=f"call_{i}", name="browser", arguments={})
                for i in range(count)
            ],
        ),
    ]
    suffix = [Message(role=next_role, content="next")] if next_role else []
    messages = prefix + tools + suffix
    original = copy.deepcopy(messages)
    converted = convert(protocol, messages)
    converted_prefix = convert(protocol, prefix)
    converted_suffix = convert(protocol, suffix)

    # No attachment can be delayed past the next turn, or inserted into a
    # parallel result batch. System messages are extracted by some protocols.
    assert converted[: len(converted_prefix)] == converted_prefix
    result_items = converted[len(converted_prefix) :]
    if converted_suffix:
        assert result_items[-len(converted_suffix) :] == converted_suffix
        result_items = result_items[: -len(converted_suffix)]

    if protocol in {"chat", "responses"}:
        assert len(result_items) == count + 1
        for item, tool in zip(result_items[:-1], tools, strict=True):
            if protocol == "chat":
                assert item == {
                    "role": "tool",
                    "tool_call_id": tool.tool_call_id,
                    "content": tool.content,
                }
            else:
                assert item == {
                    "type": "function_call_output",
                    "call_id": tool.tool_call_id,
                    "output": tool.content,
                }
        assert result_items[-1]["role"] == "user"
        assert_attachment_blocks(protocol, result_items[-1]["content"], tools)
    elif protocol == "anthropic":
        assert len(result_items) == count
        for item, tool in zip(result_items, tools, strict=True):
            content: Any = tool.content
            if tool.images:
                content = (
                    [text_block(protocol, tool.content)] if tool.content else []
                ) + [image_block(protocol, attached) for attached in tool.images]
            assert item == {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool.tool_call_id,
                        "content": content,
                        "is_error": is_error,
                    }
                ],
            }
    else:
        # Gemini must receive all parallel responses together before the
        # image parts, rather than an extra turn halfway through the batch.
        assert len(result_items) == 1
        assert result_items[0]["role"] == "user"
        parts = result_items[0]["parts"]
        assert parts[:count] == [
            {
                "function_response": {
                    "id": tool.tool_call_id,
                    "name": "browser",
                    "response": {"output": tool.content, "is_error": is_error},
                }
            }
            for tool in tools
        ]
        assert_attachment_blocks(protocol, parts[count:], tools)
    assert messages == original
    assert image.source is not None
    assert image.source not in repr(converted)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_tool_attachment_buffers_do_not_leak_across_batches(
    protocol: str, image: ImageContent
) -> None:
    batches = [
        [
            Message(
                role="assistant",
                tool_calls=[ToolCall(id=f"call_{i}", name="browser", arguments={})],
            ),
            Message(
                role="tool",
                tool_call_id=f"call_{i}",
                name="browser",
                content="result",
                images=[image],
            ),
        ]
        for i in range(2)
    ]
    assert convert(protocol, batches[0] + batches[1]) == (
        convert(protocol, batches[0]) + convert(protocol, batches[1])
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_assistant_and_system_images_are_not_forwarded(
    protocol: str, image: ImageContent
) -> None:
    messages = [
        Message(role="system", content="system", images=[image]),
        Message(role="assistant", content="assistant", images=[image]),
    ]
    assert convert(protocol, messages) == convert(
        protocol, [Message(role=m.role, content=m.content) for m in messages]
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_plain_text_parallel_tool_requests_are_unchanged(protocol: str) -> None:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="checking",
            tool_calls=[
                ToolCall(id="one", name="read", arguments={}),
                ToolCall(id="two", name="read", arguments={}),
            ],
        ),
        Message(role="tool", tool_call_id="one", name="read", content="ok"),
        Message(
            role="tool",
            tool_call_id="two",
            name="read",
            content="failed",
            is_error=True,
        ),
        Message(role="assistant", content="done"),
        Message(role="user", content=""),
    ]
    if protocol == "chat":
        assert convert(protocol, messages) == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                    for call_id in ("one", "two")
                ],
            },
            {"role": "tool", "tool_call_id": "one", "content": "ok"},
            {"role": "tool", "tool_call_id": "two", "content": "failed"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": ""},
        ]
    elif protocol == "responses":
        converted, instructions = _messages_to_responses(messages)
        assert instructions == "system"
        assert converted == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "checking"},
            *[
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "read",
                    "arguments": "{}",
                }
                for call_id in ("one", "two")
            ],
            {"type": "function_call_output", "call_id": "one", "output": "ok"},
            {"type": "function_call_output", "call_id": "two", "output": "failed"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": ""},
        ]
    elif protocol == "anthropic":
        system, converted = _messages_to_anthropic(messages)
        assert system == "system"
        assert converted == [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    *[
                        {"type": "tool_use", "id": call_id, "name": "read", "input": {}}
                        for call_id in ("one", "two")
                    ],
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "one",
                        "content": "ok",
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "two",
                        "content": "failed",
                        "is_error": True,
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": ""},
        ]
    else:
        system, converted = _messages_to_gemini(messages)
        assert system == "system"
        assert converted == [
            {"role": "user", "parts": [{"text": "hello"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "checking"},
                    {"function_call": {"name": "read", "args": {}}},
                    {"function_call": {"name": "read", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "one",
                            "name": "read",
                            "response": {"output": "ok", "is_error": False},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "two",
                            "name": "read",
                            "response": {"output": "failed", "is_error": True},
                        }
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "done"}]},
            {"role": "user", "parts": [{"text": ""}]},
        ]


def test_gemini_sdk_serializes_inline_data_as_original_png(image: ImageContent) -> None:
    from google.genai import types

    messages = [
        Message(role="user", images=[image]),
        Message(role="tool", tool_call_id="call_1", name="browser", images=[image]),
    ]
    _, contents = _messages_to_gemini(messages)
    serialized_images = []
    for content in contents:
        sdk_content = types.Content.model_validate(content)
        wire_content = sdk_content.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        for part in wire_content["parts"]:
            if "inlineData" in part:
                blob = part["inlineData"]
                assert blob["mimeType"] == "image/png"
                serialized_images.append(base64.urlsafe_b64decode(blob["data"]))
    assert serialized_images == [base64.b64decode(PNG_BASE64)] * 2
