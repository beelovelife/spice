from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace as NS

import pytest

from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.providers.openai import OpenAIProvider
from spice.llm.providers.gemini import GeminiProvider
from spice.llm.types import Done, ModelRequestOptions, StreamError, ToolCallEvent


class SDKStream:
    """OpenAI's stream has async close(), not aclose()."""
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("finish", [None, "stop", "tool_calls"])
def test_openai_completion_requires_provider_finish_signal(finish):
    stream = SDKStream([NS(usage=None, choices=[NS(
        finish_reason=finish,
        delta=NS(content="partial", tool_calls=[NS(
            index=0, id="call", function=NS(name="write", arguments="{}"),
        )]),
    )])])

    async def create(**kwargs):
        return stream

    client = NS(chat=NS(completions=NS(create=create)))

    async def run():
        return [event async for event in OpenAIProvider()._astream_chat_completions(
            client, Model(id="fake", provider="openai"), [Message(role="user", content="run")],
            [], ModelRequestOptions(api_key="test"), 0,
        )]

    events = asyncio.run(run())
    assert stream.closed
    if finish is None:
        assert isinstance(events[-1], StreamError)
        assert not any(isinstance(event, (Done, ToolCallEvent)) for event in events)
    else:
        assert isinstance(events[-1], Done)


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", None])
def test_responses_incomplete_preserves_cause_usage_and_limit(reason, caplog, monkeypatch):
    # CLI tests configure Spice's file-only logger earlier in a full-suite run.
    monkeypatch.setattr(logging.getLogger("spice"), "propagate", True)
    stream = SDKStream([NS(
        type="response.incomplete",
        response=NS(
            incomplete_details=NS(reason=reason) if reason else None,
            usage=NS(input_tokens=100, output_tokens=16000, total_tokens=16100),
        ),
    )])

    async def create(**kwargs):
        assert kwargs["max_output_tokens"] == 16000
        return stream

    client = NS(responses=NS(create=create))

    async def run():
        return [event async for event in OpenAIProvider()._astream_responses(
            client, Model(id="fake", provider="openai"), [], [],
            ModelRequestOptions(api_key="test", max_tokens=16000), 0,
        )]

    events = asyncio.run(run())
    assert stream.closed
    assert len(events) == 1
    done = events[0]
    assert isinstance(done, Done) and not done.is_complete
    assert done.finish_reason == "incomplete"
    assert done.incomplete_reason == reason
    assert done.output_token_limit == 16000
    assert done.usage.output_tokens == 16000
    assert f"reason={reason or 'unknown'}" in caplog.text
    assert "max_output_tokens=16000 output_tokens=16000" in caplog.text


@pytest.mark.parametrize("protocol", ["completions", "responses"])
def test_provider_close_releases_underlying_sdk_stream(monkeypatch, protocol):
    chunks = [NS(type="response.output_text.delta", delta="partial")] if protocol == "responses" else [
        NS(usage=None, choices=[NS(finish_reason=None, delta=NS(content="partial", tool_calls=[]))]),
    ]
    stream = SDKStream(chunks)

    async def create(**kwargs):
        return stream

    client = NS(responses=NS(create=create), chat=NS(completions=NS(create=create)))
    monkeypatch.setattr("spice.llm.providers.openai.get_cached_client", lambda *_args: client)

    async def run():
        events = OpenAIProvider(use_responses=protocol == "responses").astream(
            Model(id="fake", provider="openai"), [], [], ModelRequestOptions(api_key="test"),
        )
        await anext(events)
        await events.aclose()
        assert stream.closed

    asyncio.run(run())


@pytest.mark.parametrize("finish", [None, "STOP", "MAX_TOKENS", "SAFETY"])
def test_gemini_preserves_terminal_status_and_closes_stream(monkeypatch, finish):
    stream = SDKStream([NS(usage_metadata=None, candidates=[NS(
        content=NS(parts=[NS(text="partial", function_call=None)]), finish_reason=finish,
    )])])

    async def create(**kwargs):
        return stream

    client = NS(aio=NS(models=NS(generate_content_stream=create)))
    monkeypatch.setattr("spice.llm.providers.gemini.get_cached_client", lambda *_args: client)

    async def run():
        return [event async for event in GeminiProvider().astream(
            Model(id="fake", provider="gemini"), [], [], ModelRequestOptions(api_key="test"),
        )]

    events = asyncio.run(run())
    assert stream.closed
    if finish in {None, "SAFETY"}:
        assert isinstance(events[-1], StreamError)
    else:
        assert events[-1].finish_reason == ("stop" if finish == "STOP" else "max_tokens")
