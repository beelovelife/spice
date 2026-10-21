from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest
from PIL import Image

from spice.agent.compaction import check_compaction_needed, estimate_message_tokens, _serialize_messages
from spice.agent.events import ToolExecutionEndEvent, TurnEndEvent
from spice.agent.file_references import expand_file_references
from spice.agent.loop import run_turn
from spice.agent.sessions import SessionStore, message_from_dict, message_to_dict
from spice.agent.tool_results import prepare_tool_message_for_session, tool_result_message
from spice.agent.turn_journal import TurnJournal, recover_interrupted_tools
from spice.llm.images import load_image
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy
from spice.llm.routing import ModelCandidate, ModelRoute
from spice.llm.stream import stream_model
from spice.llm.types import Done, ModelRequestOptions, StreamError, TextDelta, ToolCallEvent
from spice.storage.sqlite_sessions import SqliteSessionStore
from spice.tools.base import ToolResult
from spice.tools.image import create_image_tools


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "ui.png"
    Image.new("RGB", (40, 20), "green").save(path)
    return path


@pytest.fixture(params=["jsonl", "sqlite"])
def store(tmp_path, request):
    if request.param == "jsonl":
        return SessionStore(tmp_path / "sessions", cwd=tmp_path)
    return SqliteSessionStore(tmp_path / "spice.db", cwd=tmp_path)


def test_image_message_round_trip_and_snapshot(store, tmp_path, image_path):
    image = load_image(image_path)
    result = ToolResult("attached", images=[image])
    message = tool_result_message(ToolCall("tc1", "view_image", {"path": "ui.png"}), result)
    serialized = json.loads(json.dumps(message_to_dict(message)))
    assert message_from_dict(serialized) == message
    info = store.create(cwd=tmp_path, provider="fake", model="vision")
    saved = prepare_tool_message_for_session(message, cwd=tmp_path, session_id=info.id)
    store.append_message(info.id, saved)
    image_path.unlink()
    loaded = store.load_messages(info.id)
    assert loaded[0].images == [image]
    assert loaded[0].tool_call_id == "tc1"


def test_checkpoint_recovers_image_without_rerunning_tool(store, tmp_path, image_path):
    image = load_image(image_path)
    info = store.create(cwd=tmp_path, provider="fake", model="vision")
    journal = TurnJournal(store, info.id, cwd=tmp_path, cursor=0, provider="fake", model="vision")
    call = ToolCall("image1", "view_image", {"path": "ui.png"})
    journal.flush([Message(role="assistant", tool_calls=[call])])
    journal.checkpoint(ToolExecutionEndEvent(call.id, call.name, ToolResult("attached", images=[image])))
    assert recover_interrupted_tools(store, info.id) == 1
    assert recover_interrupted_tools(store, info.id) == 0
    assert store.load_messages(info.id)[-1].images == [image]


def test_older_tool_result_keeps_image_on_resume(store, tmp_path, image_path):
    info = store.create(cwd=tmp_path, provider="fake", model="vision")
    image = load_image(image_path)
    for index in range(5):
        call = ToolCall(str(index), "view_image", {"path": "ui.png"})
        store.append_message(info.id, tool_result_message(call, ToolResult("attached", images=[image])))
    assert all(message.images == [image] for message in store.load_messages(info.id))


def test_user_image_reference_has_real_attachment_and_budget(tmp_path, image_path):
    model = Model(id="vision", provider="fake", supports_vision=True, context_window=128000)
    result = expand_file_references("Inspect @ui.png @ui.png", cwd=tmp_path, model=model)
    assert len(result.images) == 1
    assert 'injected="true"' in result.message
    assert "not enabled" not in result.message
    assert result.images[0].data not in result.message
    model.context_window = 10
    assert not expand_file_references("@ui.png", cwd=tmp_path, model=model).images


def test_user_image_limit_and_corrupt_file(tmp_path, image_path):
    model = Model(id="vision", provider="fake", supports_vision=True, context_window=128000)
    for index in range(6):
        (tmp_path / f"{index}.png").write_bytes(image_path.read_bytes())
    result = expand_file_references(" ".join(f"@{index}.png" for index in range(6)), cwd=tmp_path, model=model)
    assert len(result.images) == 5
    assert "at most 5" in result.message
    (tmp_path / "bad.png").write_text("bad")
    invalid = expand_file_references("@bad.png", cwd=tmp_path, model=model)
    assert invalid.warnings and not invalid.images


def test_trace_records_image_metadata_without_payload(image_path):
    from spice.agent.trace import _message_to_dict
    image = load_image(image_path)
    trace = _message_to_dict(Message(role="user", content="Inspect", images=[image]))
    assert trace["images"][0]["width"] == image.width
    assert "data" not in trace["images"][0]
    assert image.data not in json.dumps(trace)


def test_compaction_accounts_for_images_without_base64_in_summary(image_path):
    image = load_image(image_path)
    message = Message(role="user", content="Inspect", images=[image])
    assert estimate_message_tokens(message) > estimate_message_tokens(Message(role="user", content="Inspect"))
    summary = _serialize_messages([message])
    assert image.source is not None and image.source in summary
    assert image.data not in summary
    assert "pixels omitted" in summary


def test_text_model_rejects_existing_images_before_provider_lookup(image_path, monkeypatch):
    def unexpected_provider(*args, **kwargs):
        raise AssertionError("must reject before provider execution")
    monkeypatch.setattr("spice.llm.stream.get_provider", unexpected_provider)
    async def run():
        return [event async for event in stream_model(Model(id="text", provider="fake"),
            [Message(role="user", images=[load_image(image_path)])], [], ModelRequestOptions())]
    events = asyncio.run(run())
    assert len(events) == 1 and isinstance(events[0], StreamError)
    assert events[0].kind == "unsupported"


def test_session_prompt_passes_user_image_and_recovers_it(store, tmp_path, image_path, monkeypatch):
    from spice.agent import agent_session as session_module
    from spice.agent.agent_session import AgentSession
    from spice.llm.config import SpiceConfig
    seen = []
    async def stream(model, messages, tools, options):
        seen.extend(messages)
        yield TextDelta("Image received")
        yield Done("stop")
    original = session_module.build_model_route
    def route(*args, **kwargs):
        result = original(*args, **kwargs)
        result.stream_factory = stream
        return result
    monkeypatch.setattr(session_module, "build_model_route", route)
    config = SpiceConfig(memory_enabled=False, subagents_enabled=False)
    session = AgentSession(cwd=tmp_path, session_store=store, config=config)
    session.model = Model(id="vision", provider="openai", supports_vision=True)
    async def run():
        return [event async for event in session.prompt("Inspect @ui.png")]
    asyncio.run(run())
    user = next(message for message in seen if message.role == "user")
    assert user.images == [load_image(image_path)]
    assert store.load_messages(session.session_id)[0].images == user.images


def test_agent_loop_delivers_tool_image_to_next_request(tmp_path, image_path, monkeypatch):
    requests = []
    async def stream(model, messages, tools, options):
        requests.append(list(messages))
        if len(requests) == 1:
            yield ToolCallEvent(id="image1", name="view_image", arguments={"path": "ui.png"})
            yield Done("tool_calls")
        else:
            assert messages[-1].role == "tool"
            assert messages[-1].images[0].data == load_image(image_path).data
            yield TextDelta("I see the image")
            yield Done("stop")
    monkeypatch.setattr("spice.agent.loop.stream_model", stream)
    history = []
    async def run():
        return [event async for event in run_turn(prompt="Inspect screenshot", messages=history,
            model=Model(id="vision", provider="fake", supports_vision=True), tools=create_image_tools(),
            options=ModelRequestOptions(), cwd=tmp_path, confirm=None)]
    events = asyncio.run(run())
    assert len(requests) == 2
    result = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
    assert not result.result.is_error and result.result.images
    assert history[-1].content == "I see the image"


def test_images_do_not_accumulate_across_tool_rounds_or_user_turns(tmp_path, image_path, monkeypatch):
    requests = []
    image = load_image(image_path)

    async def stream(model, messages, tools, options):
        requests.append(list(messages))
        if len(requests) == 1:
            for call_id in ("day", "night"):
                yield ToolCallEvent(call_id, "view_image", {"path": "ui.png"})
            yield Done("tool_calls")
        elif len(requests) == 2:
            yield ToolCallEvent("fresh", "view_image", {"path": "ui.png"})
            yield Done("tool_calls")
        else:
            yield TextDelta("Visual findings recorded")
            yield Done("stop")

    monkeypatch.setattr("spice.agent.loop.stream_model", stream)
    history = []

    async def run():
        for prompt, images in (("Inspect", [image]), ("Continue", [])):
            async for _ in run_turn(
                prompt=prompt, images=images, messages=history,
                model=Model(id="vision", provider="fake", supports_vision=True),
                tools=create_image_tools(), options=ModelRequestOptions(), cwd=tmp_path, confirm=None,
            ):
                pass

    asyncio.run(run())

    assert [sum(len(m.images) for m in request) for request in requests] == [1, 2, 1, 0]
    assert [m.tool_call_id for m in requests[1] if m.images] == ["day", "night"]
    assert [m.tool_call_id for m in requests[2] if m.images] == ["fresh"]
    assert sum(len(m.images) for m in history) == 4
    assert all(m.content.startswith("Image attached:") for m in history if m.role == "tool")


@pytest.mark.parametrize("answered", [False, True])
def test_resumed_session_sends_only_unanswered_images(store, tmp_path, image_path, monkeypatch, answered):
    image = load_image(image_path)
    info = store.create(cwd=tmp_path, provider="fake", model="vision")
    for index in range(40):
        call = ToolCall(f"image-{index}", "view_image", {"path": "ui.png"})
        store.append_message(info.id, Message(role="assistant", tool_calls=[call]))
        store.append_message(info.id, tool_result_message(call, ToolResult("attached", images=[image])))
    if answered:
        store.append_message(info.id, Message(role="assistant", content="Visual findings"))
    history = store.load_messages(info.id)
    requests = []

    async def stream(model, messages, tools, options):
        requests.append(list(messages))
        yield Done("stop")

    monkeypatch.setattr("spice.agent.loop.stream_model", stream)

    async def run():
        return [event async for event in run_turn(
            prompt="Continue", messages=history,
            model=Model(id="vision", provider="fake", supports_vision=True), tools=[],
            options=ModelRequestOptions(), cwd=tmp_path, confirm=None,
        )]

    asyncio.run(run())

    assert sum(len(m.images) for m in requests[0]) == (0 if answered else 1)
    assert sum(len(m.images) for m in history) == 40
    assert sum(len(m.images) for m in store.load_messages(info.id)) == 40


@pytest.mark.parametrize("fallback", [False, True])
def test_pending_images_survive_retry_and_fallback(tmp_path, image_path, fallback):
    image = load_image(image_path)
    model = Model(id="primary", provider="fake", supports_vision=True)
    backup = replace(model, id="backup")
    options = ModelRequestOptions()
    requests = []

    async def stream(candidate, messages, *_args):
        requests.append((candidate.id, list(messages)))
        if len(requests) == 1:
            yield StreamError("temporary failure", kind="network", retryable=True)
        else:
            yield Done("stop")

    route = ModelRoute(
        [ModelCandidate(m.id, m, options) for m in (model, backup)],
        retry_policy=ModelRetryPolicy(max_attempts=1 if fallback else 2, base_delay_ms=0),
        fallback_enabled=fallback, stream_factory=stream,
    )
    history = [Message(role="user", images=[image]), Message(role="assistant", content="Old findings")]

    async def run():
        return [event async for event in run_turn(
            prompt="Inspect again", images=[image], messages=history, model=model,
            tools=[], options=options, model_route=route, cwd=tmp_path, confirm=None,
        )]

    asyncio.run(run())

    assert [candidate for candidate, _ in requests] == ["primary", "backup" if fallback else "primary"]
    assert [sum(len(m.images) for m in request) for _, request in requests] == [1, 1]
    assert sum(len(m.images) for m in history) == 2


def test_context_budget_ignores_old_images_but_counts_pending_images(image_path):
    image = replace(load_image(image_path), width=2048, height=2048)
    model = Model(id="small", provider="fake", context_window=3000, output_tokens=100)
    history = [Message(role="user", images=[image]), Message(role="assistant", content="Findings")]

    assert not check_compaction_needed(history, model).should_compact
    assert check_compaction_needed([*history, Message(role="user", images=[image])], model).should_compact


def test_compaction_rebuild_does_not_restore_old_image_payloads(tmp_path, image_path, monkeypatch):
    image = replace(load_image(image_path), width=2048, height=2048)
    model = Model(id="small", provider="fake", context_window=3000, output_tokens=100)
    history = [
        Message(role="user", content="Reference", images=[image]),
        Message(role="assistant", content="Findings"),
        Message(role="user", content="x" * 20000),
    ]
    requests, compactions = [], []

    async def compact(*_args):
        compactions.append(True)
        history[:] = [*history[:2], Message(role="user", content="Compacted context")]

    async def stream(_model, messages, *_args):
        requests.append(list(messages))
        yield Done("stop")

    monkeypatch.setattr("spice.agent.loop.stream_model", stream)

    async def run():
        return [event async for event in run_turn(
            prompt="Continue", messages=history, model=model, tools=[], options=ModelRequestOptions(),
            cwd=tmp_path, confirm=None, compact_context=compact,
        )]

    events = asyncio.run(run())

    assert compactions == [True]
    assert len(requests) == 1 and not any(m.images for m in requests[0])
    assert history[0].images == [image]
    assert isinstance(events[-1], TurnEndEvent)
    assert events[-1].stop_reason == "stop"


@pytest.mark.parametrize("supports_vision", [False, True])
def test_provider_dispatch_omits_old_images_without_changing_history(image_path, monkeypatch, supports_vision):
    image = load_image(image_path)
    history = [Message(role="user", images=[image]), Message(role="assistant", content="Findings")]
    requests = []

    class Provider:
        async def astream(self, model, messages, tools, options):
            requests.append(messages)
            yield Done("stop")

    monkeypatch.setattr("spice.llm.stream.get_provider", lambda *args, **kwargs: Provider())

    async def run():
        return [event async for event in stream_model(
            Model(id="model", provider="fake", supports_vision=supports_vision),
            history, [], ModelRequestOptions(),
        )]

    events = asyncio.run(run())

    assert isinstance(events[-1], Done)
    assert len(requests) == 1 and not any(m.images for m in requests[0])
    assert history[0].images == [image]
