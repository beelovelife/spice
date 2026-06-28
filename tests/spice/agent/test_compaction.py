from __future__ import annotations

import asyncio
import json

import spice.agent.compaction as compaction_module
from spice.agent.compaction import (
    CompactionSettings,
    check_compaction_needed,
    estimate_messages_tokens,
    prepare_compaction,
)
from spice.agent.sessions import SessionStore
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model


def test_should_compact_when_estimate_exceeds_context_minus_reserve() -> None:
    model = Model(id="tiny", provider="test", context_window=1000, output_tokens=100)
    messages = [Message(role="user", content="x" * 3600)]

    status = check_compaction_needed(messages, model, CompactionSettings(reserve_tokens=200))

    assert status.threshold_tokens == 800
    assert status.estimated_tokens > status.threshold_tokens
    assert status.should_compact


def test_prepare_compaction_uses_entry_boundaries(tmp_path) -> None:
    store = SessionStore(tmp_path)
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    for index in range(10):
        store.append_message(info.id, Message(role="user", content=f"message {index} " + ("x" * 100)))

    entries = store.path_entries(info.id)
    plan = prepare_compaction(entries, CompactionSettings(keep_recent_tokens=80, min_messages=4))

    assert plan is not None
    assert plan.first_kept_entry_id in {entry.id for entry in entries}
    assert plan.messages_to_summarize
    assert plan.kept_messages


def test_prepare_compaction_uses_previous_snake_case_boundary(tmp_path) -> None:
    store = SessionStore(tmp_path)
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    for index in range(8):
        store.append_message(info.id, Message(role="user", content=f"old {index} " + ("x" * 100)))
    first_plan = prepare_compaction(store.path_entries(info.id), CompactionSettings(keep_recent_tokens=60, min_messages=4))
    assert first_plan is not None
    compact_id = store.append_compaction(
        info.id,
        summary="old summary",
        first_kept_entry_id=first_plan.first_kept_entry_id,
        tokens_before=first_plan.tokens_before,
    )
    for index in range(4):
        store.append_message(info.id, Message(role="user", content=f"new {index} " + ("x" * 100)))

    second_plan = prepare_compaction(store.path_entries(info.id), CompactionSettings(keep_recent_tokens=60, min_messages=4))

    assert second_plan is not None
    compact_path_index = [entry.id for entry in store.path_entries(info.id)].index(compact_id)
    first_kept_index = [entry.id for entry in store.path_entries(info.id)].index(first_plan.first_kept_entry_id)
    compacted_contents = [
        entry.data["message"]["content"]
        for entry in store.path_entries(info.id)[:first_kept_index]
        if entry.type == "message"
    ]
    planned_contents = [message.content for message in [*second_plan.messages_to_summarize, *second_plan.kept_messages]]
    assert all(content not in planned_contents for content in compacted_contents)
    assert compact_path_index >= 0


def test_prepare_compaction_does_not_split_tool_call_round(tmp_path) -> None:
    store = SessionStore(tmp_path)
    info = store.create(cwd=tmp_path, provider="openai", model="gpt-5.1")
    store.append_message(info.id, Message(role="user", content="old 1"))
    store.append_message(info.id, Message(role="user", content="old 2"))
    assistant_id = store.append_message(
        info.id,
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "large.txt"})],
        ),
    )
    tool_id = store.append_message(
        info.id,
        Message(role="tool", content="x" * 400, tool_call_id="tc1", name="read_file"),
    )

    plan = prepare_compaction(store.path_entries(info.id), CompactionSettings(keep_recent_tokens=10, min_messages=4))

    assert plan is not None
    assert plan.first_kept_entry_id == assistant_id
    assert [message.role for message in plan.kept_messages] == ["assistant", "tool"]
    assert plan.kept_messages[1].tool_call_id == "tc1"
    assert tool_id in {entry.id for entry in store.path_entries(info.id)}


def test_estimator_counts_tool_call_arguments() -> None:
    plain = estimate_messages_tokens([Message(role="assistant", content="done")])
    with_tool = estimate_messages_tokens(
        [
            Message(
                role="assistant",
                content="done",
                tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "large.json"})],
            )
        ]
    )

    assert with_tool > plain


def test_safe_json_fallback_remains_json() -> None:
    encoded = compaction_module._safe_json({"bad": {1, 2}})

    decoded = json.loads(encoded)
    assert "_unserializable" in decoded
    assert "{1, 2}" in decoded["_unserializable"]
