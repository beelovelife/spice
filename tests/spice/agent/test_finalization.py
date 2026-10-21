from __future__ import annotations

import asyncio
import json

import pytest

import spice.agent.finalization as finalization
from spice.agent.events import (
    AgentErrorEvent, AssistantMessageEvent, TextDeltaEvent,
    ToolExecutionEndEvent, TurnEndEvent, TurnFinalizationStartEvent,
)
from spice.agent.loop import run_turn
from spice.agent.sessions import message_from_dict, message_to_dict
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.retry import ModelRetryPolicy
from spice.llm.routing import ModelCandidate, ModelRoute
from spice.llm.types import Done, ModelRequestOptions, StreamError, TextDelta, ToolCallEvent
from spice.llm.usage import TokenUsage, usage_scope
from spice.agent.tool_results import tool_result_message
from spice.llm.messages import ToolCall
from spice.tools.base import Tool, ToolContext, tool_result
from spice.tools.file import apply_patch, edit_file, write_file
from spice.tools.verification import verification_label


@pytest.mark.parametrize("ending", ["done", "error", "exception", "eof", "length", "tool", "empty", "timeout"])
def test_budget_has_one_tool_free_final_request_and_safe_fallback(tmp_path, monkeypatch, ending):
    requests, executions, records, closed = [], [], [], []
    monkeypatch.setattr(finalization, "FINALIZATION_TIMEOUT_SECONDS", 0.02)
    model = Model(id="model", provider="fake", context_window=100000, output_tokens=64000)

    async def stream(model, messages, tools, options):
        requests.append((tools, options, messages))
        if tools:
            yield ToolCallEvent(f"call-{len(requests)}", "write", {"path": "result.txt"})
            yield Done("tool_calls", TokenUsage(10, 2))
            return
        try:
            assert options.max_tokens <= finalization.FINALIZATION_MAX_TOKENS
            if ending == "done":
                yield TextDelta("已写入文件。剩余工作：运行验收。")
                yield Done("stop", TokenUsage(20, 5))
            elif ending == "error":
                yield TextDelta("Do not display this partial summary")
                yield StreamError("api_key=private-value", retryable=True)
            elif ending == "exception":
                raise ConnectionError("api_key=private-value")
            elif ending == "length":
                yield TextDelta("Do not display this truncated summary")
                yield Done("length", TokenUsage(20, 20))
            elif ending == "tool":
                yield ToolCallEvent("forbidden", "write", {"path": "unexpected.txt"})
                yield Done("tool_calls")
            elif ending == "empty":
                yield Done("stop")
            elif ending == "timeout":
                await asyncio.Event().wait()
        finally:
            closed.append(True)

    async def execute(args, context):
        executions.append(args["path"])
        (tmp_path / args["path"]).write_text("saved")
        return tool_result("Wrote result.txt")

    route = ModelRoute(
        [ModelCandidate("main", model, ModelRequestOptions(max_tokens=64000))],
        retry_policy=ModelRetryPolicy(max_attempts=3, base_delay_ms=0),
        fallback_enabled=False, stream_factory=stream,
    )
    messages = [Message(role="system", content="rules")]

    async def run():
        with usage_scope(records.append):
            return [event async for event in run_turn(
                prompt="创建文件", messages=messages, model=model,
                tools=[Tool("write", "write", {"type": "object"}, execute)],
                options=route.actual.options, model_route=route,
                cwd=tmp_path, confirm=None, max_tool_rounds=2,
                finalization_context=lambda: {
                    "pending_tasks": ["运行验收"],
                    "verification": {"status": "unverified", "reason": "未执行检查"},
                },
            )]

    events = asyncio.run(run())
    assert len(requests) == 3
    assert requests[-1][0] == []
    assert executions == ["result.txt", "result.txt"]
    assert closed == [True]
    assert (tmp_path / "result.txt").read_text() == "saved"
    assert len([e for e in events if isinstance(e, ToolExecutionEndEvent)]) == 2
    assert len([e for e in events if isinstance(e, TurnFinalizationStartEvent)]) == 1
    assert not any(isinstance(e, AgentErrorEvent) for e in events)
    end = events[-1]
    assert isinstance(end, TurnEndEvent) and end.status == "paused"
    assert end.stop_reason == "max_tool_rounds"
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == [end.text]
    assert end.checkpoint.pending_tasks == ["运行验收"]
    assert end.checkpoint.verification_status == "unverified"
    assert end.checkpoint.rounds_used == 2
    assert "private-value" not in end.text
    assert "Do not display" not in end.text
    assert messages[-1].content == end.text and not messages[-1].tool_calls
    assert isinstance(events[-2], AssistantMessageEvent) and events[-2].tool_calls == []
    assert [r.source for r in records] == ["main", "main", "finalization"]
    assert end.checkpoint.summary_source == ("model" if ending == "done" else "local")
    assert bool(end.checkpoint.summary_error) == (ending != "done")
    if ending == "done":
        assert records[-1].tokens.output_tokens == 5
        assert messages[-1].metadata["usage"]["source"] == "finalization"


def test_finalization_uses_selected_fallback_model_and_its_options(tmp_path):
    primary = Model(id="primary", provider="fake")
    fallback = Model(id="fallback", provider="fake")
    calls = []

    async def stream(model, messages, tools, options):
        calls.append((model.id, bool(tools), options.base_url))
        if model is primary:
            yield StreamError("unavailable", kind="server", retryable=True)
        elif tools:
            yield ToolCallEvent("call", "read", {})
            yield Done("tool_calls")
        else:
            yield TextDelta("Continue after checking results")
            yield Done("stop")

    async def execute(*_args):
        return tool_result("read completed")

    route = ModelRoute(
        [ModelCandidate(m.id, m, ModelRequestOptions(base_url=f"https://{m.id}.test"))
         for m in (primary, fallback)],
        retry_policy=ModelRetryPolicy(enabled=False), fallback_enabled=True, stream_factory=stream,
    )

    async def run():
        return [e async for e in run_turn(
            prompt="work", messages=[], model=primary, model_route=route,
            tools=[Tool("read", "read", {"type": "object"}, execute)],
            options=route.actual.options, cwd=tmp_path, confirm=None, max_tool_rounds=1,
        )]

    events = asyncio.run(run())
    assert calls == [
        ("primary", True, "https://primary.test"),
        ("fallback", True, "https://fallback.test"),
        ("fallback", False, "https://fallback.test"),
    ]
    assert events[-1].checkpoint.summary_source == "model"


def test_small_context_falls_back_without_compaction_or_an_oversized_summary_request(tmp_path):
    calls = []
    model = Model(id="tiny", provider="fake", context_window=1000, output_tokens=100)

    async def stream(*args):
        calls.append(args)
        yield Done("stop")

    route = ModelRoute(
        [ModelCandidate("tiny", model, ModelRequestOptions(max_tokens=100))],
        retry_policy=ModelRetryPolicy(enabled=False), fallback_enabled=False, stream_factory=stream,
    )

    async def run():
        return [e async for e in finalization.finalize_tool_budget(
            messages=[], prompt="x" * 3000, route=route, round_limit=1,
        )]

    events = asyncio.run(run())
    assert calls == []
    assert events[-1].status == "paused"
    assert events[-1].checkpoint.summary_source == "local"


def test_normal_completion_or_failure_does_not_make_an_extra_summary_request(tmp_path):
    async def run(failed):
        requests = []
        model = Model(id="model", provider="fake")

        async def stream(*_args):
            requests.append(True)
            if failed:
                yield StreamError("failed")
            else:
                yield TextDelta("done")
                yield Done("stop")

        route = ModelRoute(
            [ModelCandidate("main", model, ModelRequestOptions())],
            retry_policy=ModelRetryPolicy(enabled=False), fallback_enabled=False, stream_factory=stream,
        )
        events = [e async for e in run_turn(
            prompt="hello", messages=[], model=model, model_route=route, tools=[],
            options=route.actual.options, cwd=tmp_path, confirm=None,
        )]
        assert requests == [True]
        assert events[-1].status == ("failed" if failed else "completed")
        assert not any(isinstance(e, TurnFinalizationStartEvent) for e in events)

    asyncio.run(run(False))
    asyncio.run(run(True))


@pytest.mark.parametrize("status", ["verified", "failed", "stale", "unverified"])
def test_final_summary_keeps_recorded_changes_todos_and_verification(status):
    requests = []
    model = Model(id="model", provider="fake", context_window=100000)

    async def stream(model, messages, tools, options):
        requests.append(json.loads(messages[-1].content))
        yield TextDelta("已保存代码，继续处理剩余事项。")
        yield Done("stop")

    def result(name, args, *, error=False, details=None):
        message = tool_result_message(ToolCall("id", name, args), tool_result("result", details or {}))
        message.is_error = error
        return message

    messages = [
        result("write_file", {"path": "scene.js"}, details={"changed": True, "changed_paths": ["scene.js"]}),
        result("edit_file", {"path": "./scene.js"}, details={"changed": True, "changed_paths": ["scene.js"]}),
        result("apply_patch", {"operations": [{"path": "index.html"}, {"path": "style.css"}]},
               details={"changed": True, "changed_paths": ["index.html", "style.css"]}),
        result("edit_file", {"path": "preview.js", "dry_run": True}),
        result("edit_file", {"path": "preview2.js"}, details={"dry_run": True}),
        result("write_file", {"path": "failed.js"}, error=True),
        *[result("bash", {"command": f"screenshot {i}"}) for i in range(15)],
    ]
    route = ModelRoute(
        [ModelCandidate("main", model, ModelRequestOptions())],
        retry_policy=ModelRetryPolicy(enabled=False), fallback_enabled=False, stream_factory=stream,
    )

    async def run():
        return [e async for e in finalization.finalize_tool_budget(
            messages=messages, prompt="创建场景", route=route, round_limit=100,
            context_factory=lambda: {
                "completed_tasks": ["创建场景代码"], "pending_tasks": ["视觉验收"],
                "verification": {"status": status, "reason": "指定验证证据"},
            },
        )]

    end = asyncio.run(run())[-1]
    assert end.checkpoint.changed_files == ["scene.js", "index.html", "style.css"]
    assert end.checkpoint.completed_tasks == ["创建场景代码"]
    assert len(end.checkpoint.recent_tools) == 12
    assert requests[0]["changed_files"] == end.checkpoint.changed_files
    assert requests[0]["completed_tasks"] == ["创建场景代码"]
    assert requests[0]["verification"]["status"] == status
    assert f"验证记录：{verification_label(status)}；指定验证证据" in end.text
    assert "未完成待办：\n- 视觉验收" in end.text


@pytest.mark.parametrize("name", ["write_file", "edit_file", "apply_patch"])
@pytest.mark.parametrize("evidence", ["missing", "count_only", "noop", "argument_preview", "result_preview", "error", "metadata_error"])
def test_checkpoint_does_not_infer_changes_from_requested_paths(name, evidence):
    args = {"path": "unknown.txt", "operations": [{"path": "unknown.txt"}]}
    details = {"changed": True, "changed_paths": ["unknown.txt"]}
    if evidence == "missing":
        details = {}
    elif evidence == "count_only":
        details = {"changed": True, "files_changed": 1}
    elif evidence == "noop":
        details["changed"] = False
    elif evidence == "argument_preview":
        args["dry_run"] = True
    elif evidence == "result_preview":
        details["dry_run"] = True
    message = tool_result_message(ToolCall("call", name, args), tool_result("Recorded result", details))
    if evidence == "error":
        message.is_error = True
    elif evidence == "metadata_error":
        message.metadata["tool_result"]["is_error"] = True
    before = json.dumps(message_to_dict(message))

    checkpoint = finalization._checkpoint([message], "work", 1, None)

    assert checkpoint.changed_files == []
    assert checkpoint.recent_tools[0]["tool"] == name
    assert checkpoint.recent_tools[0]["result"] == "Recorded result"
    assert "已记录文件变更" not in finalization._local_summary(checkpoint)
    assert json.dumps(message_to_dict(message)) == before


@pytest.mark.parametrize("ending", ["done", "error"])
def test_finalization_uses_real_changes_after_message_roundtrip(tmp_path, ending):
    unchanged = tmp_path / "unchanged.txt"
    deleted = tmp_path / "deleted.txt"
    unchanged.write_text("original")
    deleted.write_text("obsolete")
    context = ToolContext(cwd=tmp_path)
    requests = []
    model = Model(id="model", provider="fake", context_window=100000)

    async def stream(model, messages, tools, options):
        requests.append(json.loads(messages[-1].content))
        if ending == "error":
            yield StreamError("unavailable")
        else:
            yield TextDelta("已记录文件操作，等待验收。")
            yield Done("stop")

    route = ModelRoute(
        [ModelCandidate("main", model, ModelRequestOptions())],
        retry_policy=ModelRetryPolicy(enabled=False), fallback_enabled=False, stream_factory=stream,
    )

    async def run():
        messages = []
        for execute, args in [
            (write_file, {"path": "unchanged.txt", "content": "original"}),
            (edit_file, {"path": "unchanged.txt", "old_text": "original", "new_text": "original"}),
            (edit_file, {"path": "unchanged.txt", "old_text": "original", "new_text": "preview", "dry_run": True}),
            (apply_patch, {"operations": [
                {"type": "replace", "path": "unchanged.txt", "old_text": "original", "new_text": "original"},
                {"type": "add", "path": "new.txt", "content": "created"},
                {"type": "delete", "path": "deleted.txt"},
            ]}),
            (edit_file, {"path": "./new.txt", "old_text": "created", "new_text": "updated"}),
            (apply_patch, {"operations": [{"type": "add", "path": "preview.txt", "content": "preview"}], "dry_run": True}),
            (apply_patch, {"operations": [
                {"type": "add", "path": "failed.txt", "content": "not applied"},
                {"type": "replace", "path": "unchanged.txt", "old_text": "missing", "new_text": "invalid"},
            ]}),
        ]:
            result = await execute(args, context)
            message = tool_result_message(ToolCall(str(len(messages)), execute.__name__, args), result)
            messages.append(message_from_dict(json.loads(json.dumps(message_to_dict(message)))))
        return [e async for e in finalization.finalize_tool_budget(
            messages=messages, prompt="更新文件", route=route, round_limit=100,
        )]

    end = asyncio.run(run())[-1]
    expected = [str((tmp_path / "new.txt").resolve()), str(deleted.resolve())]
    assert end.checkpoint.changed_files == expected
    assert requests[0]["changed_files"] == expected
    assert unchanged.read_text() == "original"
    assert (tmp_path / "new.txt").read_text() == "updated"
    assert not deleted.exists()
    assert not (tmp_path / "preview.txt").exists()
    assert not (tmp_path / "failed.txt").exists()
    assert end.checkpoint.summary_source == ("local" if ending == "error" else "model")
    if ending == "error":
        change_line = next(line for line in end.text.splitlines() if "已记录文件变更" in line)
        assert change_line == "- 已记录文件变更：" + "、".join(expected)
