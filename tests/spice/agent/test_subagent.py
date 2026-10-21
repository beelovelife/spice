from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path

import spice.agent.subagent as subagent_module
from spice.agent.events import AgentErrorEvent, AssistantMessageEvent, TurnEndEvent
from spice.agent.subagent import SubagentManager, SubagentTask
from spice.agent.agent_session import AgentSession
from spice.agent.sessions import SessionStore
from spice.llm.config import SpiceConfig
from spice.llm.models import Model
from spice.llm.types import ModelRequestOptions
from spice.tools.base import Tool, ToolContext, tool_result
from spice.tools.subagent import create_subagent_tool


async def _noop(args: dict, context: ToolContext):
    return tool_result("ok")


class SubagentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_run_turn = subagent_module.run_turn

    async def asyncTearDown(self) -> None:
        subagent_module.run_turn = self.original_run_turn

    def _manager(self, tools: list[Tool] | None = None) -> SubagentManager:
        return SubagentManager(
            cwd=Path.cwd(),
            model=Model(id="fake", provider="fake"),
            options_factory=ModelRequestOptions,
            tools_factory=lambda: tools or [],
        )

    async def test_spawn_subagents_rejects_more_than_three_tasks(self) -> None:
        tool = create_subagent_tool()
        result = await tool.execute(
            {"tasks": [{"task": "one"}, {"task": "two"}, {"task": "three"}, {"task": "four"}]},
            ToolContext(cwd=Path.cwd(), subagent_manager=self._manager()),
        )

        self.assertTrue(result.is_error)
        self.assertIn("at most 3 tasks", result.content)

    async def test_spawn_subagents_aggregates_success_and_failure(self) -> None:
        async def fake_run_turn(**kwargs):
            prompt = kwargs["prompt"]
            if prompt == "bad":
                yield AgentErrorEvent("boom")
                return
            yield AssistantMessageEvent(text=f"done {prompt}", tool_calls=[])
            yield TurnEndEvent(text=f"done {prompt}")

        subagent_module.run_turn = fake_run_turn
        tool = create_subagent_tool()
        result = await tool.execute(
            {"tasks": [{"label": "ok", "task": "good"}, {"label": "fail", "task": "bad"}]},
            ToolContext(cwd=Path.cwd(), subagent_manager=self._manager()),
        )

        self.assertTrue(result.is_error)
        self.assertIn("Subagents completed: 1/2 succeeded", result.content)
        self.assertIn("## ok", result.content)
        self.assertIn("done good", result.content)
        self.assertIn("## fail", result.content)
        self.assertIn("boom", result.content)
        self.assertEqual(result.details["total"], 2)

    async def test_run_many_executes_tasks_concurrently(self) -> None:
        async def fake_run_turn(**kwargs):
            await asyncio.sleep(0.05)
            yield TurnEndEvent(text=kwargs["prompt"])

        subagent_module.run_turn = fake_run_turn
        manager = self._manager()
        started = time.perf_counter()

        summary = await manager.run_many(
            [
                SubagentTask("one"),
                SubagentTask("two"),
                SubagentTask("three"),
            ]
        )

        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.13)
        self.assertEqual([result.result for result in summary.results], ["one", "two", "three"])

    async def test_budget_pause_is_reported_to_parent_as_unfinished(self) -> None:
        async def fake_run_turn(**kwargs):
            yield AssistantMessageEvent("已保存进度，尚待验证。", [])
            yield TurnEndEvent("已保存进度，尚待验证。", stop_reason="max_tool_rounds")

        subagent_module.run_turn = fake_run_turn
        summary = await self._manager().run_many([SubagentTask("创建场景")])
        self.assertEqual(summary.success_count, 0)
        self.assertEqual(summary.results[0].status, "paused")
        self.assertIn("尚待验证", summary.results[0].result)

    async def test_subagent_tools_exclude_spawn_and_todo(self) -> None:
        tools = [
            Tool("spawn_subagents", "spawn", {"type": "object"}, _noop),
            Tool("update_todo", "todo", {"type": "object"}, _noop),
            Tool("read_file", "read", {"type": "object"}, _noop),
        ]
        manager = self._manager(tools)

        self.assertEqual([tool.name for tool in manager._subagent_tools()], ["read_file"])


def test_subagents_inherit_parent_policy_environment_and_active_capabilities(tmp_path, monkeypatch):
    config = SpiceConfig(subagents_enabled=True)
    config.tools["max_concurrency"] = 1
    config.sandbox["workspace"]["max_write_bytes"] = 17
    config.model_routing["retry"]["maxAttempts"] = 1

    def unexpected_config_reload():
        raise AssertionError("A child must not reload global settings behind its parent's policy.")

    monkeypatch.setattr(subagent_module, "load_config", unexpected_config_reload)
    session = AgentSession(cwd=tmp_path, session_store=SessionStore(tmp_path / "sessions"), config=config)
    manager = session.subagent_manager
    assert manager.config is config
    assert manager.workspace_policy is session.workspace_policy
    assert manager.environment is session.environment
    assert manager.registry is session.registry
    session._edit_tools = [Tool("custom", "custom", {"type": "object"}, _noop)]
    assert [tool.name for tool in manager._subagent_tools()] == ["custom"]
    received = []

    async def fake_run_turn(**kwargs):
        received.append(kwargs)
        yield TurnEndEvent("done")

    monkeypatch.setattr(subagent_module, "run_turn", fake_run_turn)
    summary = asyncio.run(manager.run_many([SubagentTask("inspect")]))
    assert summary.success_count == 1
    assert received[0]["workspace"] is session.workspace_policy
    assert received[0]["environment"] is session.environment
    assert received[0]["tools_settings"]["max_concurrency"] == 1
    assert received[0]["model_route"].retry_policy.max_attempts == 1


def test_cancelling_subagent_group_waits_for_children_to_stop(tmp_path, monkeypatch):
    started = asyncio.Event()
    stopped = []

    async def fake_run_turn(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(kwargs["prompt"])
        yield TurnEndEvent("unreachable")

    monkeypatch.setattr(subagent_module, "run_turn", fake_run_turn)
    manager = SubagentManager(cwd=tmp_path, model=Model(id="fake", provider="fake"),
                              options_factory=ModelRequestOptions, tools_factory=lambda: [])

    async def run():
        group = asyncio.create_task(manager.run_many([SubagentTask("one"), SubagentTask("two")]))
        await started.wait()
        group.cancel()
        try:
            await group
        except asyncio.CancelledError:
            pass
        assert sorted(stopped) == ["one", "two"]

    asyncio.run(run())
