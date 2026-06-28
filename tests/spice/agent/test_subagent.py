from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path

import spice.agent.subagent as subagent_module
from spice.agent.events import AgentErrorEvent, AssistantMessageEvent, TurnEndEvent
from spice.agent.subagent import SubagentManager, SubagentTask
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

    async def test_subagent_tools_exclude_spawn_and_todo(self) -> None:
        tools = [
            Tool("spawn_subagents", "spawn", {"type": "object"}, _noop),
            Tool("update_todo", "todo", {"type": "object"}, _noop),
            Tool("read_file", "read", {"type": "object"}, _noop),
        ]
        manager = self._manager(tools)

        self.assertEqual([tool.name for tool in manager._subagent_tools()], ["read_file"])
