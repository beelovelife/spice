"""Subagent tools."""

from __future__ import annotations

from typing import Any

from spice.agent.subagent import MAX_CONCURRENT_SUBAGENTS, SubagentTask
from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result


def create_subagent_tool() -> Tool:
    async def spawn_subagents(args: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.subagent_manager is None:
            return tool_error("Subagents are not available in this runtime.")

        raw_tasks = args.get("tasks")
        if not isinstance(raw_tasks, list):
            return tool_error("tasks must be an array.")
        if len(raw_tasks) > MAX_CONCURRENT_SUBAGENTS:
            return tool_error(f"spawn_subagents accepts at most {MAX_CONCURRENT_SUBAGENTS} tasks.")

        tasks: list[SubagentTask] = []
        for index, item in enumerate(raw_tasks):
            if not isinstance(item, dict):
                return tool_error(f"tasks[{index}] must be an object.")
            task = item.get("task")
            if not isinstance(task, str) or not task.strip():
                return tool_error(f"tasks[{index}].task must be a non-empty string.")
            label = item.get("label")
            if label is not None and not isinstance(label, str):
                return tool_error(f"tasks[{index}].label must be a string.")
            tasks.append(SubagentTask(task=task.strip(), label=label.strip() if isinstance(label, str) else None))

        summary = await context.subagent_manager.run_many(tasks)
        is_error = summary.success_count != len(summary.results)
        return ToolResult(content=summary.to_text(), is_error=is_error, details=summary.to_details())

    return Tool(
        name="spawn_subagents",
        description=(
            "Run up to 3 isolated subagents concurrently, with each subagent handling one independent "
            "task, then return a structured summary of all results to the main agent. Use this only "
            "when the subtasks are independent, can run concurrently, and can be summarized by the "
            "main agent afterward. Do not use it for chained workflows where later tasks depend on "
            "earlier outputs. 中文：并发运行最多 3 个隔离的子 agent，每个子 agent 负责一个独立子任务，"
            "并在全部完成后把结构化结果汇总返回给主 agent。仅在多个子任务彼此独立、可以并发完成并由主 "
            "agent 汇总时使用。不要用它执行必须按顺序依赖前一步输出的链式任务。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_CONCURRENT_SUBAGENTS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "minLength": 1},
                            "task": {"type": "string", "minLength": 1},
                        },
                        "required": ["task"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
        execute=spawn_subagents,
    )
