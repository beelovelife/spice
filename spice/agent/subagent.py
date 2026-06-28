"""Subagent execution support."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from spice.agent.events import AgentErrorEvent, AssistantMessageEvent, ToolExecutionEndEvent, TurnEndEvent
from spice.agent.prompts import build_system_prompt
from spice.llm.config import load_config
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.llm.types import ModelRequestOptions
from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.tools.base import ConfirmFn, Tool, truncate_tail
from spice.tools.file_state import FileStateStore

MAX_CONCURRENT_SUBAGENTS = 3
SUBAGENT_MAX_TOOL_ROUNDS = 15
SUBAGENT_RESULT_CHARS = 50_000


async def run_turn(**kwargs):
    from spice.agent.loop import run_turn as agent_run_turn

    async for event in agent_run_turn(**kwargs):
        yield event


@dataclass(slots=True)
class SubagentTask:
    task: str
    label: str | None = None


@dataclass(slots=True)
class SubagentResult:
    task_id: str
    label: str
    task: str
    status: str
    result: str
    rounds: int = 0
    tool_count: int = 0
    duration_ms: int = 0
    error: str | None = None
    truncated: bool = False
    original_chars: int = 0

    def to_details(self) -> dict:
        return {
            "task_id": self.task_id,
            "label": self.label,
            "task": self.task,
            "status": self.status,
            "result": self.result,
            "rounds": self.rounds,
            "tool_count": self.tool_count,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "truncated": self.truncated,
            "original_chars": self.original_chars,
        }


@dataclass(slots=True)
class SubagentRunSummary:
    results: list[SubagentResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for result in self.results if result.status == "ok")

    def to_text(self) -> str:
        total = len(self.results)
        lines = [f"Subagents completed: {self.success_count}/{total} succeeded"]
        for result in self.results:
            lines.extend(
                [
                    "",
                    f"## {result.label}",
                    f"Status: {result.status}",
                    f"Task: {result.task}",
                    "",
                    result.result or "(no output)",
                ]
            )
            if result.truncated:
                lines.append("")
                lines.append(f"[subagent output truncated from {result.original_chars} chars]")
        return "\n".join(lines)

    def to_details(self) -> dict:
        return {
            "subagents": [result.to_details() for result in self.results],
            "success_count": self.success_count,
            "total": len(self.results),
        }


class SubagentManager:
    """Run isolated subagents for independent tasks."""

    def __init__(
        self,
        *,
        cwd: Path,
        model: Model,
        options_factory: Callable[[], ModelRequestOptions],
        confirm: ConfirmFn | None = None,
        tools_factory: Callable[[], list[Tool]],
        max_concurrent: int = MAX_CONCURRENT_SUBAGENTS,
        max_tool_rounds: int = SUBAGENT_MAX_TOOL_ROUNDS,
        result_chars: int = SUBAGENT_RESULT_CHARS,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.options_factory = options_factory
        self.confirm = confirm
        self.tools_factory = tools_factory
        self.max_concurrent = max_concurrent
        self.max_tool_rounds = max_tool_rounds
        self.result_chars = result_chars
        config = load_config()
        self.workspace_policy = create_workspace_policy(config.sandbox, cwd=self.cwd)
        self.environment = create_environment(config.sandbox, cwd=self.cwd)

    def set_model(self, model: Model) -> None:
        self.model = model

    async def run_many(self, tasks: list[SubagentTask]) -> SubagentRunSummary:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_limited(task: SubagentTask, index: int) -> SubagentResult:
            async with semaphore:
                return await self.run_one(task, index=index)

        results = await asyncio.gather(*(run_limited(task, index) for index, task in enumerate(tasks)))
        return SubagentRunSummary(results=list(results))

    async def run_one(self, task: SubagentTask, *, index: int = 0) -> SubagentResult:
        task_id = uuid4().hex[:8]
        label = _clean_label(task.label) or f"subagent-{index + 1}"
        started = time.perf_counter()
        messages = [Message(role="system", content=self._build_system_prompt())]
        final_text = ""
        rounds = 0
        tool_count = 0
        error: str | None = None

        try:
            async for event in run_turn(
                prompt=task.task,
                messages=messages,
                model=self.model,
                tools=self._subagent_tools(),
                options=self.options_factory(),
                cwd=self.cwd,
                workspace=self.workspace_policy,
                environment=self.environment,
                confirm=self.confirm,
                file_states=FileStateStore(),
                session_label=f"subagent:{label}:{task_id}",
                max_tool_rounds=self.max_tool_rounds,
            ):
                if isinstance(event, AssistantMessageEvent):
                    rounds += 1
                    tool_count += len(event.tool_calls)
                    if event.text:
                        final_text = event.text
                elif isinstance(event, TurnEndEvent):
                    final_text = event.text
                elif isinstance(event, AgentErrorEvent):
                    error = event.message
        except Exception as exc:
            error = str(exc)

        duration_ms = int((time.perf_counter() - started) * 1000)
        status = "error" if error else "ok"
        raw_result = error or final_text or "Task completed but no final response was generated."
        result_text, truncated, original_chars = _truncate_subagent_result(raw_result, self.result_chars)
        return SubagentResult(
            task_id=task_id,
            label=label,
            task=task.task,
            status=status,
            result=result_text,
            rounds=rounds,
            tool_count=tool_count,
            duration_ms=duration_ms,
            error=error,
            truncated=truncated,
            original_chars=original_chars,
        )

    def _subagent_tools(self) -> list[Tool]:
        return [
            tool
            for tool in self.tools_factory()
            if tool.name not in {"spawn_subagents", "update_todo"}
        ]

    def _build_system_prompt(self) -> str:
        base = build_system_prompt(
            self.cwd,
            self._subagent_tools(),
            runtime_model=f"{self.model.provider}/{self.model.id}",
        )
        subagent_instructions = (
            "Subagent instructions:\n"
            "You are a subagent started by the main agent. Your only responsibility is to complete "
            "the specific task assigned to you.\n"
            "You run in an isolated context: do not assume access to the full main conversation "
            "history. Rely only on the assigned task, workspace files, and available tools.\n"
            "Your final response will be returned to the main agent, not directly to the user. "
            "Briefly state what you did, what you found, what evidence supports it, and any "
            "remaining uncertainty."
        )
        return f"{base}\n\n{subagent_instructions}"


def _clean_label(label: str | None) -> str | None:
    if not label:
        return None
    cleaned = " ".join(label.strip().split())
    return cleaned[:60] or None


def _truncate_subagent_result(text: str, limit: int) -> tuple[str, bool, int]:
    original_chars = len(text)
    if original_chars <= limit:
        return text, False, original_chars
    return truncate_tail(text, limit), True, original_chars
