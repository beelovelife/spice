"""Turn outcomes and branch-aware continuation context, independent of frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from spice.agent.sessions import SessionEntry

TURN_OUTCOME_ENTRY = "turn_outcome"
TurnStatus = Literal["completed", "paused", "failed", "interrupted"]


def turn_status(stop_reason: str) -> TurnStatus:
    if stop_reason == "max_tool_rounds":
        return "paused"
    if stop_reason in {"user_interrupted", "consumer_closed"}:
        return "interrupted"
    if stop_reason in {"error", "context_limit", "fatal_tool_error"}:
        return "failed"
    return "completed"


@dataclass
class TurnCheckpoint:
    objective: str
    rounds_used: int
    round_limit: int
    summary: str = ""
    summary_source: Literal["model", "local"] = "local"
    summary_error: str | None = None
    verification_status: str = "unverified"
    verification_reason: str = "尚无明确执行的验证命令"
    verification_checks: list[dict[str, Any]] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    recent_tools: list[dict[str, Any]] = field(default_factory=list)


def latest_turn_outcome(entries: Iterable[SessionEntry]) -> dict[str, Any] | None:
    outcome = None
    for entry in entries:
        if entry.type == "custom" and entry.data.get("customType") == TURN_OUTCOME_ENTRY:
            outcome = entry.data
    return outcome


def continuation_context(outcome: dict[str, Any] | None) -> str | None:
    if not outcome or outcome.get("status") != "paused":
        return None
    checkpoint = outcome.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    summary = str(checkpoint.get("summary") or outcome.get("text") or "")[:8000]
    return (
        "Previous turn paused after exhausting its tool-round budget. "
        "Completed tool calls and file changes were preserved; do not replay them blindly. "
        "Use this checkpoint to continue from the recorded state. The latest user request "
        "takes precedence if the objective has changed.\n"
        f"Continuation checkpoint:\n{summary}"
    )


def pause_notice(round_limit: int | None = None) -> str:
    budget = f" {round_limit} 轮" if round_limit is not None else "本轮"
    return f"已用完{budget}工具预算，暂停待继续。输入“继续”可接着处理。"
