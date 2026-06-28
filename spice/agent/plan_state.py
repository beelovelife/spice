"""Session-local plan/edit mode state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

InteractionMode = Literal["edit", "plan"]
ApprovalMode = Literal["manual", "auto"]

PLAN_STATE_ENTRY = "plan_state"


@dataclass
class PlanState:
    mode: InteractionMode = "edit"
    objective: str = ""
    steps: list[str] = field(default_factory=list)
    approval_mode: ApprovalMode = "manual"

    @property
    def is_plan_mode(self) -> bool:
        return self.mode == "plan"

    def to_dict(self) -> dict[str, Any]:
        return {
            "customType": PLAN_STATE_ENTRY,
            "mode": self.mode,
            "objective": self.objective,
            "steps": list(self.steps),
            "approvalMode": self.approval_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanState":
        mode = data.get("mode") if data.get("mode") in {"edit", "plan"} else "edit"
        approval = data.get("approvalMode") if data.get("approvalMode") in {"manual", "auto"} else "manual"
        raw_steps = data.get("steps")
        steps = [str(step).strip() for step in raw_steps if str(step).strip()] if isinstance(raw_steps, list) else []
        return cls(
            mode=mode,
            objective=str(data.get("objective") or ""),
            steps=steps,
            approval_mode=approval,
        )


def plan_prompt(user_prompt: str, state: PlanState) -> str:
    objective = state.objective or user_prompt
    return f"""[PLAN MODE ACTIVE]
You are in read-only planning mode. Explore and plan, but do not modify files or run destructive commands.
Use only the available read-only tools. If the task is underspecified, ask concise clarifying questions.
When ready, produce a numbered plan under a "Plan:" heading.

Objective:
{objective}

User request:
{user_prompt}
"""


def execution_prompt(state: PlanState) -> str:
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(state.steps, start=1))
    return f"""Execute the approved plan in edit mode.

Objective:
{state.objective}

Plan:
{steps or "(use the plan from the previous assistant message)"}
"""


def extract_plan_steps(text: str) -> list[str]:
    lines = text.splitlines()
    plan_start = 0
    for index, line in enumerate(lines):
        if line.strip().lower().rstrip(":") == "plan":
            plan_start = index + 1
            break

    steps: list[str] = []
    for line in lines[plan_start:]:
        stripped = line.strip()
        match = re.match(r"^(?:[-*]\s+|\d+[.)]\s+)(.+)$", stripped)
        if not match:
            if steps and not stripped:
                break
            continue
        step = _clean_step(match.group(1))
        if step:
            steps.append(step)
    return steps


def _clean_step(text: str) -> str:
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    return " ".join(cleaned.split()).strip()
