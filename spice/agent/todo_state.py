"""Session-local todo state for execution progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TODO_STATE_ENTRY = "todo_state"
TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
VALID_TODO_STATUSES: set[str] = {"pending", "in_progress", "completed", "cancelled"}
ACTIVE_TODO_STATUSES: set[str] = {"pending", "in_progress"}
MAX_TODO_ITEMS = 256
MAX_TODO_CONTENT_CHARS = 4000
TRUNCATION_MARKER = "... [truncated]"


@dataclass
class TodoItem:
    id: str
    content: str
    status: TodoStatus = "pending"

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "content": self.content, "status": self.status}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoItem":
        item_id = str(data.get("id") or "").strip() or "?"
        content = str(data.get("content") or "").strip() or "(no description)"
        status = str(data.get("status") or "pending").strip().lower()
        if status not in VALID_TODO_STATUSES:
            status = "pending"
        return cls(id=item_id, content=_cap_content(content), status=status)  # type: ignore[arg-type]


@dataclass
class TodoState:
    items: list[TodoItem] = field(default_factory=list)

    def read(self) -> list[dict[str, str]]:
        return [item.to_dict() for item in self.items]

    def replace(self, todos: list[dict[str, Any]]) -> None:
        items = [_normalize_item(item) for item in _dedupe_by_id(todos)]
        _validate_items(items)
        self.items = items[:MAX_TODO_ITEMS]

    def merge(self, todos: list[dict[str, Any]]) -> None:
        by_id = {item.id: item for item in self.items}
        order = [item.id for item in self.items]
        for raw in _dedupe_by_id(todos):
            item = _normalize_item(raw)
            if item.id in by_id:
                current = by_id[item.id]
                current.content = item.content
                current.status = item.status
            else:
                by_id[item.id] = item
                order.append(item.id)
        items = [by_id[item_id] for item_id in order if item_id in by_id]
        _validate_items(items)
        self.items = items[:MAX_TODO_ITEMS]

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.items),
            "pending": sum(1 for item in self.items if item.status == "pending"),
            "in_progress": sum(1 for item in self.items if item.status == "in_progress"),
            "completed": sum(1 for item in self.items if item.status == "completed"),
            "cancelled": sum(1 for item in self.items if item.status == "cancelled"),
        }

    def active_items(self) -> list[TodoItem]:
        return [item for item in self.items if item.status in ACTIVE_TODO_STATUSES]

    def has_active_items(self) -> bool:
        return bool(self.active_items())

    def runtime_context(self) -> str | None:
        if not self.items:
            return None
        markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "cancelled": "[~]",
        }
        lines = ["Current todo list:"]
        for item in self.items:
            lines.append(f"- {markers.get(item.status, '[ ]')} {item.id}. {item.content} ({item.status})")
        lines.append("Completed and cancelled items are history only; continue from in_progress and pending items unless the user explicitly asks otherwise.")
        return "\n".join(lines)

    def status_line(self) -> str:
        if not self.items:
            return ""
        active = self.active_items()
        if not active:
            return ""
        completed = sum(1 for item in self.items if item.status == "completed")
        current = next((item for item in active if item.status == "in_progress"), active[0])
        return f"{completed}/{len(self.items)} {current.content}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "customType": TODO_STATE_ENTRY,
            "items": self.read(),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoState":
        raw_items = data.get("items")
        state = cls()
        if isinstance(raw_items, list):
            state.items = [_normalize_item(item) for item in raw_items if isinstance(item, dict)][:MAX_TODO_ITEMS]
            _validate_items(state.items)
        return state


def todos_from_plan_steps(steps: list[str]) -> TodoState:
    state = TodoState()
    state.replace(
        [
            {"id": str(index), "content": step, "status": "pending"}
            for index, step in enumerate(steps, start=1)
        ]
    )
    return state


def _normalize_item(data: dict[str, Any]) -> TodoItem:
    return TodoItem.from_dict(data)


def _validate_items(items: list[TodoItem]) -> None:
    in_progress = [item.id for item in items if item.status == "in_progress"]
    if len(in_progress) > 1:
        raise ValueError("Only one todo item may be in_progress at a time.")


def _dedupe_by_id(todos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_index: dict[str, int] = {}
    for index, item in enumerate(todos):
        item_id = str(item.get("id") or "").strip() or "?"
        last_index[item_id] = index
    return [todos[index] for index in sorted(last_index.values())]


def _cap_content(content: str) -> str:
    if len(content) <= MAX_TODO_CONTENT_CHARS:
        return content
    keep = MAX_TODO_CONTENT_CHARS - len(TRUNCATION_MARKER)
    return content[:keep] + TRUNCATION_MARKER
