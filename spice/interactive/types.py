"""UI-agnostic interactive types for slash commands and ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from spice.agent.agent_session import AgentSession
    from spice.interactive.confirm import ConfirmPolicy


@dataclass(frozen=True)
class TextView:
    text: str
    style: Literal["plain", "success", "warning", "error", "dim"] = "plain"


@dataclass(frozen=True)
class TableView:
    title: str | None
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PanelView:
    title: str
    body: str


CommandView = TextView | TableView | PanelView


@dataclass
class CommandResult:
    handled: bool = True
    exit_requested: bool = False
    clear_requested: bool = False
    session: AgentSession | None = None
    followup_prompt: str | None = None
    replay_session_history: bool = False
    views: list[CommandView] = field(default_factory=list)


@dataclass(frozen=True)
class ChoiceItem:
    id: str
    label: str
    detail: str = ""
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceRequest:
    title: str
    items: tuple[ChoiceItem, ...]
    current_id: str | None = None
    columns: tuple[str, ...] | None = None
    show_current_mark: bool = False


@dataclass(frozen=True)
class ConfirmChoice:
    id: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class ConfirmRequest:
    question: str
    choices: tuple[ConfirmChoice, ...]
    current_id: str | None = None


# Allow / deny plus session-scoped policy upgrades.
ConfirmDecision = str


@runtime_checkable
class InteractivePort(Protocol):
    async def choose(self, request: ChoiceRequest) -> str | None:
        """Return selected choice id, or None if cancelled."""
        ...

    async def confirm(self, request: ConfirmRequest) -> ConfirmDecision:
        """Return selected confirm choice id (e.g. allow / deny)."""
        ...


@dataclass
class CommandContext:
    session: AgentSession
    cwd: Path
    port: InteractivePort
    raw: str
    args: str
    confirm_policy: ConfirmPolicy
    # Optional extra state adapters may attach (not used by core handlers).
    extras: dict[str, Any] = field(default_factory=dict)


CommandHandler = Callable[[CommandContext], Awaitable[CommandResult]]
