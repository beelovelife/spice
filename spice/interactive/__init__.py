"""UI-agnostic interactive layer shared by CLI, TUI, and future frontends."""

from spice.interactive.commands import (
    SlashCommand,
    SlashCommandRegistry,
    is_execute_request,
    sustained_goal_prompt,
)
from spice.interactive.confirm import ConfirmPolicy
from spice.interactive.sessions import (
    SESSION_PICKER_LIMIT,
    entry_active_label,
    entry_preview,
    replace_session,
)
from spice.interactive.types import (
    ChoiceItem,
    ChoiceRequest,
    CommandContext,
    CommandResult,
    ConfirmChoice,
    ConfirmRequest,
    InteractivePort,
    PanelView,
    TableView,
    TextView,
)

__all__ = [
    "SESSION_PICKER_LIMIT",
    "ChoiceItem",
    "ChoiceRequest",
    "CommandContext",
    "CommandResult",
    "ConfirmChoice",
    "ConfirmPolicy",
    "ConfirmRequest",
    "InteractivePort",
    "PanelView",
    "SlashCommand",
    "SlashCommandRegistry",
    "TableView",
    "TextView",
    "entry_active_label",
    "entry_preview",
    "is_execute_request",
    "replace_session",
    "sustained_goal_prompt",
]
