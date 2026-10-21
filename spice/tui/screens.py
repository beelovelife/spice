"""Modal interaction screens used by the Textual adapter."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from spice.interactive.types import ChoiceRequest, ConfirmRequest


class ChoiceScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, request: ChoiceRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        options = [
            Option(_option_text(item.label, item.detail), id=item.id)
            for item in self.request.items
        ]
        with Vertical(classes="callback-dialog"):
            yield Static(self.request.title, classes="callback-title", markup=False)
            yield OptionList(*options, id="callback-options", markup=False)
            yield Static("Enter choose · Esc cancel", classes="callback-help", markup=False)

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        if self.request.current_id:
            for index, option in enumerate(options.options):
                if option.id == self.request.current_id:
                    options.highlighted = index
                    break
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApprovalScreen(ModalScreen[str]):
    BINDINGS = [
        Binding("escape", "deny", "Deny", priority=True),
        Binding("1", "choose(0)", "Choice 1", show=False, priority=True),
        Binding("2", "choose(1)", "Choice 2", show=False, priority=True),
        Binding("3", "choose(2)", "Choice 3", show=False, priority=True),
        Binding("4", "choose(3)", "Choice 4", show=False, priority=True),
        Binding("5", "choose(4)", "Choice 5", show=False, priority=True),
    ]

    def __init__(self, request: ConfirmRequest, *, preview: str = "") -> None:
        super().__init__()
        self.request = request
        self.preview = preview
        self.ready_at = 0.0

    def compose(self) -> ComposeResult:
        options = [
            Option(f"{index}. {_option_text(item.label, item.detail)}", id=item.id)
            for index, item in enumerate(self.request.choices, start=1)
        ]
        with Vertical(classes="callback-dialog approval-dialog"):
            yield Static(self.request.question, classes="callback-title", markup=False)
            if self.preview:
                yield Static(self.preview, classes="approval-preview", markup=False)
            yield OptionList(*options, id="callback-options", markup=False)
            yield Static("Enter choose · Esc deny", classes="callback-help", markup=False)

    def on_mount(self) -> None:
        self.ready_at = time.monotonic() + 0.5
        options = self.query_one(OptionList)
        if self.request.current_id:
            for index, option in enumerate(options.options):
                if option.id == self.request.current_id:
                    options.highlighted = index
                    break
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if time.monotonic() >= self.ready_at:
            self.dismiss(event.option_id or "deny")

    def action_choose(self, index: int) -> None:
        if time.monotonic() < self.ready_at:
            return
        options = self.query_one(OptionList)
        if 0 <= index < options.option_count:
            self.dismiss(options.get_option_at_index(index).id or "deny")

    def action_deny(self) -> None:
        self.dismiss("deny")


def _option_text(label: str, detail: str) -> str:
    return f"{label}  —  {detail}" if detail else label

