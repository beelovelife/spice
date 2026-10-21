"""Prompt-toolkit legacy TUI adapters for shared interactive ports."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console

from spice.cli.tool_render import _preview_text
from spice.interactive.confirm import is_file_edit_tool
from spice.interactive.types import ChoiceRequest, ConfirmRequest

if TYPE_CHECKING:
    from spice.tui.app import SpiceTUI


class TuiConsole:
    def __init__(self, owner: SpiceTUI) -> None:
        self.owner = owner

    def print(self, *objects, **kwargs) -> None:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        )
        console.print(*objects, **kwargs)
        self.owner._append(output.getvalue())


class TuiInteractivePort:
    def __init__(self, owner: SpiceTUI) -> None:
        self.owner = owner

    async def choose(self, request: ChoiceRequest) -> str | None:
        return await self.owner._port_choose(request)

    async def confirm(self, request: ConfirmRequest) -> str:
        return await self.owner._port_confirm(request)


class TuiToolConfirmPort(TuiInteractivePort):
    """Render one tool preview while ConfirmPolicy serializes the modal."""

    def __init__(self, owner: SpiceTUI, tool_name: str, args: dict) -> None:
        super().__init__(owner)
        self.tool_name = tool_name
        self.args = args

    async def confirm(self, request: ConfirmRequest) -> str:
        self.owner._ensure_newline()
        if not is_file_edit_tool(self.tool_name):
            args_preview, _ = _preview_text(
                str(self.args),
                max_chars=700,
                max_lines=8,
            )
            self.owner._append(f"\nConfirm {self.tool_name}\n{args_preview}\n")
        return await self.owner._port_confirm(request)
