from __future__ import annotations

import asyncio

from spice.extensions.manager import ExtensionEvent, ExtensionManager
from spice.tools.base import ToolContext


def test_extension_loads_python_file_with_command_tool_and_hook(tmp_path):
    ext_file = tmp_path / "demo.py"
    ext_file.write_text(
        """
def extension(api):
    @api.command("hello", "Say hello")
    def hello(args):
        return "hello " + args

    @api.tool(description="Double a number")
    def double(x: int):
        return x * 2

    @api.on("input")
    def on_input(event):
        event.data["text"] = event.data["text"].upper()
""",
        encoding="utf-8",
    )

    manager = ExtensionManager(extensions_dir=tmp_path)
    manager.load_default()

    assert not manager.errors
    assert "hello" in manager.commands()
    assert [tool.name for tool in manager.tools()] == ["double"]


def test_extension_tool_schema_infers_common_complex_annotations(tmp_path):
    ext_file = tmp_path / "demo.py"
    ext_file.write_text(
        """from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

@dataclass
class Bounds:
    start: int
    end: int = 100

class Filters(TypedDict):
    active: bool
    owner: str

def extension(api):
    @api.tool(description="Search items")
    def search(
        mode: Literal["fast", "safe"],
        ids: list[int],
        labels: dict[str, str],
        filters: Filters,
        bounds: Bounds | None = None,
        limit: int | None = None,
    ):
        return "ok"
""",
        encoding="utf-8",
    )

    manager = ExtensionManager(extensions_dir=tmp_path)
    manager.load_default()

    schema = manager.tools()[0].parameters

    assert schema["required"] == ["mode", "ids", "labels", "filters"]
    assert schema["properties"]["mode"] == {"enum": ["fast", "safe"], "type": "string"}
    assert schema["properties"]["ids"] == {"type": "array", "items": {"type": "integer"}}
    assert schema["properties"]["labels"] == {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    assert schema["properties"]["filters"] == {
        "type": "object",
        "properties": {"active": {"type": "boolean"}, "owner": {"type": "string"}},
        "required": ["active", "owner"],
    }
    assert schema["properties"]["bounds"] == {
        "type": "object",
        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
        "required": ["start"],
    }
    assert schema["properties"]["limit"] == {"type": "integer"}


def test_extension_command_and_tool_execute(tmp_path):
    ext_file = tmp_path / "demo.py"
    ext_file.write_text(
        """
def extension(api):
    @api.command("hello", "Say hello")
    def hello(args):
        return "hello " + args

    @api.tool(description="Double a number")
    def double(x: int):
        return x * 2
""",
        encoding="utf-8",
    )
    manager = ExtensionManager(extensions_dir=tmp_path)
    manager.load_default()

    async def run():
        command_result = await manager.handle_command("hello", "spice")
        tool_result = await manager.tools()[0].execute({"x": 4}, ToolContext(cwd=tmp_path))
        return command_result, tool_result

    command_result, tool_result = asyncio.run(run())

    assert command_result == "hello spice"
    assert tool_result.content == "8"


def test_extension_hooks_can_transform_input_and_block_tool(tmp_path):
    ext_file = tmp_path / "demo.py"
    ext_file.write_text(
        """
def extension(api):
    @api.on("input")
    def on_input(event):
        event.data["text"] = event.data["text"] + "!"

    @api.on("tool_call_start")
    def on_tool_call(event):
        event.blocked = True
        event.block_reason = "blocked"
""",
        encoding="utf-8",
    )
    manager = ExtensionManager(extensions_dir=tmp_path)
    manager.load_default()

    async def run():
        text = await manager.transform_input("hello")
        event = await manager.emit("tool_call_start", ExtensionEvent(type="tool_call_start"))
        return text, event

    text, event = asyncio.run(run())

    assert text == "hello!"
    assert event.blocked is True
    assert event.block_reason == "blocked"
