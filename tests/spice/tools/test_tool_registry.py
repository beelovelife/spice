from __future__ import annotations

import pytest

from spice.tools.base import Tool, ToolContext, tool_result
from spice.tools.tool_registry import ToolCallError, ToolCallPlan, ToolRegistry, create_coding_tools, create_read_only_tools


async def _noop(args: dict, context: ToolContext):
    return tool_result("ok")


def test_tool_registry_rejects_duplicate_tool_names() -> None:
    tool = Tool("demo", "demo", {"type": "object"}, _noop)

    with pytest.raises(ValueError, match="Duplicate tool names"):
        ToolRegistry([tool, tool])


def test_prepare_call_normalizes_safe_scalar_types() -> None:
    registry = ToolRegistry(
        [
            Tool(
                "demo",
                "demo",
                {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                    },
                    "required": ["count", "enabled"],
                },
                _noop,
            )
        ]
    )

    plan = registry.prepare_call("demo", {"count": "5", "enabled": "true"})

    assert isinstance(plan, ToolCallPlan)
    assert plan.arguments == {"count": 5, "enabled": True}


def test_prepare_call_reports_nested_schema_errors() -> None:
    registry = ToolRegistry(
        [
            Tool(
                "demo",
                "demo",
                {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
                    },
                    "required": ["items"],
                },
                _noop,
            )
        ]
    )

    result = registry.prepare_call("demo", {"items": [{}]})

    assert isinstance(result, ToolCallError)
    assert "items[0].name" in result.message


def test_read_files_is_available_in_coding_and_read_only_toolsets() -> None:
    assert "read_files" in [tool.name for tool in create_coding_tools()]
    assert "read_files" in [tool.name for tool in create_read_only_tools()]


def test_memory_tool_is_available_only_when_enabled_in_coding_toolset() -> None:
    assert "memory" not in [tool.name for tool in create_coding_tools()]
    assert "memory" in [tool.name for tool in create_coding_tools(memory_enabled=True)]
    assert "memory" not in [tool.name for tool in create_read_only_tools()]


def test_subagent_tool_is_available_only_when_enabled_in_coding_toolset() -> None:
    assert "spawn_subagents" not in [tool.name for tool in create_coding_tools()]
    assert "spawn_subagents" in [tool.name for tool in create_coding_tools(subagents_enabled=True)]
    assert "spawn_subagents" not in [tool.name for tool in create_read_only_tools()]


def test_long_task_tools_are_not_registered_in_static_toolsets() -> None:
    coding_names = [tool.name for tool in create_coding_tools()]
    read_only_names = [tool.name for tool in create_read_only_tools()]

    assert "long_task" not in coding_names
    assert "complete_long_task" not in coding_names
    assert "long_task" not in read_only_names
    assert "complete_long_task" not in read_only_names
