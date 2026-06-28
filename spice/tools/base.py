"""Tool definitions and helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spice.agent.subagent import SubagentManager
    from spice.sandbox.base import ExecutionEnvironment
    from spice.sandbox.policy import WorkspacePolicy
    from spice.tools.file_state import FileStateStore


ConfirmFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    cwd: Path
    workspace: WorkspacePolicy | None = None
    environment: ExecutionEnvironment | None = None
    confirm: ConfirmFn | None = None
    emit_update: Callable[[str], Awaitable[None]] | None = None
    file_states: FileStateStore | None = None
    subagent_manager: SubagentManager | None = None


ToolExecuteFn = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecuteFn
    requires_confirmation: bool = False


def validate_tool_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Validate the small JSON Schema subset used by built-in tools."""
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    return _validate_schema(schema, args, path="")


def normalize_tool_arguments(schema: dict[str, Any], args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return safely normalized arguments plus validation errors."""
    if not isinstance(args, dict):
        return {}, ["arguments must be an object"]
    normalized, errors = _normalize_schema(schema, dict(args), path="")
    if errors:
        return dict(args), errors
    if isinstance(normalized, dict):
        return normalized, _validate_schema(schema, normalized, path="")
    return dict(args), ["arguments must be an object"]


def _validate_schema(schema: dict[str, Any], value: Any, *, path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected and not _matches_json_type(value, expected):
        errors.append(f"{path or 'arguments'} must be {expected}")
        return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path or 'value'} must be one of: {', '.join(map(str, enum))}")
        return errors

    if isinstance(value, dict):
        errors.extend(_validate_object(schema, value, path=path))
    elif isinstance(value, list):
        errors.extend(_validate_array(schema, value, path=path))
    elif isinstance(value, str):
        errors.extend(_validate_string(schema, value, path=path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        errors.extend(_validate_number(schema, value, path=path))
    return errors


def _validate_object(schema: dict[str, Any], value: dict[str, Any], *, path: str) -> list[str]:
    errors: list[str] = []

    required = schema.get("required") or []
    if isinstance(required, list):
        for name in required:
            if name not in value:
                errors.append(f"missing required argument: {_join_path(path, str(name))}")

    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return errors

    additional = schema.get("additionalProperties", True)
    for name, item in value.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            if additional is False:
                errors.append(f"unexpected argument: {_join_path(path, str(name))}")
            continue
        errors.extend(_validate_schema(property_schema, item, path=_join_path(path, str(name))))
    return errors


def _validate_array(schema: dict[str, Any], value: list[Any], *, path: str) -> list[str]:
    errors: list[str] = []
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(f"{path or 'array'} must contain at least {min_items} items")
    if isinstance(max_items, int) and len(value) > max_items:
        errors.append(f"{path or 'array'} must contain at most {max_items} items")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            errors.extend(_validate_schema(item_schema, item, path=f"{path}[{index}]" if path else f"[{index}]"))
    return errors


def _validate_string(schema: dict[str, Any], value: str, *, path: str) -> list[str]:
    errors: list[str] = []
    min_length = schema.get("minLength")
    max_length = schema.get("maxLength")
    if isinstance(min_length, int) and len(value) < min_length:
        errors.append(f"{path or 'string'} must be at least {min_length} characters")
    if isinstance(max_length, int) and len(value) > max_length:
        errors.append(f"{path or 'string'} must be at most {max_length} characters")
    return errors


def _validate_number(schema: dict[str, Any], value: int | float, *, path: str) -> list[str]:
    errors: list[str] = []
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        errors.append(f"{path or 'number'} must be >= {minimum}")
    if isinstance(maximum, (int, float)) and value > maximum:
        errors.append(f"{path or 'number'} must be <= {maximum}")
    return errors


def _normalize_schema(schema: dict[str, Any], value: Any, *, path: str) -> tuple[Any, list[str]]:
    expected = schema.get("type")
    value = _coerce_value(value, expected)
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            normalized = dict(value)
            errors: list[str] = []
            for name, property_schema in properties.items():
                if name not in normalized or not isinstance(property_schema, dict):
                    continue
                normalized[name], property_errors = _normalize_schema(
                    property_schema,
                    normalized[name],
                    path=_join_path(path, str(name)),
                )
                errors.extend(property_errors)
            return normalized, errors
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            normalized_items: list[Any] = []
            errors = []
            for index, item in enumerate(value):
                normalized_item, item_errors = _normalize_schema(
                    item_schema,
                    item,
                    path=f"{path}[{index}]" if path else f"[{index}]",
                )
                normalized_items.append(normalized_item)
                errors.extend(item_errors)
            return normalized_items, errors
    return value, []


def _coerce_value(value: Any, expected: str | list[str] | None) -> Any:
    expected_types = expected if isinstance(expected, list) else [expected]
    if "integer" in expected_types and isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    if "number" in expected_types and isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return value
    if "boolean" in expected_types and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def _join_path(prefix: str, name: str) -> str:
    if not prefix:
        return name
    return f"{prefix}.{name}"


def _matches_json_type(value: Any, expected: str | list[str]) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    for item in expected_types:
        if item == "string" and isinstance(value, str):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
        if item == "null" and value is None:
            return True
    return False


def tool_result(content: str, details: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(content=content, details=details or {})


def tool_error(content: str, details: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(content=content, is_error=True, details=details or {})


def truncate_head(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated: kept first {limit} characters]"


def truncate_tail(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return f"[truncated: kept last {limit} characters]\n\n" + text[-limit:]


def workspace_path(cwd: Path, raw_path: str) -> Path:
    from spice.sandbox.policy import WorkspacePolicy

    return WorkspacePolicy.from_settings({}, cwd=cwd).resolve_read(raw_path)
