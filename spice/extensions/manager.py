"""Trusted local Python extensions."""

from __future__ import annotations

import inspect
import importlib.util
import sys
import types
from enum import Enum
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

from spice.tools.base import Tool, ToolContext, ToolResult, tool_error, tool_result

EXTENSIONS_DIR = Path.home() / ".spice" / "extensions"


@dataclass
class ExtensionCommand:
    name: str
    description: str
    handler: Callable[..., Any]
    source: Path


@dataclass
class Extension:
    path: Path
    tools: dict[str, Tool] = field(default_factory=dict)
    commands: dict[str, ExtensionCommand] = field(default_factory=dict)
    handlers: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)


@dataclass
class ExtensionEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""


class ExtensionAPI:
    def __init__(self, extension: Extension, cwd: Path) -> None:
        self.extension = extension
        self.cwd = cwd

    def register_tool(self, tool: Tool) -> None:
        self.extension.tools[tool.name] = tool

    def tool(
        self,
        name: str | None = None,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        requires_confirmation: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            schema = parameters or _schema_from_signature(func)

            async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
                try:
                    result = _call_with_args(func, args, context)
                    if inspect.isawaitable(result):
                        result = await result
                except Exception as exc:
                    return tool_error(str(exc))
                if isinstance(result, ToolResult):
                    return result
                return tool_result("" if result is None else str(result))

            self.register_tool(
                Tool(
                    name=tool_name,
                    description=description or inspect.getdoc(func) or tool_name,
                    parameters=schema,
                    execute=execute,
                    requires_confirmation=requires_confirmation,
                )
            )
            return func

        return decorator

    def register_command(self, name: str, handler: Callable[..., Any], description: str = "") -> None:
        normalized = name.removeprefix("/")
        if not normalized:
            raise ValueError("Extension command name cannot be empty.")
        self.extension.commands[normalized] = ExtensionCommand(
            name=normalized,
            description=description,
            handler=handler,
            source=self.extension.path,
        )

    def command(self, name: str, description: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register_command(name, func, description)
            return func

        return decorator

    def on(self, event: str, handler: Callable[..., Any] | None = None) -> Callable[..., Any] | None:
        def register(func: Callable[..., Any]) -> Callable[..., Any]:
            self.extension.handlers.setdefault(event, []).append(func)
            return func

        if handler is not None:
            return register(handler)
        return register


class ExtensionManager:
    def __init__(self, cwd: Path | None = None, extensions_dir: Path | None = None) -> None:
        self.cwd = (cwd or Path.cwd()).resolve()
        self.extensions_dir = (extensions_dir or EXTENSIONS_DIR).expanduser()
        self.extensions: list[Extension] = []
        self.errors: list[str] = []

    def discover(self) -> list[Path]:
        if not self.extensions_dir.exists():
            return []
        paths: list[Path] = []
        for child in sorted(self.extensions_dir.iterdir()):
            if child.name.startswith("_"):
                continue
            if child.is_file() and child.suffix == ".py":
                paths.append(child)
            elif child.is_dir():
                init = child / "__init__.py"
                if init.exists():
                    paths.append(init)
                    continue
                for name in ("extension.py", "main.py", "index.py"):
                    entry = child / name
                    if entry.exists():
                        paths.append(entry)
                        break
        return paths

    def load_default(self) -> None:
        self.load_paths(self.discover())

    def load_paths(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                self.extensions.append(self.load_extension(path))
            except Exception as exc:
                self.errors.append(f"{path}: {exc}")

    def load_extension(self, path: Path) -> Extension:
        path = path.expanduser().resolve()
        module_name = f"spice_extension_{path.stem}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load extension: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        factory = _find_extension_factory(module)
        if factory is None:
            raise ValueError("Extension must define extension(api), activate(api), or default(api).")

        extension = Extension(path=path)
        api = ExtensionAPI(extension, self.cwd)
        factory(api)
        return extension

    def tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for extension in self.extensions:
            tools.extend(extension.tools.values())
        return tools

    def commands(self) -> dict[str, ExtensionCommand]:
        commands: dict[str, ExtensionCommand] = {}
        for extension in self.extensions:
            commands.update(extension.commands)
        return commands

    async def emit(self, event_name: str, event: ExtensionEvent) -> ExtensionEvent:
        for extension in self.extensions:
            for handler in extension.handlers.get(event_name, []):
                result = _call_handler(handler, event, self)
                if inspect.isawaitable(result):
                    await result
        return event

    async def transform_input(self, text: str) -> str:
        event = ExtensionEvent(type="input", data={"text": text})
        await self.emit("input", event)
        return str(event.data.get("text", text))

    async def handle_command(self, name: str, args: str, context: Any = None) -> Any:
        command = self.commands().get(name)
        if not command:
            raise ValueError(f"Unknown extension command: /{name}")
        result = _call_command(command.handler, args, context, self)
        if inspect.isawaitable(result):
            return await result
        return result


def _find_extension_factory(module: Any) -> Callable[[ExtensionAPI], Any] | None:
    for name in ("extension", "activate", "default"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def _schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"context", "ctx"}:
            continue
        properties[name] = _json_schema(type_hints.get(name, parameter.annotation))
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _json_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        value_type = _common_json_type(values)
        if value_type:
            schema["type"] = value_type
        return schema

    if origin in {Union, types.UnionType}:
        non_null_args = [arg for arg in args if arg is not type(None)]
        if len(non_null_args) == 1:
            return _json_schema(non_null_args[0])
        return {"anyOf": [_json_schema(arg) for arg in non_null_args]}

    if origin in {list, tuple, set, frozenset}:
        return {"type": "array", "items": _json_schema(args[0]) if args else {}}

    if origin is dict:
        schema = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            schema["additionalProperties"] = _json_schema(args[1])
        return schema

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [item.value for item in annotation]
        schema = {"enum": values}
        value_type = _common_json_type(values)
        if value_type:
            schema["type"] = value_type
        return schema

    if inspect.isclass(annotation) and is_dataclass(annotation):
        return _object_schema_from_annotations(get_type_hints(annotation), _dataclass_required_fields(annotation))

    if inspect.isclass(annotation) and is_typeddict(annotation):
        required_keys = set(getattr(annotation, "__required_keys__", set()))
        return _object_schema_from_annotations(get_type_hints(annotation), required_keys)

    json_type = _json_type(annotation)
    return {"type": json_type} if json_type else {}


def _json_type(annotation: Any) -> str | None:
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is str:
        return "string"
    if annotation is list:
        return "array"
    if annotation is dict:
        return "object"
    if annotation is Any:
        return None
    return "string"


def _common_json_type(values: list[Any]) -> str | None:
    if not values:
        return None
    types_seen = {_json_type(type(value)) for value in values}
    return next(iter(types_seen)) if len(types_seen) == 1 else None


def _object_schema_from_annotations(annotations: dict[str, Any], required: set[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: _json_schema(annotation) for name, annotation in annotations.items()},
    }
    if required:
        schema["required"] = sorted(required)
    return schema


def _dataclass_required_fields(annotation: type[Any]) -> set[str]:
    required: set[str] = set()
    for item in fields(annotation):
        if item.default is MISSING and item.default_factory is MISSING:
            required.add(item.name)
    return required


def _call_with_args(func: Callable[..., Any], args: dict[str, Any], context: ToolContext) -> Any:
    signature = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name in {"context", "ctx"}:
            kwargs[name] = context
        elif name in args:
            kwargs[name] = args[name]
    return func(**kwargs)


def _call_handler(handler: Callable[..., Any], event: ExtensionEvent, manager: ExtensionManager) -> Any:
    params = list(inspect.signature(handler).parameters)
    if len(params) >= 2:
        return handler(event, manager)
    return handler(event)


def _call_command(handler: Callable[..., Any], args: str, context: Any, manager: ExtensionManager) -> Any:
    params = list(inspect.signature(handler).parameters)
    if len(params) >= 3:
        return handler(args, context, manager)
    if len(params) == 2:
        return handler(args, context)
    if len(params) == 1:
        return handler(args)
    return handler()
