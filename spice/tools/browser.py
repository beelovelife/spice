"""Browser runtime checks with explicit verification evidence."""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib.util
import json
import shlex
import sys
from typing import Any
from urllib.parse import urlsplit

from spice.llm.error_safety import sanitize_error_text
from spice.llm.images import MAX_IMAGE_BYTES, image_from_bytes
from spice.llm.messages import ImageContent
from spice.sandbox.base import ExecResult
from spice.sandbox.factory import create_environment, create_workspace_policy
from spice.tools.base import Tool, ToolContext, ToolResult, fatal_tool_error, tool_error
from spice.tools.verification import run_verified_operation

ERROR_FIELDS = ("page_errors", "console_errors", "request_failures", "http_errors")


def browser_dependency_available() -> bool:
    return importlib.util.find_spec("playwright") is not None


async def browser_check(args: dict[str, Any], context: ToolContext) -> ToolResult:
    url = str(args.get("url") or "")
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("url must be an HTTP(S) URL without credentials.")
        timeout = float(args.get("timeout", 30))
        wait_ms = int(args.get("wait_ms", 1000))
        selectors = args.get("selectors", [])
        if not 1 <= timeout <= 120 or not 0 <= wait_ms <= 10000:
            raise ValueError("timeout must be 1–120 seconds and wait_ms must be 0–10000.")
        if not isinstance(selectors, list) or len(selectors) > 20 or any(not isinstance(item, str) or not item.strip() for item in selectors):
            raise ValueError("selectors must be at most 20 non-empty CSS selectors.")
    except (ValueError, TypeError) as exc:
        return tool_error(str(exc), code="invalid_browser_arguments")
    workspace = context.workspace or create_workspace_policy(None, cwd=context.cwd)
    environment = context.environment or create_environment(None, cwd=context.cwd)
    if environment.name != "local":
        return tool_error(
            "browser_check currently requires the local execution environment; it will not bypass the configured sandbox.",
            code="browser_environment_unsupported",
        )
    try:
        cwd = workspace.resolve_exec_cwd(str(context.cwd))
    except PermissionError as exc:
        return fatal_tool_error(str(exc), code="workspace_policy_denied")

    parameters = {"url": url, "timeout": timeout, "wait_ms": wait_ms, "selectors": selectors}
    command = shlex.join([sys.executable, "-I", "-m", "spice.tools.browser_runner", json.dumps(parameters)])
    identity = "browser_check " + json.dumps(parameters, sort_keys=True)
    report: dict[str, Any] = {}
    images: list[ImageContent] = []

    async def execute() -> ExecResult:
        nonlocal report
        if not browser_dependency_available():
            return _failed_execution("Browser dependency is unavailable. Install the browser extra with `uv sync --extra browser`.")
        result = await environment.run(command, cwd=cwd, timeout=timeout + 20)
        if result.timed_out:
            return result
        try:
            if len(result.stdout) > MAX_IMAGE_BYTES * 2:
                raise ValueError("Browser report exceeds the output limit.")
            data = json.loads(result.stdout)
            if not isinstance(data, dict) or not isinstance(data.get("passed"), bool):
                raise ValueError("Browser returned an invalid report.")
            report = data
            snapshot = report.pop("screenshot_base64", None)
            if any(not isinstance(data.get(key), list) for key in (*ERROR_FIELDS, "checks")):
                raise ValueError("Browser report is missing check results.")
            if type(data.get("error_count")) is not int or data["error_count"] < 0:
                raise ValueError("Browser report is missing its error count.")
            checks = data["checks"]
            if not report.get("execution_error") and (len(checks) != len(selectors) or any(
                not isinstance(check, dict) or not isinstance(check.get("passed"), bool)
                or check.get("selector") != selector for check, selector in zip(checks, selectors)
            )):
                raise ValueError("Browser did not complete the requested selector checks.")
            if snapshot is not None:
                if not isinstance(snapshot, str) or len(snapshot) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
                    raise ValueError("Browser screenshot exceeds the image limit.")
                image = await asyncio.to_thread(
                    image_from_bytes, base64.b64decode(snapshot, validate=True), source=f"browser_check:{url}",
                )
                if context.supports_vision:
                    images.append(image)
                report["screenshot"] = {
                    "captured": True, "attached": context.supports_vision,
                    "width": image.width, "height": image.height,
                }
                if not context.supports_vision:
                    report["image_notice"] = "Screenshot captured but not sent: current model does not support image input; visual inspection was not performed."
            else:
                report["screenshot"] = {"captured": False, "attached": False}
            passed = (
                result.exit_code == 0 and report["passed"] and report["error_count"] == 0
                and not any(report[key] for key in ERROR_FIELDS)
                and not report.get("execution_error") and all(check["passed"] for check in checks)
                and snapshot is not None
            )
            report["passed"] = bool(passed)
            report["scope"] = "Runtime errors and requested visible selectors during the observation window; not visual or comprehensive functional acceptance."
            text = json.dumps(report, ensure_ascii=False)
            return ExecResult(output=text, stdout=text, stderr="", exit_code=0 if passed else 1,
                              details={"environment": environment.name})
        except (ValueError, TypeError, binascii.Error) as exc:
            reason = report.get("execution_error") or str(exc)
            report = {"passed": False, "execution_error": reason}
            return _failed_execution(reason)

    try:
        result = await run_verified_operation(identity, root=workspace.root, cwd=cwd, environment=environment, operation=execute)
    except (PermissionError, RuntimeError, OSError) as exc:
        return tool_error(sanitize_error_text(str(exc)), code="browser_execution_failed")
    if result.timed_out:
        report = {"passed": False, "execution_error": "Browser check timed out."}
    if not report:
        report = {"passed": False, "execution_error": result.output}
    return ToolResult(
        content=json.dumps(report, ensure_ascii=False), is_error=result.exit_code != 0 or result.timed_out,
        details={**result.details, "exit_code": result.exit_code, "browser_check": report},
        error_code="browser_check_failed" if result.exit_code != 0 or result.timed_out else None,
        images=images,
    )


def _failed_execution(message: str) -> ExecResult:
    text = sanitize_error_text(message)
    return ExecResult(output=text, stdout="", stderr=text, exit_code=1)


def create_browser_tools() -> list[Tool]:
    return [Tool(
        name="browser_check",
        description=(
            "Open an HTTP(S) page in an isolated browser, check page errors, console errors, failed requests, "
            "HTTP errors and optional visible CSS selectors, and return a screenshot to vision-capable models. "
            "Failures remain failures even when a screenshot is captured. Requires the optional browser extra "
            "and Chromium or Chrome. Does not prove visual correctness or test unrequested interactions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "timeout": {"type": "number", "minimum": 1, "maximum": 120},
                "wait_ms": {"type": "integer", "minimum": 0, "maximum": 10000},
                "selectors": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 20},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        execute=browser_check,
        requires_confirmation=True,
        timeout_seconds=150,
    )]
