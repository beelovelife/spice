"""Isolated, optional Playwright page inspection (not visual acceptance testing)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
import sys
from contextlib import redirect_stdout
from typing import Any
from urllib.parse import urlsplit

_MAX_ERRORS = 50
_MAX_TEXT = 2000
_MAX_WAIT_MS = 10_000
_MAX_SELECTORS = 50
_ALLOWED_SCHEMES = frozenset({"http", "https", "data", "blob"})
_NON_CSS = re.compile(
    r"^[\w:-]+\s*=|:(?:has-text|text|text-is|text-matches|nth-match|light|"
    r"above|below|left-of|right-of|near)\s*\(|:visible\b",
    re.IGNORECASE,
)


def _clip(value: str, limit: int = _MAX_TEXT) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _report() -> dict[str, Any]:
    return {
        "passed": False,
        "page_errors": [],
        "console_errors": [],
        "request_failures": [],
        "http_errors": [],
        "checks": [],
        "error_count": 0,
        "screenshot_base64": None,
        "execution_error": None,
        "omitted_error_count": 0,
        "scope": (
            "Only runtime errors during navigation, the bounded observation window, "
            "selector checks and screenshot capture are observed. CSS checks require "
            "count > 0 and first.is_visible(). This is not visual acceptance testing. "
            "At most 50 distinct event errors and 2000 text characters per entry are retained; "
            "error_count includes all observed errors, failed checks and execution errors."
        ),
        "ignored_rules": [],
        "observation_ms": 0,
    }


class _Errors:
    """Bound retained event details without losing the number of error occurrences."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.collect_events = True
        self.entries: dict[tuple[str, bytes, bytes], dict[str, Any]] = {}

    def add(
        self, category: str, message: str, *, url: str = "", stack: str = "", status: int | None = None
    ) -> None:
        if not self.collect_events:
            return
        self.report["error_count"] += 1
        key = (
            category,
            hashlib.sha256(message.encode("utf-8", errors="replace")).digest(),
            hashlib.sha256(url.encode("utf-8", errors="replace")).digest(),
        )
        if key in self.entries:
            self.entries[key]["count"] += 1
            return
        if len(self.entries) >= _MAX_ERRORS:
            self.report["omitted_error_count"] += 1
            return
        text = {"message": message}
        if url:
            text["url"] = url
        if stack:
            text["stack"] = stack
        # A shared text budget also bounds an entry with both URL and stack.
        field_limit = _MAX_TEXT // len(text)
        entry: dict[str, Any] = {
            name: _clip(value, field_limit) for name, value in text.items()
        }
        entry["count"] = 1
        if status is not None:
            entry["status"] = status
        self.entries[key] = entry
        self.report[category].append(entry)

    def execution(self, message: str) -> None:
        self.report["error_count"] += 1
        previous = self.report["execution_error"]
        self.report["execution_error"] = _clip(
            f"{previous}\n{message}" if previous else message
        )


def _validate(
    url: str, timeout: float, wait_ms: int, selectors: list[str] | None
) -> tuple[float, int, list[str]]:
    if (
        not isinstance(url, str)
        or not url
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url)
    ):
        raise ValueError("url must be an absolute HTTP(S) URL without whitespace")
    # Backslashes can produce different authorities in URL parsers and Chromium.
    if "\\" in url:
        raise ValueError("url must not contain backslashes")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "url must use http or https; file/data/javascript URLs are not allowed"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url must not contain a username or password")
    _ = parsed.port  # Validate malformed/out-of-range ports before starting a browser.
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number of seconds")
    if isinstance(wait_ms, bool) or not isinstance(wait_ms, int) or wait_ms < 0:
        raise ValueError("wait_ms must be a non-negative integer")
    if selectors is not None and not isinstance(selectors, list):
        raise ValueError("selectors must be a list of CSS strings or null")
    selected = list(selectors or [])
    if len(selected) > _MAX_SELECTORS:
        raise ValueError(f"at most {_MAX_SELECTORS} CSS selectors are allowed")
    for selector in selected:
        if (
            not isinstance(selector, str)
            or not selector.strip()
            or len(selector) > _MAX_TEXT
            or ">>" in selector
            or _NON_CSS.search(selector.strip())
        ):
            raise ValueError(
                "selectors must be non-empty CSS strings (maximum 2000 characters); "
                "Playwright selector engines, chaining and custom pseudo-classes are not allowed"
            )
    return float(timeout), min(wait_ms, _MAX_WAIT_MS), selected


async def _guard_request(route: Any, errors: _Errors) -> None:
    request = route.request
    try:
        allowed = urlsplit(request.url).scheme.lower() in _ALLOWED_SCHEMES
    except ValueError:
        allowed = False
    if allowed:
        await route.continue_()
    else:
        errors.add(
            "request_failures", "Blocked non-HTTP(S)/data/blob scheme", url=request.url
        )
        await route.abort("blockedbyclient")


async def _launch(playwright: Any, deadline: float) -> Any:
    def remaining_ms() -> float:
        return max(1, (deadline - asyncio.get_running_loop().time()) * 1000)

    options = {"headless": True, "chromium_sandbox": True}
    try:
        return await playwright.chromium.launch(**options, timeout=remaining_ms())
    except Exception as exc:
        if "Executable doesn't exist" not in str(exc):
            raise
        # Never install software or fall back for sandbox/permission/startup failures.
        try:
            return await playwright.chromium.launch(
                **options, channel="chrome", timeout=remaining_ms()
            )
        except Exception as fallback:
            raise RuntimeError(
                "Bundled Chromium executable is missing; Chrome fallback failed: "
                f"{fallback}"
            ) from fallback


async def _close(resource: Any, name: str, errors: _Errors) -> None:
    if resource is not None:
        try:
            # Cleanup is outside the inspection budget, but is independently bounded.
            await asyncio.wait_for(resource.close(), timeout=5)
        except Exception as exc:
            errors.execution(f"Could not close {name}: {exc}")


async def inspect_page(
    url: str,
    *,
    timeout: float = 30,
    wait_ms: int = 1000,
    selectors: list[str] | None = None,
) -> dict:
    """Inspect an authorized URL within one budget; cleanup has a separate deadline."""
    report = _report()
    errors = _Errors(report)
    try:
        budget, observation_ms, selected = _validate(url, timeout, wait_ms, selectors)
    except (ValueError, TypeError, OverflowError) as exc:
        errors.execution(f"Invalid inspection arguments: {exc}")
        return report
    report["observation_ms"] = observation_ms

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        errors.execution(
            "Playwright is unavailable. Install the optional dependency: uv sync --extra browser"
        )
        return report

    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    # Reserve screenshot time so a timed-out page can still provide visual evidence.
    work_deadline = deadline - min(2.0, budget * 0.2)
    browser = context = page = None
    try:
        async with async_playwright() as playwright:
            try:
                async with asyncio.timeout_at(deadline):
                    browser = await _launch(playwright, work_deadline)
                    context = await browser.new_context(
                        viewport={"width": 1400, "height": 900},
                        accept_downloads=False,
                        service_workers="block",
                    )
                    # Route at context level to cover popups as well as the inspected page.
                    await context.route(
                        "**/*", lambda route: _guard_request(route, errors)
                    )
                    page = await context.new_page()
                    page.on(
                        "pageerror",
                        lambda error: errors.add(
                            "page_errors", error.message, stack=error.stack or ""
                        ),
                    )
                    page.on(
                        "console",
                        lambda message: (
                            errors.add("console_errors", message.text)
                            if message.type == "error"
                            else None
                        ),
                    )
                    page.on(
                        "requestfailed",
                        lambda request: errors.add(
                            "request_failures",
                            request.failure or "Request failed",
                            url=request.url,
                        ),
                    )
                    page.on(
                        "response",
                        lambda response: (
                            errors.add(
                                "http_errors",
                                f"HTTP {response.status}",
                                url=response.url,
                                status=response.status,
                            )
                            if response.status >= 400
                            else None
                        ),
                    )
                    try:
                        async with asyncio.timeout_at(work_deadline):
                            try:
                                await page.goto(
                                    url,
                                    wait_until="domcontentloaded",
                                    timeout=max(
                                        1, (work_deadline - loop.time()) * 1000
                                    ),
                                )
                            except Exception as exc:
                                errors.execution(f"Navigation failed: {exc}")
                            await asyncio.sleep(observation_ms / 1000)
                            for selector in selected:
                                check = {
                                    "selector": selector,
                                    "passed": False,
                                    "matches": 0,
                                    "visible": False,
                                    "error": None,
                                }
                                try:
                                    locator = page.locator(f"css={selector}")
                                    check["matches"] = await locator.count()
                                    check["visible"] = (
                                        await locator.first.is_visible()
                                        if check["matches"]
                                        else False
                                    )
                                    check["passed"] = (
                                        check["matches"] > 0 and check["visible"]
                                    )
                                    if not check["passed"]:
                                        check["error"] = (
                                            "CSS selector has no matches or its first match is not visible"
                                        )
                                except Exception as exc:
                                    check["error"] = _clip(str(exc))
                                report["checks"].append(check)
                                if not check["passed"]:
                                    report["error_count"] += 1
                    except TimeoutError:
                        errors.execution(
                            "Page inspection timed out before observation/selector checks completed"
                        )
                    finally:
                        for selector in selected[len(report["checks"]) :]:
                            report["checks"].append(
                                {
                                    "selector": selector,
                                    "passed": False,
                                    "matches": None,
                                    "visible": None,
                                    "error": "Not evaluated: inspection interrupted",
                                }
                            )
                            report["error_count"] += 1
                        if loop.time() < deadline:
                            try:
                                image = await page.screenshot(
                                    type="png",
                                    full_page=False,
                                    timeout=max(1, (deadline - loop.time()) * 1000),
                                )
                                report["screenshot_base64"] = base64.b64encode(
                                    image
                                ).decode("ascii")
                            except Exception as exc:
                                errors.execution(f"Screenshot failed: {exc}")
                        else:
                            errors.execution(
                                "No time remaining in the inspection budget for a screenshot"
                            )
            finally:
                # Teardown aborts healthy in-flight requests; do not count these as page errors.
                errors.collect_events = False
                await _close(context, "browser context", errors)
                await _close(browser, "browser", errors)
    except TimeoutError:
        errors.execution("Overall page inspection timeout exceeded")
    except Exception as exc:
        errors.execution(f"Browser inspection failed: {exc}")
    report["passed"] = (
        report["error_count"] == 0 and report["screenshot_base64"] is not None
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Accept exactly one JSON argument and emit exactly one JSON report."""
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValueError(
                'expected one JSON argument: {"url": "https://example.com"}'
            )
        options = json.loads(args[0])
        if not isinstance(options, dict) or "url" not in options:
            raise ValueError("the JSON argument must be an object containing url")
        if set(options) - {"url", "timeout", "wait_ms", "selectors"}:
            raise ValueError("allowed JSON fields: url, timeout, wait_ms, selectors")
        with redirect_stdout(sys.stderr):
            report = asyncio.run(inspect_page(**options))
    except (Exception, KeyboardInterrupt) as exc:
        report = _report()
        _Errors(report).execution(f"Browser runner failed: {exc or 'interrupted'}")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
