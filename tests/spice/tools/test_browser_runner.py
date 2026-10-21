from __future__ import annotations

import asyncio
import base64
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest

from spice.tools import browser_runner as runner


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/disconnect":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if path == "/redirect-file":
            self.send_response(302)
            self.send_header("Location", "file:///spice-browser-runner-denied.html")
            self.end_headers()
            return
        pages = {
            "/": '<h1 id="ready">Ready</h1><span class="hidden" hidden>Hidden</span>'
            '<span class="hidden">Second match is visible</span>',
            "/errors": """
                <h1>Errors</h1><script>
                function explode() { throw new Error('repeat boom'); }
                for (let i = 1; i <= 3; i++) setTimeout(explode, i * 60);
                console.error('repeat console'); console.error('repeat console');
                setTimeout(() => requestAnimationFrame(() => {
                    throw new Error('animation boom');
                }), 200);
                </script>
            """,
            "/http-errors": """
                <h1>HTTP errors</h1><script>
                for (let i = 0; i < 3; i++) fetch('/not-found');
                </script>
            """,
            "/request-failure": "<h1>Failure</h1><script>fetch('/disconnect').catch(() => {});</script>",
            "/memory": """
                <h1 id="ready">Memory resources</h1>
                <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/%3E">
                <script>
                const image = document.createElement('img');
                image.src = URL.createObjectURL(new Blob([
                    '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
                ], {type: 'image/svg+xml'}));
                document.body.appendChild(image);
                </script>
            """,
            "/file-resource": '<h1>Blocked</h1><iframe src="file:///spice-browser-runner-denied.html"></iframe>',
        }
        status = 200 if path in pages else 404
        body = (
            '<!doctype html><html><head><link rel="icon" href="data:,"></head><body>'
            + pages.get(path, "Not found")
            + "</body></html>"
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser_available(local_server):
    report = asyncio.run(runner.inspect_page(local_server, timeout=20, wait_ms=0))
    error = (report["execution_error"] or "").lower()
    if "uv sync --extra browser" in error:
        pytest.skip(report["execution_error"])
    if "chrome fallback failed" in error and (
        "is not found" in error or "executable doesn't exist" in error
    ):
        pytest.skip(
            f"Neither bundled Chromium nor local Chrome is installed: {report['execution_error']}"
        )
    assert report["passed"], {
        key: value for key, value in report.items() if key != "screenshot_base64"
    }


def _inspect(local_server, path="/", **kwargs):
    return asyncio.run(runner.inspect_page(local_server + path, **kwargs))


def _assert_screenshot(report):
    image = base64.b64decode(report["screenshot_base64"], validate=True)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", image[16:24]) == (1400, 900)


def test_normal_page(local_server, browser_available):
    report = _inspect(local_server, selectors=["h1#ready"], wait_ms=50)
    assert report["passed"] is True
    assert report["error_count"] == 0
    assert report["execution_error"] is None
    assert report["checks"] == [
        {
            "selector": "h1#ready",
            "passed": True,
            "matches": 1,
            "visible": True,
            "error": None,
        }
    ]
    assert "not visual acceptance" in report["scope"]
    assert report["ignored_rules"] == []
    _assert_screenshot(report)


def test_repeated_errors_and_animation_are_observed(local_server, browser_available):
    report = _inspect(local_server, "/errors", wait_ms=500)
    assert report["passed"] is False
    errors = {entry["message"]: entry for entry in report["page_errors"]}
    assert errors["repeat boom"]["count"] == 3
    assert "/errors" in errors["repeat boom"]["stack"]
    assert errors["animation boom"]["count"] == 1
    assert report["console_errors"] == [{"message": "repeat console", "count": 2}]
    assert report["error_count"] == 6
    _assert_screenshot(report)


def test_http_errors_aggregate_by_status_and_url(local_server, browser_available):
    report = _inspect(local_server, "/http-errors", wait_ms=300)
    assert report["passed"] is False
    assert report["http_errors"] == [
        {
            "message": "HTTP 404",
            "url": local_server + "/not-found",
            "count": 3,
            "status": 404,
        }
    ]
    assert report["error_count"] >= 3
    _assert_screenshot(report)


def test_failed_request(local_server, browser_available):
    report = _inspect(local_server, "/request-failure", wait_ms=300)
    assert report["passed"] is False
    assert any(
        item["url"].endswith("/disconnect") for item in report["request_failures"]
    )
    assert report["error_count"] >= 1
    _assert_screenshot(report)


def test_missing_and_hidden_first_selector(local_server, browser_available):
    report = _inspect(
        local_server, wait_ms=0, selectors=["#absent", ".hidden", "#ready"]
    )
    assert report["passed"] is False
    assert report["error_count"] == 2
    assert [check["passed"] for check in report["checks"]] == [False, False, True]
    assert [check["matches"] for check in report["checks"]] == [0, 2, 1]
    _assert_screenshot(report)


def test_invalid_css_is_a_failed_check_not_javascript(local_server, browser_available):
    report = _inspect(local_server, wait_ms=0, selectors=["[", "#ready"])
    assert report["passed"] is False
    assert report["error_count"] == 1
    assert report["checks"][0]["error"]
    assert report["checks"][1]["passed"] is True
    _assert_screenshot(report)


def test_data_and_blob_resources_work(local_server, browser_available):
    report = _inspect(local_server, "/memory", wait_ms=200, selectors=["#ready", "img"])
    assert report["passed"] is True
    assert report["error_count"] == 0


@pytest.mark.parametrize("path", ["/file-resource", "/redirect-file"])
def test_cross_scheme_file_access_is_blocked(local_server, browser_available, path):
    report = _inspect(local_server, path, wait_ms=50)
    assert report["passed"] is False
    assert report["error_count"] > 0
    assert (
        report["execution_error"]
        or report["console_errors"]
        or report["request_failures"]
    )
    _assert_screenshot(report)


def test_observation_uses_overall_budget_and_reserves_screenshot(
    local_server, browser_available
):
    start = time.monotonic()
    report = _inspect(local_server, timeout=4, wait_ms=10_000, selectors=["#ready"])
    elapsed = time.monotonic() - start
    assert report["passed"] is False
    assert "timed out" in report["execution_error"].lower()
    assert report["checks"][0]["passed"] is False
    assert "Not evaluated" in report["checks"][0]["error"]
    assert elapsed < 7  # Includes independently bounded context/browser cleanup.
    _assert_screenshot(report)


@pytest.mark.parametrize("selector,code", [("#ready", 0), ("#absent", 1)])
def test_module_cli_single_json_report(local_server, browser_available, selector, code):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spice.tools.browser_runner",
            json.dumps(
                {
                    "url": local_server,
                    "wait_ms": 0,
                    "selectors": [selector],
                }
            ),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == code, result.stderr
    assert len(result.stdout.splitlines()) == 1
    report = json.loads(result.stdout)
    assert report["passed"] is (code == 0)
    _assert_screenshot(report)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/test.html",
        "data:text/html,test",
        "javascript:alert(1)",
        "ftp://localhost/test",
        "https://user:password@localhost/",
        "http://@localhost/",
        "http:///missing-host",
        "http://localhost:99999",
        " http://localhost/",
        "http://localhost\\@example.com/",
        "http://localhost/\nnext",
    ],
)
def test_rejects_invalid_urls_before_launch(url, monkeypatch):
    launch = AsyncMock()
    monkeypatch.setattr(runner, "_launch", launch)
    report = asyncio.run(runner.inspect_page(url))
    assert report["passed"] is False
    assert report["error_count"] == 1
    assert "Invalid inspection arguments" in report["execution_error"]
    launch.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": True},
        {"timeout": "30"},
        {"wait_ms": -1},
        {"wait_ms": True},
        {"wait_ms": 1.5},
        {"selectors": "#ready"},
        {"selectors": [None]},
        {"selectors": [""]},
        {"selectors": ["#ready"] * 51},
        {"selectors": ["x" * 2001]},
        {"selectors": ["xpath=//*"]},
        {"selectors": ["body >> text=Ready"]},
        {"selectors": [":text('Ready')"]},
        {"selectors": ["text=Ready"]},
    ],
)
def test_rejects_invalid_options(options):
    report = asyncio.run(runner.inspect_page("http://localhost/", **options))
    assert report["passed"] is False
    assert "Invalid inspection arguments" in report["execution_error"]


def test_wait_is_bounded_and_css_supported():
    assert (
        runner._validate(
            "https://localhost/",
            30,
            100_000,
            ["div:has(> span)", 'a[href="https://localhost/"]'],
        )[1]
        == 10_000
    )


def test_missing_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright.async_api", None)
    report = asyncio.run(runner.inspect_page("http://localhost/"))
    assert report["passed"] is False
    assert report["screenshot_base64"] is None
    assert "uv sync --extra browser" in report["execution_error"]
    assert report["error_count"] == 1


def test_errors_are_globally_bounded_and_retain_first_stack():
    report = runner._report()
    errors = runner._Errors(report)
    errors.add("page_errors", "same", stack="first stack")
    errors.add("page_errors", "same", stack="second stack")
    errors.add("console_errors", "same")  # Different category is a separate error kind.
    for i in range(80):
        errors.add("http_errors", "long" * 1000, url=f"http://localhost/{i}")
    errors.add("page_errors", "same", stack="third stack")
    entries = sum(
        (
            report[key]
            for key in (
                "page_errors",
                "console_errors",
                "request_failures",
                "http_errors",
            )
        ),
        [],
    )
    assert len(entries) == 50
    assert report["page_errors"][0] == {
        "message": "same",
        "stack": "first stack",
        "count": 3,
    }
    assert report["error_count"] == 84
    assert (
        sum(entry["count"] for entry in entries) + report["omitted_error_count"] == 84
    )
    assert all(
        sum(len(value) for value in entry.values() if isinstance(value, str)) <= 2000
        for entry in entries
    )


def test_long_error_prefixes_do_not_conflate_distinct_messages():
    report = runner._report()
    errors = runner._Errors(report)
    for suffix in ("a", "b", "a"):
        errors.add("console_errors", "x" * 3000 + suffix)
    assert [entry["count"] for entry in report["console_errors"]] == [2, 1]


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("http://localhost/", True),
        ("https://localhost/", True),
        ("data:image/png;base64,abc", True),
        ("blob:http://localhost/id", True),
        ("file:///tmp/private", False),
        ("ftp://localhost/", False),
        ("javascript:alert(1)", False),
        ("custom:application", False),
    ],
)
def test_scheme_guard(url, allowed):
    route = SimpleNamespace(
        request=SimpleNamespace(url=url), continue_=AsyncMock(), abort=AsyncMock()
    )
    report = runner._report()
    asyncio.run(runner._guard_request(route, runner._Errors(report)))
    assert route.continue_.await_count == int(allowed)
    assert route.abort.await_count == int(not allowed)
    assert report["error_count"] == int(not allowed)


@pytest.mark.parametrize(
    "message,fallback",
    [
        ("Executable doesn't exist at /missing/chromium", True),
        ("Sandbox initialization failed", False),
        ("Permission denied", False),
    ],
)
def test_only_missing_bundled_executable_falls_back_to_sandboxed_chrome(
    message, fallback
):
    browser = object()
    chromium = SimpleNamespace(
        launch=AsyncMock(side_effect=[RuntimeError(message), browser])
    )

    async def launch():
        return await runner._launch(
            SimpleNamespace(chromium=chromium), asyncio.get_running_loop().time() + 10
        )

    if fallback:
        assert asyncio.run(launch()) is browser
        assert chromium.launch.await_args_list[1].kwargs["channel"] == "chrome"
    else:
        with pytest.raises(RuntimeError, match=message):
            asyncio.run(launch())
    assert chromium.launch.await_count == (2 if fallback else 1)
    for call in chromium.launch.await_args_list:
        assert call.kwargs["chromium_sandbox"] is True
        assert "user_data_dir" not in call.kwargs
    assert "channel" not in chromium.launch.await_args_list[0].kwargs


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["{}", "{}"],
        ["broken"],
        ["[]"],
        ["{}"],
        ['{"url":"http://localhost/", "unknown": true}'],
    ],
)
def test_cli_argument_errors_are_json(args, capsys):
    assert runner.main(args) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert len(output.out.splitlines()) == 1
    assert report["passed"] is False
    assert report["execution_error"]


def test_cli_keeps_library_stdout_off_report(monkeypatch, capsys):
    async def inspect(**kwargs):
        print("library diagnostic")
        report = runner._report()
        report["passed"] = True
        return report

    monkeypatch.setattr(runner, "inspect_page", inspect)
    assert runner.main(['{"url":"http://localhost/"}']) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["passed"] is True
    assert output.err == "library diagnostic\n"


@pytest.fixture
def fake_browser(monkeypatch):
    events = {}

    async def goto(*args, **kwargs):
        assert set(events) == {"pageerror", "console", "requestfailed", "response"}
        assert kwargs["wait_until"] == "domcontentloaded"

    page = SimpleNamespace(
        on=lambda name, callback: events.__setitem__(name, callback),
        goto=AsyncMock(side_effect=goto),
        screenshot=AsyncMock(return_value=b"png"),
    )

    async def close_context():
        # A request interrupted by runner cleanup must not fail a healthy inspection.
        if "requestfailed" in events:
            events["requestfailed"](
                SimpleNamespace(
                    failure="net::ERR_ABORTED", url="http://localhost/long-poll"
                )
            )

    context = SimpleNamespace(
        route=AsyncMock(),
        new_page=AsyncMock(return_value=page),
        close=AsyncMock(side_effect=close_context),
    )
    browser = SimpleNamespace(
        new_context=AsyncMock(return_value=context), close=AsyncMock()
    )
    playwright = SimpleNamespace(
        chromium=SimpleNamespace(launch=AsyncMock(return_value=browser))
    )

    @asynccontextmanager
    async def driver():
        yield playwright

    monkeypatch.setitem(
        sys.modules, "playwright.async_api", SimpleNamespace(async_playwright=driver)
    )
    return SimpleNamespace(page=page, context=context, browser=browser)


def test_cleanup_request_abort_is_outside_observation_window(fake_browser):
    report = asyncio.run(runner.inspect_page("http://localhost/", wait_ms=0))
    assert report["passed"] is True
    assert report["error_count"] == 0
    fake_browser.context.close.assert_awaited_once()
    fake_browser.browser.close.assert_awaited_once()
    fake_browser.page.screenshot.assert_awaited_once()
    assert fake_browser.page.screenshot.await_args.kwargs["full_page"] is False
    assert fake_browser.browser.new_context.await_args.kwargs["viewport"] == {
        "width": 1400,
        "height": 900,
    }
    assert (
        fake_browser.browser.new_context.await_args.kwargs["service_workers"] == "block"
    )


@pytest.mark.parametrize("stage", ["goto", "screenshot", "new_page", "close"])
def test_errors_still_close_context_and_browser(fake_browser, stage):
    owner = (
        fake_browser.page if stage in {"goto", "screenshot"} else fake_browser.context
    )
    getattr(owner, stage).side_effect = RuntimeError(f"{stage} failed")
    report = asyncio.run(runner.inspect_page("http://localhost/", wait_ms=0))
    assert report["passed"] is False
    assert f"{stage} failed" in report["execution_error"]
    fake_browser.context.close.assert_awaited_once()
    fake_browser.browser.close.assert_awaited_once()
    if stage in {"goto", "close"}:
        assert report["screenshot_base64"] is not None


@pytest.mark.parametrize("stage", ["goto", "screenshot"])
def test_timeout_covers_navigation_and_screenshot(fake_browser, stage):
    async def stall(*args, **kwargs):
        await asyncio.Event().wait()

    getattr(fake_browser.page, stage).side_effect = stall
    start = time.monotonic()
    report = asyncio.run(
        runner.inspect_page("http://localhost/", timeout=0.3, wait_ms=0)
    )
    assert time.monotonic() - start < 1
    assert report["passed"] is False
    assert "time" in report["execution_error"].lower()
    fake_browser.context.close.assert_awaited_once()
    fake_browser.browser.close.assert_awaited_once()
    if stage == "goto":
        assert report["screenshot_base64"] is not None
    else:
        assert report["screenshot_base64"] is None


def test_cancellation_still_closes_resources(fake_browser):
    async def scenario():
        started = asyncio.Event()

        async def stall(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        fake_browser.page.goto.side_effect = stall
        task = asyncio.create_task(runner.inspect_page("http://localhost/", wait_ms=0))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    fake_browser.context.close.assert_awaited_once()
    fake_browser.browser.close.assert_awaited_once()
