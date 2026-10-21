from __future__ import annotations

import asyncio
import base64
import json
import shlex
from io import BytesIO

import pytest
from PIL import Image

from spice.sandbox.base import ExecResult
from spice.tools import browser
from spice.tools.base import ToolContext
from spice.tools.verification import verification_scope


@pytest.fixture
def screenshot():
    out = BytesIO()
    Image.new("RGB", (10, 10), "red").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


@pytest.fixture
def report(screenshot):
    return {"passed": True, "page_errors": [], "console_errors": [], "request_failures": [],
            "http_errors": [], "checks": [], "error_count": 0, "screenshot_base64": screenshot,
            "execution_error": None}


class FakeEnvironment:
    name = "local"

    def __init__(self, report, *, exit_code=0, timed_out=False):
        self.report = report
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.calls = []

    async def run(self, command, *, cwd, timeout):
        self.calls.append((command, cwd, timeout))
        text = json.dumps(self.report)
        return ExecResult(output=text, stdout=text, stderr="", exit_code=self.exit_code, timed_out=self.timed_out)


@pytest.fixture(autouse=True)
def installed(monkeypatch):
    monkeypatch.setattr(browser, "browser_dependency_available", lambda: True)


def run_check(tmp_path, report, *, args=None, vision=True, exit_code=0, timed_out=False):
    environment = FakeEnvironment(report, exit_code=exit_code, timed_out=timed_out)
    records = []
    context = ToolContext(cwd=tmp_path, environment=environment, supports_vision=vision)
    with verification_scope(records.append):
        result = asyncio.run(browser.browser_check(args or {"url": "http://localhost:8000"}, context))
    return result, records, environment


def test_success_returns_image_and_verification(tmp_path, report):
    result, records, env = run_check(tmp_path, report)
    assert not result.is_error
    assert result.images
    assert records[0]["status"] == "verified"
    assert "screenshot_base64" not in result.content
    assert result.images[0].data not in result.content
    argv = shlex.split(env.calls[0][0])
    assert argv[1:4] == ["-I", "-m", "spice.tools.browser_runner"]


@pytest.mark.parametrize("field", browser.ERROR_FIELDS)
def test_error_even_with_exit_zero_is_failure_and_keeps_image(tmp_path, report, field):
    report[field] = [{"message": "failure", "count": 506}]
    report["error_count"] = 506
    result, records, _ = run_check(tmp_path, report)
    assert result.is_error and result.images
    assert records[0]["status"] == "failed"
    assert result.details["exit_code"] == 1
    assert not json.loads(result.content)["passed"]


def test_failed_assertion_and_nonzero_exit_cannot_pass(tmp_path, report):
    report["checks"] = [{"selector": "#missing", "passed": False}]
    result, records, _ = run_check(tmp_path, report, args={"url": "http://localhost", "selectors": ["#missing"]})
    assert result.is_error and records[0]["status"] == "failed"
    report["checks"] = []
    result, records, _ = run_check(tmp_path, report, exit_code=1)
    assert result.is_error and records[0]["status"] == "failed"


@pytest.mark.parametrize("mutation", ["missing-results", "missing-image", "invalid-image", "missing-assertion", "wrong-assertion", "overflow-errors"])
def test_invalid_report_never_records_verified(tmp_path, report, mutation):
    args = {"url": "http://localhost"}
    if mutation == "missing-results":
        report.pop("page_errors")
    elif mutation == "missing-image":
        report.pop("screenshot_base64")
    elif mutation == "invalid-image":
        report["screenshot_base64"] = "not base64"
    elif mutation == "missing-assertion":
        args["selectors"] = ["#ready"]
    elif mutation == "wrong-assertion":
        args["selectors"] = ["#ready"]
        report["checks"] = [{"selector": "#other", "passed": True}]
    else:
        report["error_count"] = 1000
    result, records, _ = run_check(tmp_path, report, args=args)
    assert result.is_error
    assert records[0]["status"] == "failed"


def test_text_model_can_check_runtime_but_cannot_claim_visual_inspection(tmp_path, report):
    result, records, _ = run_check(tmp_path, report, vision=False)
    assert not result.is_error and not result.images
    assert "visual inspection was not performed" in result.content
    assert records[0]["status"] == "verified"


def test_missing_dependency_records_failure_without_execution(tmp_path, report, monkeypatch):
    monkeypatch.setattr(browser, "browser_dependency_available", lambda: False)
    result, records, environment = run_check(tmp_path, report)
    assert result.is_error and "browser extra" in result.content
    assert records[0]["status"] == "failed"
    assert not environment.calls


def test_timeout_and_cancellation_do_not_verify(tmp_path, report):
    result, records, _ = run_check(tmp_path, report, timed_out=True)
    assert result.is_error and records[0]["status"] == "failed"
    environment = FakeEnvironment(report)
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError
    environment.run = cancel
    records = []
    with verification_scope(records.append), pytest.raises(asyncio.CancelledError):
        asyncio.run(browser.browser_check({"url": "http://localhost"}, ToolContext(cwd=tmp_path, environment=environment)))
    assert records[0]["status"] == "unverified"


def test_does_not_bypass_nonlocal_environment(tmp_path, report):
    env = FakeEnvironment(report)
    env.name = "docker"
    result = asyncio.run(browser.browser_check({"url": "http://localhost"}, ToolContext(cwd=tmp_path, environment=env)))
    assert result.error_code == "browser_environment_unsupported"
    assert not env.calls


@pytest.mark.parametrize("url", ["file:///etc/passwd", "data:text/html,hi", "javascript:alert(1)", "http://user:password@localhost"])
def test_rejects_unsafe_url_schemes_and_credentials(tmp_path, report, url):
    result, records, environment = run_check(tmp_path, report, args={"url": url})
    assert result.is_error
    assert not records and not environment.calls


def test_shell_metacharacters_are_quoted_not_executed(tmp_path, report):
    url = "http://localhost/?q=';touch /tmp/not-executed;#"
    result, _, environment = run_check(tmp_path, report, args={"url": url})
    assert not result.is_error
    argv = shlex.split(environment.calls[0][0])
    assert len(argv) == 5 and json.loads(argv[-1])["url"] == url


def test_browser_tool_requires_confirmation():
    assert browser.create_browser_tools()[0].requires_confirmation


@pytest.fixture
def local_site():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/missing.js":
                self.send_error(404)
                return
            script = ""
            if self.path == "/errors":
                script = "<script>for(let i=0;i<506;i++)setTimeout(()=>{const item=null;item.color},0)</script>"
            elif self.path == "/resource":
                script = '<script src="/missing.js"></script>'
            html = ('<!doctype html><link rel="icon" href="data:,"><main id="ready">Ready</main>' + script).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("path,selectors,passed", [
    ("/", ["#ready"], True), ("/errors", [], False),
    ("/resource", [], False), ("/", ["#missing"], False),
])
def test_real_browser_tool_end_to_end(tmp_path, local_site, path, selectors, passed):
    pytest.importorskip("playwright.async_api")
    records = []
    with verification_scope(records.append):
        result = asyncio.run(browser.browser_check(
            {"url": local_site + path, "selectors": selectors, "wait_ms": 500},
            ToolContext(cwd=tmp_path, supports_vision=True),
        ))
    report = json.loads(result.content)
    reason = str(report.get("execution_error") or "")
    if "Executable doesn't exist" in reason or "Chromium distribution" in reason:
        pytest.skip(reason)
    assert report["passed"] == passed, result.content
    assert result.is_error == (not passed), result.content
    assert records[-1]["status"] == ("verified" if passed else "failed")
    assert result.images and result.images[0].media_type == "image/png", result.content
    if path == "/errors":
        assert sum(error["count"] for error in report["page_errors"]) == 506
        assert len(report["page_errors"]) == 1
    if path == "/resource":
        assert any(error["status"] == 404 for error in report["http_errors"])
