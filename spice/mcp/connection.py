"""A single MCP server connection backed by the official Python SDK.

The transport context is owned by one long-lived asyncio task.  This matters
because the SDK uses anyio cancel scopes which must be entered and exited by
the same task, while CLI/TUI prompts and shutdown may run in different tasks.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spice.llm.config import CONFIG_DIR
from spice.mcp.config import McpServerConfig

_SAFE_ENV = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR"}
_SECRET_PATTERN = re.compile(
    r"(?:Bearer\s+\S+|(?:api[_-]?key|token|password|secret)\s*[=:]\s*\S+|sk-[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


@dataclass
class _Request:
    operation: str
    payload: Any
    future: asyncio.Future[Any]


class McpConnection:
    def __init__(self, config: McpServerConfig, *, cwd: Path) -> None:
        self.config = config
        self.cwd = cwd
        self.session: Any | None = None
        self.tools: list[Any] = []
        self.stale = False
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[list[Any]] | None = None
        self._queue: asyncio.Queue[_Request] | None = None
        self._start_lock = asyncio.Lock()
        self._stderr = None

    async def connect(self) -> list[Any]:
        async with self._start_lock:
            if self._task is not None and not self._task.done() and self._ready is not None:
                ready = self._ready
            else:
                loop = asyncio.get_running_loop()
                self._ready = loop.create_future()
                self._queue = asyncio.Queue()
                self._task = asyncio.create_task(self._run(), name=f"spice-mcp-{self.config.name}")
                ready = self._ready
        return await asyncio.shield(ready)

    async def refresh_tools(self) -> list[Any]:
        await self.connect()
        tools = await self._request("list_tools", None)
        self.tools = list(tools)
        self.stale = False
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        await self.connect()
        try:
            return await self._request("call_tool", (name, arguments))
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed response does not prove the remote tool did not run. Tear
            # down the broken owner so a later, explicit call can reconnect, but
            # never replay this potentially non-idempotent operation.
            await self._wait_for_owner_exit()
            raise

    async def close(self) -> None:
        task = self._task
        if task is None:
            return
        if not task.done() and self._queue is not None:
            try:
                await self._request("close", None)
            except Exception:
                pass
        await self._wait_for_owner_exit()

    async def _request(self, operation: str, payload: Any) -> Any:
        if self._task is None or self._task.done() or self._queue is None:
            raise RuntimeError(f"MCP server {self.config.name!r} is not connected")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(_Request(operation, payload, future))
        return await future

    async def _run(self) -> None:
        stack = AsyncExitStack()
        fatal: BaseException | None = None
        try:
            session, listed = await self._open(stack)
            self.session = session
            self.tools = list(listed.tools)
            self.stale = False
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(self.tools)
            while True:
                assert self._queue is not None
                request = await self._queue.get()
                if request.operation == "close":
                    request.future.set_result(None)
                    break
                try:
                    if request.operation == "list_tools":
                        async with asyncio.timeout(self.config.connect_timeout):
                            result = await session.list_tools()
                        request.future.set_result(list(result.tools))
                    elif request.operation == "call_tool":
                        name, arguments = request.payload
                        async with asyncio.timeout(self.config.tool_timeout):
                            result = await session.call_tool(name, arguments=arguments)
                        request.future.set_result(result)
                    else:
                        request.future.set_exception(RuntimeError(f"Unknown MCP operation: {request.operation}"))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    request.future.set_exception(exc)
                    raise
        except asyncio.CancelledError as exc:
            fatal = exc
            raise
        except BaseException as exc:
            fatal = exc
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            self.session = None
            self.tools = []
            self.stale = False
            self._fail_pending(fatal or RuntimeError(f"MCP server {self.config.name!r} closed"))
            try:
                await stack.aclose()
            finally:
                self._close_stderr()

    async def _open(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        if self.config.transport == "stdio":
            log_path = CONFIG_DIR / "logs" / f"mcp-{_safe_name(self.config.name)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr = log_path.open("a", encoding="utf-8", errors="replace", buffering=1)
            env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV or key.startswith("XDG_")}
            env.update(self.config.env)
            params = StdioServerParameters(
                command=self.config.command or "",
                args=list(self.config.args),
                env=env,
                cwd=self.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params, errlog=self._stderr))
        else:
            read, write, _session_id = await stack.enter_async_context(
                streamablehttp_client(
                    self.config.url or "",
                    headers=self.config.headers,
                    timeout=self.config.connect_timeout,
                )
            )

        async def message_handler(message: Any) -> None:
            if "ToolListChanged" in type(getattr(message, "root", message)).__name__:
                self.stale = True

        session = await stack.enter_async_context(ClientSession(read, write, message_handler=message_handler))
        async with asyncio.timeout(self.config.connect_timeout):
            await session.initialize()
            listed = await session.list_tools()
        return session, listed

    async def _wait_for_owner_exit(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        self._ready = None
        self._queue = None

    def _fail_pending(self, error: BaseException) -> None:
        if self._queue is None:
            return
        while not self._queue.empty():
            request = self._queue.get_nowait()
            if not request.future.done():
                request.future.set_exception(error)

    def _close_stderr(self) -> None:
        if self._stderr is not None:
            try:
                self._stderr.close()
            finally:
                self._stderr = None


def sanitize_mcp_error(value: BaseException | str) -> str:
    text = str(value).strip() or repr(value)
    return _SECRET_PATTERN.sub("[REDACTED]", text)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
