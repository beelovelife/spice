"""The single-run adapter: assistant text or JSONL on stdout, diagnostics on stderr."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, is_dataclass, replace
from typing import TextIO

from rich.console import Console

from spice.agent.events import (
    AgentErrorEvent, AgentEvent, AssistantMessageEvent, ModelFallbackEvent,
    ModelRetryEvent, TextDeltaEvent, ToolExecutionEndEvent, ToolExecutionStartEvent, TurnEndEvent,
    TurnFinalizationStartEvent,
)
from spice.agent.turn_outcome import pause_notice
from spice.cli.render import CliRenderer


class RunOutput:
    def __init__(self, *, json_events: bool = False, markdown: bool = True,
                 stdout: TextIO | None = None, stderr: TextIO | None = None) -> None:
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self.json_events = json_events
        self.exit_code = 1  # Success requires an explicit, successful turn end.
        self._failed = False
        self._sequence = 0
        self._round_streamed = False
        self._last_assistant_text = ""
        self._line_open = False
        self._terminal = (
            CliRenderer(Console(file=self.stdout), markdown=True)
            if markdown and self.stdout.isatty() and not json_events else None
        )
        self._confirm_renderer = CliRenderer(Console(file=self.stderr), markdown=False)

    @property
    def can_confirm(self) -> bool:
        return not self.json_events and sys.stdin.isatty() and self.stderr.isatty()

    async def confirm(self, name: str, args: dict) -> bool:
        return await self._confirm_renderer.confirm(name, args)

    def diagnostic(self, text: str) -> None:
        self.stderr.write(text + "\n")
        self.stderr.flush()

    def render_event(self, event: AgentEvent) -> None:
        if isinstance(event, AgentErrorEvent):
            self._failed = True
            self.exit_code = 130 if event.kind in {"user_interrupted", "consumer_closed"} else 1
            self.diagnostic(event.message)
        elif isinstance(event, TurnEndEvent):
            if event.status == "interrupted":
                self.exit_code = 130
            elif event.status == "failed" or self._failed:
                self.exit_code = 1
            elif event.status == "paused":
                self.exit_code = 2
                self.diagnostic(pause_notice(event.checkpoint.round_limit if event.checkpoint else None))
            else:
                self.exit_code = 0

        if self.json_events:
            # Oversized tool output already has an artifact; never duplicate the
            # private full-output buffer into the event stream.
            if isinstance(event, ToolExecutionEndEvent):
                event = replace(event, result=replace(event.result, full_content=None))
            if not is_dataclass(event) or isinstance(event, type):
                raise TypeError("Agent events must be dataclass instances.")
            payload = asdict(event)
            if isinstance(event, ToolExecutionEndEvent):
                payload["result"].pop("full_content", None)
            event_type = re.sub(r"(?<!^)(?=[A-Z])", "_", type(event).__name__.removesuffix("Event")).lower()
            self.stdout.write(json.dumps({
                "format": "spice-events-1", "sequence": self._sequence,
                "type": event_type, "data": payload,
            }, ensure_ascii=False, default=str) + "\n")
            self.stdout.flush()
            self._sequence += 1
            return

        if isinstance(event, TextDeltaEvent):
            self._round_streamed = True
            self._write_text(event.text)
        elif isinstance(event, AssistantMessageEvent):
            if not self._round_streamed:
                self._write_text(event.text)
            self._last_assistant_text = event.text
            self._round_streamed = False
            self._finish_line()
        elif isinstance(event, TurnEndEvent):
            if event.text and not self._round_streamed and event.text != self._last_assistant_text:
                self._write_text(event.text)
            self._finish_line()
        elif isinstance(event, ToolExecutionStartEvent):
            self.diagnostic(f"Tool: {event.tool_name}")
        elif isinstance(event, ToolExecutionEndEvent) and event.result.is_error:
            self.diagnostic(f"Tool error ({event.tool_name}): {event.result.content}")
        elif isinstance(event, ModelRetryEvent):
            self.diagnostic(f"Retrying {event.provider}/{event.model}: {event.next_attempt}/{event.max_attempts}")
        elif isinstance(event, ModelFallbackEvent):
            self.diagnostic(f"Fallback: {event.to_provider}/{event.to_model}")
        elif isinstance(event, TurnFinalizationStartEvent):
            self.diagnostic("工具轮数预算已用完，正在整理进度。")

    def _write_text(self, text: str) -> None:
        if not text:
            return
        if self._terminal is not None:
            self._terminal.render_event(TextDeltaEvent(text))
        else:
            self.stdout.write(text)
            self.stdout.flush()
        self._line_open = not text.endswith("\n")

    def _finish_line(self) -> None:
        if self._terminal is not None:
            self._terminal.finish_response()
        elif self._line_open:
            self.stdout.write("\n")
            self.stdout.flush()
        self._line_open = False

    def finish_response(self) -> None:
        if not self.json_events:
            self._finish_line()
