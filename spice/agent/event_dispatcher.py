"""Agent event dispatching for observers and frontend listeners."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from spice.agent.events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    TextDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from spice.extensions.manager import ExtensionEvent, ExtensionManager

AgentSessionListener = Callable[[AgentEvent], Any]


class AgentEventDispatcher:
    """Dispatch agent events to extension observers and frontend listeners.

    Dispatch order is part of the behavior contract: extension observers run
    first, then registered listeners run in subscription order. Listeners are
    awaited sequentially; listener exceptions are isolated and accumulated in
    ``listener_errors``.
    """

    def __init__(self, extensions: ExtensionManager | None = None) -> None:
        self.extensions = extensions
        self._listeners: list[AgentSessionListener] = []
        self.listener_errors: list[str] = []

    def subscribe(self, listener: AgentSessionListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    async def dispatch(self, event: AgentEvent) -> None:
        await self._emit_extension_observer_event(event)
        for listener in tuple(self._listeners):
            try:
                result = listener(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self.listener_errors.append(str(exc))

    async def _emit_extension_observer_event(self, event: AgentEvent) -> None:
        if self.extensions is None:
            return
        name, data = _extension_event_payload(event)
        if not name:
            return
        await self.extensions.emit(name, ExtensionEvent(type=name, data=data))


def _extension_event_payload(event: AgentEvent) -> tuple[str | None, dict[str, Any]]:
    if isinstance(event, AgentStartEvent):
        return "agent_start", {"session_id": event.session_id}
    if isinstance(event, AgentEndEvent):
        return "agent_end", {"session_id": event.session_id}
    if isinstance(event, TurnStartEvent):
        return "turn_start", {"prompt": event.prompt}
    if isinstance(event, TurnEndEvent):
        return "turn_end", {"text": event.text}
    if isinstance(event, TextDeltaEvent):
        return "text_delta", {"text": event.text}
    if isinstance(event, AssistantMessageEvent):
        return "assistant_message", {"text": event.text, "tool_calls": event.tool_calls}
    if isinstance(event, ToolExecutionStartEvent):
        return "tool_execution_start", {"tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "args": event.args}
    if isinstance(event, ToolExecutionEndEvent):
        return "tool_execution_end", {"tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "result": event.result}
    if isinstance(event, AgentErrorEvent):
        return "agent_error", {"message": event.message}
    return None, {}
