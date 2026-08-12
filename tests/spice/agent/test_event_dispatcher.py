from __future__ import annotations

import asyncio

from spice.agent.event_dispatcher import AgentEventDispatcher
from spice.agent.events import RoundCompleteEvent, ToolExecutionEndEvent, TurnStartEvent
from spice.extensions.manager import ExtensionEvent
from spice.tools.base import tool_result


class FakeExtensions:
    def __init__(self) -> None:
        self.seen = []

    async def emit(self, name: str, event: ExtensionEvent) -> ExtensionEvent:
        self.seen.append((name, event.data))
        return event


def test_agent_event_dispatcher_fans_out_and_unsubscribes() -> None:
    async def run():
        dispatcher = AgentEventDispatcher()
        seen = []
        unsubscribe = dispatcher.subscribe(lambda event: seen.append(type(event).__name__))

        await dispatcher.dispatch(TurnStartEvent(prompt="hello"))
        unsubscribe()
        await dispatcher.dispatch(TurnStartEvent(prompt="ignored"))
        return seen

    seen = asyncio.run(run())

    assert seen == ["TurnStartEvent"]


def test_agent_event_dispatcher_isolates_listener_errors() -> None:
    async def run():
        dispatcher = AgentEventDispatcher()
        seen = []

        def bad_listener(event):
            raise RuntimeError("listener failed")

        dispatcher.subscribe(bad_listener)
        dispatcher.subscribe(lambda event: seen.append(type(event).__name__))
        await dispatcher.dispatch(TurnStartEvent(prompt="hello"))
        return seen, dispatcher.listener_errors

    seen, errors = asyncio.run(run())

    assert seen == ["TurnStartEvent"]
    assert errors
    assert "listener failed" in errors[0]


def test_agent_event_dispatcher_forwards_extension_observer_events() -> None:
    async def run():
        extensions = FakeExtensions()
        dispatcher = AgentEventDispatcher(extensions)
        await dispatcher.dispatch(TurnStartEvent(prompt="hello"))
        await dispatcher.dispatch(ToolExecutionEndEvent(tool_call_id="tc1", tool_name="read_file", result=tool_result("ok")))
        return extensions.seen

    seen = asyncio.run(run())

    assert seen[0] == ("turn_start", {"prompt": "hello"})
    assert seen[1][0] == "tool_execution_end"
    assert seen[1][1]["tool_call_id"] == "tc1"
    assert seen[1][1]["tool_name"] == "read_file"
    assert seen[1][1]["result"].content == "ok"


def test_round_complete_is_forwarded_to_extensions() -> None:
    async def run():
        extensions = FakeExtensions()
        dispatcher = AgentEventDispatcher(extensions)
        await dispatcher.dispatch(RoundCompleteEvent(2))
        return extensions.seen

    assert asyncio.run(run()) == [("round_complete", {"round_index": 2})]
