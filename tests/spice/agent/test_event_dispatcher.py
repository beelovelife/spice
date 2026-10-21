from __future__ import annotations

import asyncio

from spice.agent.event_dispatcher import AgentEventDispatcher
from spice.agent.events import (
    ReasoningDeltaEvent,
    RoundCompleteEvent,
    ToolExecutionEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    TurnFinalizationStartEvent,
)
from spice.agent.turn_outcome import TurnCheckpoint
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
        unsubscribe = dispatcher.subscribe(
            lambda event: seen.append(type(event).__name__)
        )

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
        await dispatcher.dispatch(
            ToolExecutionEndEvent(
                tool_call_id="tc1", tool_name="read_file", result=tool_result("ok")
            )
        )
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


def test_pause_and_checkpoint_are_shared_with_extension_observers():
    async def run():
        extensions = FakeExtensions()
        dispatcher = AgentEventDispatcher(extensions)
        await dispatcher.dispatch(TurnFinalizationStartEvent(100, 100))
        checkpoint = TurnCheckpoint("task", 100, 100, summary="继续验收")
        await dispatcher.dispatch(TurnEndEvent(checkpoint.summary, "max_tool_rounds", checkpoint=checkpoint))
        return extensions.seen

    seen = asyncio.run(run())
    assert seen[0] == ("turn_finalization_start", {"rounds_used": 100, "round_limit": 100})
    assert seen[1][0] == "turn_end"
    assert seen[1][1]["status"] == "paused"
    assert seen[1][1]["checkpoint"]["summary"] == "继续验收"


def test_reasoning_delta_is_forwarded_to_extensions() -> None:
    async def run():
        extensions = FakeExtensions()
        dispatcher = AgentEventDispatcher(extensions)
        await dispatcher.dispatch(ReasoningDeltaEvent("checking", kind="summary"))
        return extensions.seen

    assert asyncio.run(run()) == [
        ("reasoning_delta", {"text": "checking", "kind": "summary"})
    ]


def test_extension_observer_failure_does_not_hide_events_from_frontends():
    class BrokenObserver:
        async def emit(self, *_args):
            raise RuntimeError("observer failed: api_key=private-value")

    async def run():
        dispatcher = AgentEventDispatcher(BrokenObserver())
        seen = []
        dispatcher.subscribe(seen.append)
        event = TurnStartEvent("run")
        await dispatcher.dispatch(event)
        assert seen == [event]
        assert len(dispatcher.listener_errors) == 1
        assert "private-value" not in dispatcher.listener_errors[0]

    asyncio.run(run())
