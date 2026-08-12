import asyncio

from spice.agent.hooks import CompactionCompleted, HookManager


def test_hook_manager_runs_sync_and_async_handlers_in_registration_order(tmp_path) -> None:
    manager = HookManager()
    calls = []

    def first(event) -> None:
        calls.append(("first", event.summary))

    async def second(event) -> None:
        calls.append(("second", event.summary))

    manager.register(CompactionCompleted, first)
    manager.register(CompactionCompleted, second)

    asyncio.run(manager.emit(_compaction_event(tmp_path)))

    assert calls == [("first", "summary"), ("second", "summary")]


def test_hook_manager_unsubscribe_is_idempotent(tmp_path) -> None:
    manager = HookManager()
    calls = []
    unsubscribe = manager.register(CompactionCompleted, lambda _event: calls.append("called"))

    unsubscribe()
    unsubscribe()
    asyncio.run(manager.emit(_compaction_event(tmp_path)))

    assert calls == []


def test_hook_manager_isolates_handler_errors_and_continues(tmp_path) -> None:
    manager = HookManager()
    calls = []

    def broken(_event) -> None:
        raise RuntimeError("history unavailable")

    manager.register(CompactionCompleted, broken)
    manager.register(CompactionCompleted, lambda _event: calls.append("continued"))

    asyncio.run(manager.emit(_compaction_event(tmp_path)))

    assert calls == ["continued"]
    assert len(manager.handler_errors) == 1
    assert "history unavailable" in manager.handler_errors[0]


def test_hook_manager_ignores_events_without_handlers(tmp_path) -> None:
    manager = HookManager()

    asyncio.run(manager.emit(_compaction_event(tmp_path)))

    assert manager.handler_errors == []


def _compaction_event(workspace) -> CompactionCompleted:
    return CompactionCompleted(
        session_id="session-1",
        workspace=workspace,
        summary="summary",
        reason="manual",
        focus=None,
        first_kept_entry_id="entry-1",
        tokens_before=100,
        tokens_after=40,
    )
