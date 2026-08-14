"""Built-in lifecycle hooks for the Spice agent runtime."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from spice.agent.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CompactionCompleted:
    """A compaction was durably committed to the session store."""

    session_id: str
    workspace: Path
    summary: str
    reason: Literal["manual", "auto"]
    focus: str | None
    first_kept_entry_id: str
    tokens_before: int
    tokens_after: int


HookHandler = Callable[[Any], Any]


class HookManager:
    """Dispatch non-critical lifecycle observations to isolated handlers.

    This manager is deliberately fail-open and must not be used for permission,
    sandbox, validation, or other security decisions that need to block work.
    """

    def __init__(self) -> None:
        self._handlers: dict[type, list[HookHandler]] = defaultdict(list)
        self.handler_errors: list[str] = []

    def register(self, event_type: type, handler: HookHandler) -> Callable[[], None]:
        handlers = self._handlers[event_type]
        handlers.append(handler)

        def unsubscribe() -> None:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    async def emit(self, event: object) -> None:
        for handler in tuple(self._handlers.get(type(event), ())):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                handler_name = getattr(handler, "__qualname__", type(handler).__qualname__)
                message = f"{type(event).__name__}:{handler_name}: {exc}"
                self.handler_errors.append(message)
                del self.handler_errors[:-100]
                logger.warning("hook_handler_failed event=%s handler=%s", type(event).__name__, handler_name, exc_info=True)
