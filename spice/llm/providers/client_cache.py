"""Per-event-loop cache for provider SDK clients.

Provider SDK clients (AsyncOpenAI/AsyncAnthropic/genai.Client) each own an
httpx connection pool. Creating one per request leaks connections across a
multi-round tool loop, so clients are cached and reused per credentials.

The pool is bound to the running event loop, so the cache is keyed by loop
(weakly, letting per-loop clients die with their loop after asyncio.run).
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Hashable
from typing import Any

_cache: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Hashable, Any]] = weakref.WeakKeyDictionary()


def get_cached_client(key: Hashable, factory: Callable[[], Any]) -> Any:
    """Return the cached client for key on the running loop, creating it once."""
    loop = asyncio.get_running_loop()
    per_loop = _cache.setdefault(loop, {})
    client = per_loop.get(key)
    if client is None:
        client = factory()
        per_loop[key] = client
    return client
