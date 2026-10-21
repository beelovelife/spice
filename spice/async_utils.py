"""Small async iterator ownership helpers shared by runtime layers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import inspect
from typing import TypeVar

T = TypeVar("T")


@asynccontextmanager
async def closing_iterator(iterator: AsyncIterator[T]) -> AsyncIterator[AsyncIterator[T]]:
    """Close generators deterministically without requiring it of all iterators."""
    try:
        yield iterator
    finally:
        close = getattr(iterator, "aclose", None) or getattr(iterator, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
