"""Terminal helpers."""

from __future__ import annotations

from contextlib import contextmanager
import sys


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def supports_color() -> bool:
    return sys.stdout.isatty()


def enable_cursor_blink_after_render(application) -> None:
    """Undo prompt_toolkit's VT100 ``show_cursor`` blink suppression.

    ``Vt100_Output.show_cursor`` writes DEC mode 12 reset on every render.
    Running this as an ``Application.after_render`` handler makes the blinking
    mode the final cursor command sent to terminals that honor DEC mode 12.
    """
    output = application.output
    write_raw = getattr(output, "write_raw", None)
    if write_raw is None:
        return
    write_raw("\x1b[?12h")
    output.flush()


@contextmanager
def preserve_cursor_blink():
    try:
        yield
    finally:
        pass
