from __future__ import annotations

from types import SimpleNamespace

from spice.cli.terminal import enable_cursor_blink_after_render


def test_enable_cursor_blink_runs_after_prompt_toolkit_render() -> None:
    calls = []
    output = SimpleNamespace(
        write_raw=lambda value: calls.append(("write", value)),
        flush=lambda: calls.append(("flush", None)),
    )

    enable_cursor_blink_after_render(SimpleNamespace(output=output))

    assert calls == [("write", "\x1b[?12h"), ("flush", None)]


def test_enable_cursor_blink_ignores_outputs_without_vt100_writes() -> None:
    enable_cursor_blink_after_render(SimpleNamespace(output=SimpleNamespace()))
