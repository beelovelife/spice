"""Shared activity labels and animation frames for CLI and TUI frontends."""

from __future__ import annotations

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def format_elapsed(seconds: int) -> str:
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining = divmod(seconds, 60)
    return f"{minutes}m{remaining:02d}s"


def activity_parts(label: str, elapsed_seconds: float) -> tuple[str, str]:
    """Return the animated marker and a stable, human-readable status label."""
    elapsed_seconds = max(0.0, elapsed_seconds)
    frame_index = int(elapsed_seconds * 8) % len(SPINNER_FRAMES)
    return SPINNER_FRAMES[
        frame_index
    ], f"{label} · {format_elapsed(int(elapsed_seconds))}"


def activity_text(label: str, elapsed_seconds: float) -> str:
    marker, status = activity_parts(label, elapsed_seconds)
    return f"{marker} {status}"
