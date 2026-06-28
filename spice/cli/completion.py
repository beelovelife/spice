"""Prompt-toolkit completions for Spice input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings


@dataclass(frozen=True)
class FileCandidate:
    text: str
    is_dir: bool = False


SAFE_ROOTS = {"Applications", "Users", "Volumes", "tmp", "private"}


def file_completion_candidates(cwd: Path, prefix: str) -> list[FileCandidate]:
    base, stem, absolute = _completion_base(cwd, prefix)
    try:
        children = list(base.iterdir())
    except (OSError, PermissionError):
        return []
    rows: list[FileCandidate] = []
    for child in children:
        if child.name.startswith("."):
            continue
        if absolute and base == Path("/") and child.name not in SAFE_ROOTS:
            continue
        if stem and not child.name.startswith(stem):
            continue
        rel = _candidate_text(prefix, child)
        rows.append(FileCandidate(rel + ("/" if child.is_dir() else ""), child.is_dir()))
    rows.sort(key=lambda item: (not item.is_dir, item.text.lower()))
    return rows


class SpiceInputCompleter(Completer):
    def __init__(self, slash_completer: Completer, *, cwd: Path | None = None) -> None:
        self.slash_completer = slash_completer
        self.cwd = cwd or Path.cwd()

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            yield from self.slash_completer.get_completions(document, complete_event)
            return
        marker = _file_marker_prefix(text)
        if marker is None:
            return
        marker_start, prefix = marker
        for candidate in file_completion_candidates(self.cwd, prefix):
            style = "class:completion.directory" if candidate.is_dir else "class:completion.file"
            yield Completion(
                candidate.text,
                start_position=marker_start - len(text),
                display=[(style, candidate.text)],
            )


def accept_completion(buffer) -> bool:
    state = getattr(buffer, "complete_state", None)
    if state is None:
        return False
    completion = getattr(state, "current_completion", None)
    if completion is None:
        completions = list(getattr(state, "completions", []) or [])
        if not completions:
            return False
        completion = completions[0]
    buffer.apply_completion(completion)
    if str(completion.text).endswith("/"):
        buffer.start_completion(select_first=True)
    else:
        buffer.insert_text(" ")
    return True


def start_or_accept_completion(buffer) -> bool:
    if getattr(buffer, "complete_state", None) is None:
        buffer.start_completion(select_first=True)
    return accept_completion(buffer)


def insert_text_and_maybe_complete(buffer, text: str) -> bool:
    buffer.insert_text(text)
    before = getattr(buffer.document, "text_before_cursor", "")
    if text == "@" or (text == "/" and _file_marker_prefix(before) is not None):
        buffer.start_completion(select_first=True)
    return True


def create_completion_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("tab")
    def _tab(event) -> None:
        start_or_accept_completion(event.current_buffer)

    @bindings.add("enter", filter=has_completions)
    def _enter(event) -> None:
        accept_completion(event.current_buffer)

    @bindings.add("@")
    def _at(event) -> None:
        insert_text_and_maybe_complete(event.current_buffer, "@")

    @bindings.add("/")
    def _slash(event) -> None:
        insert_text_and_maybe_complete(event.current_buffer, "/")

    return bindings


def _file_marker_prefix(text: str) -> tuple[int, str] | None:
    at = text.rfind("@")
    if at == -1:
        return None
    raw = text[at + 1 :]
    if any(char.isspace() for char in raw):
        return None
    if raw.startswith("file:"):
        return at + 1, raw[5:]
    return at + 1, raw


def _completion_base(cwd: Path, prefix: str) -> tuple[Path, str, bool]:
    absolute = prefix.startswith("/")
    raw = Path(prefix)
    if prefix.endswith("/"):
        base = raw if absolute else cwd / raw
        stem = ""
    else:
        base_part = raw.parent
        stem = raw.name
        base = raw.parent if absolute else cwd / base_part
    return base, stem, absolute


def _candidate_text(prefix: str, child: Path) -> str:
    if prefix.endswith("/"):
        return prefix + child.name
    parent = str(Path(prefix).parent)
    if parent in {"", "."}:
        return child.name if not prefix.startswith("/") else "/" + child.name
    return f"{parent}/{child.name}"
