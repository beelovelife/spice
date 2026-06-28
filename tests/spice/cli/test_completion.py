from __future__ import annotations

from pathlib import Path

from prompt_toolkit.document import Document

from spice.cli.commands import SlashCommandRegistry
from spice.cli.completion import SpiceInputCompleter, accept_completion, file_completion_candidates, insert_text_and_maybe_complete, start_or_accept_completion


def test_file_completion_candidates_list_files_and_directories(tmp_path) -> None:
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")

    candidates = file_completion_candidates(tmp_path, "")

    assert [candidate.text for candidate in candidates] == ["src/", "README.md"]


def test_file_completion_candidates_respect_directory_prefix(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')", encoding="utf-8")
    (src / "util.py").write_text("pass", encoding="utf-8")

    candidates = file_completion_candidates(tmp_path, "src/ma")

    assert [candidate.text for candidate in candidates] == ["src/main.py"]


def test_file_completion_candidates_list_directory_when_prefix_ends_with_slash(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')", encoding="utf-8")

    candidates = file_completion_candidates(tmp_path, "src/")

    assert [candidate.text for candidate in candidates] == ["src/main.py"]


def test_file_completion_candidates_ignores_permission_errors(tmp_path, monkeypatch) -> None:
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    original_iterdir = Path.iterdir

    def fake_iterdir(path):
        if path == desktop:
            raise PermissionError("Operation not permitted")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    assert file_completion_candidates(tmp_path, "Desktop/") == []


def test_spice_input_completer_uses_file_completion_after_at(tmp_path) -> None:
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    completer = SpiceInputCompleter(SlashCommandRegistry().completer(), cwd=tmp_path)

    completions = list(completer.get_completions(Document("summarize @REA"), None))

    assert [completion.text for completion in completions] == ["README.md"]


def test_spice_input_completer_styles_directories_and_files_differently(tmp_path) -> None:
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "src").mkdir()
    completer = SpiceInputCompleter(SlashCommandRegistry().completer(), cwd=tmp_path)

    completions = list(completer.get_completions(Document("@"), None))

    by_text = {completion.text: completion for completion in completions}
    assert by_text["src/"].display[0][0] == "class:completion.directory"
    assert by_text["README.md"].display[0][0] == "class:completion.file"


def test_spice_input_completer_keeps_slash_completion(tmp_path) -> None:
    completer = SpiceInputCompleter(SlashCommandRegistry().completer(), cwd=tmp_path)

    completions = list(completer.get_completions(Document("/skill"), None))

    assert [completion.text for completion in completions] == ["/skills", "/skill"]


def test_spice_input_completer_supports_file_colon_prefix(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("pass", encoding="utf-8")
    completer = SpiceInputCompleter(SlashCommandRegistry().completer(), cwd=tmp_path)

    completions = list(completer.get_completions(Document("@file:src/a"), None))

    assert [completion.text for completion in completions] == ["src/app.py"]


def test_root_completion_is_limited_to_safe_entrypoints(tmp_path) -> None:
    candidates = file_completion_candidates(tmp_path, "/")
    candidate_texts = [candidate.text for candidate in candidates]

    assert "/bin/" not in candidate_texts
    assert "/dev/" not in candidate_texts
    assert "/cores/" not in candidate_texts
    assert all(text.removeprefix("/").rstrip("/") in {"Applications", "Users", "Volumes", "tmp", "private"} for text in candidate_texts)


def test_accept_completion_reopens_menu_after_directory_completion() -> None:
    class FakeCompletion:
        text = "src/"

    class FakeState:
        current_completion = FakeCompletion()

    class FakeBuffer:
        complete_state = FakeState()

        def __init__(self) -> None:
            self.applied = None
            self.reopened = False

        def apply_completion(self, completion) -> None:
            self.applied = completion.text

        def start_completion(self, *, select_first: bool) -> None:
            self.reopened = select_first

    buffer = FakeBuffer()

    assert accept_completion(buffer)
    assert buffer.applied == "src/"
    assert buffer.reopened


def test_accept_completion_uses_first_candidate_when_current_is_missing() -> None:
    class FakeCompletion:
        text = "README.md"

    class FakeState:
        current_completion = None
        completions = [FakeCompletion()]

    class FakeBuffer:
        complete_state = FakeState()

        def __init__(self) -> None:
            self.applied = None
            self.inserted = ""

        def apply_completion(self, completion) -> None:
            self.applied = completion.text

        def insert_text(self, text: str) -> None:
            self.inserted += text

        def start_completion(self, *, select_first: bool) -> None:
            raise AssertionError("should not reopen for files")

    buffer = FakeBuffer()

    assert accept_completion(buffer)
    assert buffer.applied == "README.md"
    assert buffer.inserted == " "


def test_tab_starts_completion_and_applies_first_candidate() -> None:
    class FakeCompletion:
        text = "README.md"

    class FakeState:
        current_completion = None
        completions = [FakeCompletion()]

    class FakeBuffer:
        complete_state = None

        def __init__(self) -> None:
            self.applied = None
            self.inserted = ""

        def start_completion(self, *, select_first: bool) -> None:
            assert select_first
            self.complete_state = FakeState()

        def apply_completion(self, completion) -> None:
            self.applied = completion.text

        def insert_text(self, text: str) -> None:
            self.inserted += text

    buffer = FakeBuffer()

    assert start_or_accept_completion(buffer)
    assert buffer.applied == "README.md"
    assert buffer.inserted == " "


def test_insert_at_starts_completion() -> None:
    class FakeDocument:
        text_before_cursor = "@"

    class FakeBuffer:
        document = FakeDocument()

        def __init__(self) -> None:
            self.inserted = ""
            self.started = False

        def insert_text(self, text: str) -> None:
            self.inserted += text

        def start_completion(self, *, select_first: bool) -> None:
            self.started = select_first

    buffer = FakeBuffer()

    assert insert_text_and_maybe_complete(buffer, "@")
    assert buffer.inserted == "@"
    assert buffer.started


def test_insert_slash_after_at_directory_starts_completion() -> None:
    class FakeDocument:
        text_before_cursor = "@src/"

    class FakeBuffer:
        document = FakeDocument()

        def __init__(self) -> None:
            self.inserted = ""
            self.started = False

        def insert_text(self, text: str) -> None:
            self.inserted += text

        def start_completion(self, *, select_first: bool) -> None:
            self.started = select_first

    buffer = FakeBuffer()

    assert insert_text_and_maybe_complete(buffer, "/")
    assert buffer.inserted == "/"
    assert buffer.started
