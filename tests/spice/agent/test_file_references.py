from __future__ import annotations

from pathlib import Path

import pytest

from spice.agent.file_references import FileReferenceError, expand_file_references, parse_file_references
from spice.llm.models import Model


def _model(context_window: int = 128_000, supports_vision: bool = False) -> Model:
    return Model(id="test", provider="openai", context_window=context_window, supports_vision=supports_vision)


def test_parse_file_references_supports_paths_quotes_and_line_ranges() -> None:
    refs = parse_file_references('@README.md @file:src/app.py:3-5 @"a b.md" @folder:src/')

    assert [(ref.kind, ref.target, ref.line_start, ref.line_end) for ref in refs] == [
        ("file", "README.md", None, None),
        ("file", "src/app.py", 3, 5),
        ("file", "a b.md", None, None),
        ("folder", "src/", None, None),
    ]


def test_expand_file_reference_inlines_text_file(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# Title\nBody\n", encoding="utf-8")

    result = expand_file_references("Summarize @README.md", cwd=tmp_path, model=_model())

    assert result.expanded
    assert "Summarize @README.md" in result.message
    assert "Referenced files:" in result.message
    assert '<file path="README.md" type="markdown"' in result.message
    assert "# Title" in result.message


def test_expand_file_reference_supports_single_line_and_range(tmp_path: Path) -> None:
    path = tmp_path / "src.py"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    single = expand_file_references("@file:src.py:2", cwd=tmp_path, model=_model())
    ranged = expand_file_references("@file:src.py:2-3", cwd=tmp_path, model=_model())

    assert 'lines="2-2"' in single.message
    assert "two" in single.message
    assert "three" not in single.message
    assert 'lines="2-3"' in ranged.message
    assert "two\nthree" in ranged.message


def test_expand_file_reference_reports_invalid_line_range(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("one\n", encoding="utf-8")

    result = expand_file_references("@file:src.py:9", cwd=tmp_path, model=_model())

    assert "Reference warnings:" in result.message
    assert "beyond end of file" in result.message
    assert "Referenced files:" not in result.message


def test_expand_file_reference_uses_longest_existing_path_prefix(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    result = expand_file_references("@README.md总结一下", cwd=tmp_path, model=_model())

    assert '<file path="README.md"' in result.message
    assert "hello" in result.message


def test_expand_file_reference_truncates_large_file(tmp_path: Path) -> None:
    (tmp_path / "large.log").write_text("x" * 20_000, encoding="utf-8")

    result = expand_file_references("@large.log", cwd=tmp_path, model=_model())

    assert 'truncated="true"' in result.message
    assert "kept first 12000 characters of 20000" in result.message
    assert "Reference warnings:" in result.message


def test_expand_file_reference_applies_total_context_budget(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a" * 1000, encoding="utf-8")
    (tmp_path / "b.txt").write_text("b" * 1000, encoding="utf-8")

    result = expand_file_references("@a.txt @b.txt", cwd=tmp_path, model=_model(context_window=400))

    assert "exceed the 50% context budget" in result.message
    assert '<file path="a.txt"' in result.message


def test_expand_file_reference_rejects_image_when_model_lacks_vision(tmp_path: Path) -> None:
    (tmp_path / "ui.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(FileReferenceError, match="does not support image input"):
        expand_file_references("@ui.png analyze this", cwd=tmp_path, model=_model(supports_vision=False))


def test_expand_file_reference_uses_metadata_for_binary_file(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")

    result = expand_file_references("@data.bin", cwd=tmp_path, model=_model())

    assert 'path="data.bin"' in result.message
    assert 'injected="false"' in result.message
    assert "Binary file not inlined" in result.message


def test_expand_folder_reference_uses_directory_listing(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("a", encoding="utf-8")

    result = expand_file_references("@folder:docs", cwd=tmp_path, model=_model())

    assert 'type="directory"' in result.message
    assert "- a.md" in result.message


def test_expand_file_reference_rejects_common_secret_filenames(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")
    (tmp_path / "deploy.pem").write_text("secret", encoding="utf-8")
    (tmp_path / "credentials.json").write_text("{}", encoding="utf-8")

    env_result = expand_file_references("@.env", cwd=tmp_path, model=_model())
    pem_result = expand_file_references("@deploy.pem", cwd=tmp_path, model=_model())
    credentials_result = expand_file_references("@credentials.json", cwd=tmp_path, model=_model())

    assert "sensitive credential file" in env_result.message
    assert "sensitive credential file" in pem_result.message
    assert "sensitive credential file" in credentials_result.message
    assert "Referenced files:" not in env_result.message


def test_expand_file_reference_rejects_private_key_content(tmp_path: Path) -> None:
    path = tmp_path / "not-obvious.txt"
    path.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n-----END OPENSSH PRIVATE KEY-----", encoding="utf-8")

    result = expand_file_references("@not-obvious.txt", cwd=tmp_path, model=_model())

    assert "private key" in result.message
    assert "Referenced files:" not in result.message
