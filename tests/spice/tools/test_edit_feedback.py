from __future__ import annotations

import asyncio
import os

import pytest

from spice.tools.base import ToolContext
from spice.tools.file import apply_patch, edit_file, write_file


@pytest.mark.parametrize("operation", ["write", "edit", "patch", "reversed_patch"])
def test_noop_edit_preserves_the_file_and_reports_no_change(tmp_path, operation):
    path = tmp_path / "a.txt"
    path.write_text("original")
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    before = path.stat().st_mtime_ns
    context = ToolContext(cwd=tmp_path)
    if operation == "write":
        request = write_file({"path": "a.txt", "content": "original"}, context)
    elif operation == "edit":
        request = edit_file({"path": "a.txt", "old_text": "original", "new_text": "original"}, context)
    else:
        operations = [{"type": "replace", "path": "a.txt", "old_text": "original", "new_text": "original"}]
        if operation == "reversed_patch":
            operations = [
                {"type": "replace", "path": "a.txt", "old_text": "original", "new_text": "temporary"},
                {"type": "replace", "path": "a.txt", "old_text": "temporary", "new_text": "original"},
            ]
        request = apply_patch({"operations": operations}, context)

    result = asyncio.run(request)

    assert not result.is_error and "No changes" in result.content
    assert result.details["changed"] is False and result.details["files_changed"] == 0
    assert result.details["changed_paths"] == []
    assert path.read_text() == "original" and path.stat().st_mtime_ns == before


def test_patch_counts_and_writes_only_the_changed_file(tmp_path):
    unchanged = tmp_path / "a.txt"
    unchanged.write_text("original")
    os.utime(unchanged, ns=(1_000_000_000, 1_000_000_000))
    result = asyncio.run(apply_patch({"operations": [
        {"type": "replace", "path": "a.txt", "old_text": "original", "new_text": "original"},
        {"type": "add", "path": "b.txt", "content": "new"},
    ]}, ToolContext(cwd=tmp_path)))

    assert result.details["changed"] is True and result.details["files_changed"] == 1
    assert result.details["changed_paths"] == [str((tmp_path / "b.txt").resolve())]
    assert unchanged.stat().st_mtime_ns == 1_000_000_000
    assert (tmp_path / "b.txt").read_text() == "new"


@pytest.mark.parametrize("operation", ["edit", "patch"])
def test_dry_run_reports_a_plan_without_claiming_a_completed_edit(tmp_path, operation):
    path = tmp_path / "a.txt"
    path.write_text("original")
    args = {"path": "a.txt", "old_text": "original", "new_text": "updated"}
    context = ToolContext(cwd=tmp_path)
    if operation == "edit":
        request = edit_file({**args, "dry_run": True}, context)
    else:
        request = apply_patch({"operations": [{"type": "replace", **args}], "dry_run": True}, context)
    result = asyncio.run(request)

    assert result.details["changed"] is False and result.details["would_change"] is True
    assert result.details["changed_paths"] == []
    assert result.details["files_changed"] == 0 and path.read_text() == "original"


@pytest.mark.parametrize("operation", ["write", "edit", "patch"])
def test_successful_edit_reports_the_written_path(tmp_path, operation):
    path = tmp_path / "a.txt"
    path.write_text("original")
    context = ToolContext(cwd=tmp_path)
    if operation == "write":
        request = write_file({"path": "a.txt", "content": "updated"}, context)
    elif operation == "edit":
        request = edit_file({"path": "a.txt", "old_text": "original", "new_text": "updated"}, context)
    else:
        request = apply_patch({"operations": [
            {"type": "replace", "path": "a.txt", "old_text": "original", "new_text": "temporary"},
            {"type": "replace", "path": "a.txt", "old_text": "temporary", "new_text": "updated"},
        ]}, context)
    result = asyncio.run(request)

    assert not result.is_error and result.details["changed"] is True
    assert result.details["changed_paths"] == [str(path.resolve())]
    assert result.details["files_changed"] == 1 and path.read_text() == "updated"


def test_patch_reports_deleted_and_added_paths_in_operation_order(tmp_path):
    deleted = tmp_path / "z.txt"
    deleted.write_text("original")
    result = asyncio.run(apply_patch({"operations": [
        {"type": "delete", "path": "z.txt"},
        {"type": "add", "path": "a.txt", "content": "new"},
    ]}, ToolContext(cwd=tmp_path)))

    assert not result.is_error and result.details["changed"] is True
    assert result.details["changed_paths"] == [str(deleted.resolve()), str((tmp_path / "a.txt").resolve())]
    assert result.details["files_changed"] == 2
    assert not deleted.exists() and (tmp_path / "a.txt").read_text() == "new"
