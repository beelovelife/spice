from __future__ import annotations

from pathlib import Path

import pytest

from spice.sandbox.policy import WorkspacePolicy


def test_workspace_policy_blocks_parent_escape(tmp_path: Path) -> None:
    policy = WorkspacePolicy.from_settings({}, cwd=tmp_path)

    with pytest.raises(PermissionError, match="outside workspace"):
        policy.resolve_read("../outside.txt")


def test_workspace_policy_blocks_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        policy = WorkspacePolicy.from_settings({}, cwd=tmp_path)

        with pytest.raises(PermissionError, match="outside workspace"):
            policy.resolve_read("link/secret.txt")
    finally:
        outside.rmdir()


def test_workspace_policy_blocks_protected_writes(tmp_path: Path) -> None:
    policy = WorkspacePolicy.from_settings({}, cwd=tmp_path)

    with pytest.raises(PermissionError, match="protected"):
        policy.resolve_write(".spice/settings.json", content_size=2)

    with pytest.raises(PermissionError, match="protected"):
        policy.resolve_write(".git/config", content_size=2)


def test_workspace_policy_blocks_secret_reads_and_writes(tmp_path: Path) -> None:
    policy = WorkspacePolicy.from_settings({}, cwd=tmp_path)

    with pytest.raises(PermissionError, match="secret policy"):
        policy.resolve_read(".env")

    with pytest.raises(PermissionError, match="secret policy"):
        policy.resolve_write("key.pem", content_size=2)


def test_workspace_policy_enforces_write_size(tmp_path: Path) -> None:
    policy = WorkspacePolicy.from_settings({"max_write_bytes": 3}, cwd=tmp_path)

    with pytest.raises(PermissionError, match="Write exceeds"):
        policy.resolve_write("file.txt", content_size=4)

