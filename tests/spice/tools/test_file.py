from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spice.tools.base import ToolContext
from spice.tools.file import apply_patch, edit_file, read_file, read_files, write_file
from spice.tools.file_state import FileStateStore


class FileToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_files_reads_multiple_files_and_records_file_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            (root / "b.txt").write_text("bravo\ncharlie\n", encoding="utf-8")
            file_states = FileStateStore()

            result = await read_files(
                {
                    "files": [
                        {"path": "a.txt"},
                        {"path": "b.txt", "offset": 0, "limit": 5},
                    ]
                },
                ToolContext(cwd=root, file_states=file_states),
            )

            self.assertFalse(result.is_error)
            self.assertIn("--- a.txt", result.content)
            self.assertIn("alpha", result.content)
            self.assertIn("--- b.txt", result.content)
            self.assertIn("bravo", result.content)
            self.assertEqual(result.details["file_count"], 2)
            self.assertEqual(result.details["partial_count"], 1)
            edit_result = await edit_file({"path": "a.txt", "old_text": "alpha", "new_text": "aleph"}, ToolContext(cwd=root, file_states=file_states))
            self.assertFalse(edit_result.is_error)

    async def test_read_files_rejects_empty_files_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = await read_files({"files": []}, ToolContext(cwd=Path(tmp)))

            self.assertTrue(result.is_error)
            self.assertIn("files must be a non-empty array", result.content)

    async def test_edit_file_dry_run_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")
            result = await edit_file(
                {"path": "a.txt", "old_text": "two", "new_text": "three", "dry_run": True},
                ToolContext(cwd=root),
            )
            self.assertFalse(result.is_error)
            self.assertIn("three", result.content)
            self.assertEqual(target.read_text(encoding="utf-8"), "one\ntwo\n")

    async def test_edit_file_requires_occurrence_for_multiple_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("x\nx\n", encoding="utf-8")
            result = await edit_file({"path": "a.txt", "old_text": "x", "new_text": "y"}, ToolContext(cwd=root))
            self.assertTrue(result.is_error)
            self.assertIn("matched 2 times", result.content)

            result = await edit_file(
                {"path": "a.txt", "old_text": "x", "new_text": "y", "occurrence": 2},
                ToolContext(cwd=root),
            )
            self.assertFalse(result.is_error)
            self.assertEqual(target.read_text(encoding="utf-8"), "x\ny\n")

    async def test_edit_file_requires_prior_read_when_file_state_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")
            context = ToolContext(cwd=root, file_states=FileStateStore())

            result = await edit_file({"path": "a.txt", "old_text": "two", "new_text": "three"}, context)

            self.assertTrue(result.is_error)
            self.assertIn("read_file must be called first", result.content)
            self.assertEqual(result.details["presentation"], "guidance")

    async def test_write_file_requires_prior_read_before_overwrite_as_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("old\n", encoding="utf-8")
            context = ToolContext(cwd=root, file_states=FileStateStore())

            result = await write_file({"path": "a.txt", "content": "new\n"}, context)

            self.assertTrue(result.is_error)
            self.assertIn("read_file must be called first", result.content)
            self.assertEqual(result.details["category"], "file_state_guard")
            self.assertEqual(result.details["presentation"], "guidance")

    async def test_edit_file_rejects_stale_read_when_file_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")
            context = ToolContext(cwd=root, file_states=FileStateStore())

            read_result = await read_file({"path": "a.txt"}, context)
            self.assertFalse(read_result.is_error)
            target.write_text("one\nchanged\n", encoding="utf-8")

            result = await edit_file({"path": "a.txt", "old_text": "two", "new_text": "three"}, context)

            self.assertTrue(result.is_error)
            self.assertIn("file changed since it was last read", result.content)

    async def test_edit_file_partial_read_error_suggests_full_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\nthree\n", encoding="utf-8")
            context = ToolContext(cwd=root, file_states=FileStateStore())

            read_result = await read_file({"path": "a.txt", "limit": 4}, context)
            self.assertFalse(read_result.is_error)

            result = await edit_file({"path": "a.txt", "old_text": "two", "new_text": "2"}, context)

            self.assertTrue(result.is_error)
            self.assertIn("last read was partial", result.content)
            self.assertIn("until partial=false", result.content)

    async def test_edit_file_reports_near_match_when_old_text_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("def demo():\n    return 1\n", encoding="utf-8")

            result = await edit_file({"path": "a.txt", "old_text": "def demo():\n    return 2", "new_text": "x"}, ToolContext(cwd=root))

            self.assertTrue(result.is_error)
            self.assertIn("Closest nearby content", result.content)

    async def test_apply_patch_dry_run_validates_all_operations_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")
            context = ToolContext(cwd=root)

            result = await apply_patch(
                {
                    "operations": [
                        {"type": "replace", "path": "a.txt", "old_text": "two", "new_text": "three"},
                        {"type": "add", "path": "b.txt", "content": "new\n"},
                    ],
                    "dry_run": True,
                },
                context,
            )

            self.assertFalse(result.is_error)
            self.assertIn("three", result.content)
            self.assertEqual(result.details["lines_added"], 2)
            self.assertEqual(result.details["lines_removed"], 1)
            self.assertFalse((root / "b.txt").exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "one\ntwo\n")

    async def test_apply_patch_rejects_failed_operation_before_writing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")

            result = await apply_patch(
                {
                    "operations": [
                        {"type": "replace", "path": "a.txt", "old_text": "two", "new_text": "three"},
                        {"type": "replace", "path": "missing.txt", "old_text": "x", "new_text": "y"},
                    ]
                },
                ToolContext(cwd=root),
            )

            self.assertTrue(result.is_error)
            self.assertIn("Patch validation failed", result.content)
            self.assertEqual(target.read_text(encoding="utf-8"), "one\ntwo\n")

    async def test_apply_patch_applies_multiple_operations_to_same_file_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")

            result = await apply_patch(
                {
                    "operations": [
                        {"type": "replace", "path": "a.txt", "old_text": "one", "new_text": "1"},
                        {"type": "replace", "path": "a.txt", "old_text": "two", "new_text": "2"},
                    ]
                },
                ToolContext(cwd=root),
            )

            self.assertFalse(result.is_error)
            self.assertEqual(target.read_text(encoding="utf-8"), "1\n2\n")

    async def test_apply_patch_accepts_old_str_new_str_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")

            result = await apply_patch(
                {
                    "operations": [
                        {"type": "replace", "path": "a.txt", "old_str": "two", "new_str": "three"},
                    ]
                },
                ToolContext(cwd=root),
            )

            self.assertFalse(result.is_error)
            self.assertEqual(target.read_text(encoding="utf-8"), "one\nthree\n")

    async def test_apply_patch_rejects_conflicting_text_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("one\ntwo\n", encoding="utf-8")

            result = await apply_patch(
                {
                    "operations": [
                        {
                            "type": "replace",
                            "path": "a.txt",
                            "old_text": "two",
                            "old_str": "one",
                            "new_text": "three",
                        },
                    ]
                },
                ToolContext(cwd=root),
            )

            self.assertTrue(result.is_error)
            self.assertIn("old_text and old_str both provided with different values", result.content)
