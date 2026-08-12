from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import spice.agent.agent_session as session_module
from spice.agent.events import AgentErrorEvent, RoundCompleteEvent, TextDeltaEvent, TurnEndEvent
from spice.agent.agent_session import AgentSession
from spice.agent.sessions import SessionStore, message_from_dict, message_to_dict, workspace_key
from spice.agent.tool_results import build_tool_result_metadata, prepare_tool_message_for_session
from spice.llm.config import SpiceConfig
from spice.llm.messages import Message, ToolCall
from spice.llm.models import Model
from spice.storage.sqlite import connect_sqlite, init_sqlite_database
from spice.storage.sqlite_sessions import SqliteSessionStore
from spice.tools.base import ToolContext, tool_result


async def _drain(events) -> None:
    async for _event in events:
        pass


class SessionStoreTests(unittest.TestCase):
    def test_workspace_scoped_store_uses_workspace_key_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "sessions"
            workspace = Path(directory) / "my project"
            workspace.mkdir()
            store = SessionStore(base, cwd=workspace)
            info = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")

            self.assertTrue(workspace_key(workspace).startswith("workspace-"))
            self.assertEqual(store.workspace_key, workspace_key(workspace))
            self.assertEqual(info.path.parent, base / workspace_key(workspace))
            self.assertTrue(info.path.exists())

    def test_global_store_lists_workspace_scoped_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "sessions"
            first_cwd = Path(directory) / "one"
            second_cwd = Path(directory) / "two"
            first_cwd.mkdir()
            second_cwd.mkdir()
            first_store = SessionStore(base, cwd=first_cwd)
            second_store = SessionStore(base, cwd=second_cwd)
            first = first_store.create(cwd=first_cwd, provider="openai", model="gpt-4o-mini")
            second = second_store.create(cwd=second_cwd, provider="gemini", model="gemini-2.5-flash")
            first_store.append_message(first.id, Message(role="user", content="first"))
            second_store.append_message(second.id, Message(role="user", content="second"))

            global_store = SessionStore(base)

            self.assertEqual(global_store.latest(cwd=first_cwd).id, first.id)
            self.assertEqual(global_store.list(cwd=second_cwd)[0].id, second.id)
            self.assertEqual({item.id for item in global_store.list(cwd=None)}, {first.id, second.id})

    def test_list_supports_limit_and_empty_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "sessions"
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            store = SessionStore(base, cwd=workspace)
            empty = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            older = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            newer = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            store.append_message(older.id, Message(role="user", content="older"))
            store.append_message(newer.id, Message(role="user", content="newer"))

            self.assertEqual([item.id for item in store.list(cwd=workspace, limit=1)], [newer.id])
            self.assertNotIn(empty.id, {item.id for item in store.list(cwd=workspace)})
            self.assertIn(empty.id, {item.id for item in store.list(cwd=workspace, include_empty=True)})

    def test_session_summary_methods_avoid_redundant_reads(self) -> None:
        class CountingSessionStore(SessionStore):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.read_count = 0

            def _read(self, path):
                self.read_count += 1
                return super()._read(path)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "sessions"
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            store = CountingSessionStore(base, cwd=workspace)
            first = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            second = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            store.append_message(first.id, Message(role="user", content="first"))
            store.append_message(second.id, Message(role="user", content="second"))

            store.read_count = 0
            rows = store.list(cwd=workspace, include_empty=True)
            self.assertEqual({item.id for item in rows}, {first.id, second.id})
            self.assertEqual(store.read_count, 2)

            store.read_count = 0
            self.assertEqual(store.info(first.id).id, first.id)
            self.assertEqual(store.read_count, 1)

            store.read_count = 0
            self.assertEqual([entry.data["message"]["content"] for entry in store.path_entries(first.id)], ["first"])
            self.assertEqual(store.read_count, 1)

    def test_create_append_and_load_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content="hello"))
            store.append_message(
                info.id,
                Message(
                    role="assistant",
                    content="done",
                    tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "README.md"})],
                ),
            )

            messages = store.load_messages(info.id)

            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[0].content, "hello")
            self.assertEqual(messages[1].tool_calls[0].name, "read_file")

    def test_sqlite_store_create_append_and_load_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            store = SqliteSessionStore(Path(directory) / "spice.db", cwd=workspace)
            info = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content="hello"))
            store.append_message(
                info.id,
                Message(
                    role="assistant",
                    content="done",
                    tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "README.md"})],
                ),
            )

            messages = store.load_messages(info.id)
            listed = store.list(cwd=workspace)

            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[0].content, "hello")
            self.assertEqual(messages[1].tool_calls[0].name, "read_file")
            self.assertEqual([item.id for item in listed], [info.id])
            self.assertEqual(store.latest(cwd=workspace).id, info.id)

    def test_sqlite_store_custom_entries_follow_branch_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteSessionStore(Path(directory) / "spice.db")
            info = store.create(cwd=Path(directory), provider="openai", model="gpt-4o-mini")
            root_id = store.append_message(info.id, Message(role="user", content="root"))
            custom_id = store.append_custom(info.id, {"customType": "plan_state", "mode": "plan"}, parent_id=root_id)
            other_id = store.append_message(info.id, Message(role="assistant", content="other"), parent_id=root_id)

            custom_path = store.path_entries(info.id, leaf_id=custom_id)
            other_path = store.path_entries(info.id, leaf_id=other_id)

            self.assertEqual([entry.id for entry in custom_path], [root_id, custom_id])
            self.assertEqual([entry.id for entry in other_path], [root_id, other_id])

    def test_sqlite_store_reset_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteSessionStore(Path(directory) / "spice.db")
            info = store.create(cwd=Path(directory), provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content="hello"))

            reset = store.reset(info.id)
            self.assertIsNone(reset.leaf_id)
            self.assertEqual(reset.message_count, 0)

            store.delete(info.id)
            with self.assertRaises(ValueError):
                store.info(info.id)

    def test_sqlite_init_creates_schema_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "spice.db"

            init_sqlite_database(db_path)

            with connect_sqlite(db_path) as conn:
                tables = {row["name"] for row in conn.execute("select name from sqlite_master where type = 'table'")}
                indexes = {row["name"] for row in conn.execute("select name from sqlite_master where type = 'index'")}
            self.assertIn("sessions", tables)
            self.assertIn("session_entries", tables)
            self.assertIn("long_tasks", tables)
            self.assertIn("memory_history", tables)
            self.assertIn("artifacts", tables)
            self.assertIn("session_entries_session_ordinal", indexes)

    def test_message_metadata_round_trips(self) -> None:
        message = Message(role="tool", content="preview", name="read_file", metadata={"tool_result": {"display": "Read file"}})

        loaded = message_from_dict(message_to_dict(message))

        self.assertEqual(loaded.metadata["tool_result"]["display"], "Read file")

    def test_prepare_tool_message_for_session_persists_large_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            result = tool_result(
                "x" * 17_000,
                {"path": str(cwd / "large.txt"), "line_count": 1, "char_count": 17_000, "total_chars": 17_000},
            )
            message = Message(
                role="tool",
                content=result.content,
                tool_call_id="tc1",
                name="read_file",
                metadata=build_tool_result_metadata("read_file", {"path": "large.txt"}, result),
            )

            persisted = prepare_tool_message_for_session(message, cwd=cwd, session_id="session-1")

            tool_meta = persisted.metadata["tool_result"]
            self.assertIn("[tool output truncated]", persisted.content)
            self.assertIn("kept first 6000 and last 6000", persisted.content)
            self.assertTrue(persisted.content.startswith("x" * 100))
            self.assertTrue(persisted.content.endswith("Full output saved to " + str(Path(tool_meta["artifact_path"]))))
            self.assertTrue(tool_meta["truncated"])
            self.assertEqual(tool_meta["original_chars"], 17_000)
            artifact_path = Path(tool_meta["artifact_path"])
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact_path.read_text(encoding="utf-8"), result.content)
            self.assertLess(len(persisted.content), len(result.content))

    def test_prepare_tool_message_uses_transient_full_output_for_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            full = "head" + "x" * 13_000 + "tail"
            result = tool_result("head [truncated] tail", full_content=full)
            metadata = build_tool_result_metadata("bash", {"command": "test"}, result)
            metadata["_full_tool_output"] = full
            message = Message(role="tool", content=result.content, tool_call_id="tc1", name="bash", metadata=metadata)

            persisted = prepare_tool_message_for_session(message, cwd=cwd, session_id="session-1")

            tool_meta = persisted.metadata["tool_result"]
            assert "_full_tool_output" not in persisted.metadata
            assert Path(tool_meta["artifact_path"]).read_text(encoding="utf-8") == full
            assert persisted.content.startswith("head")
            assert "tail" in persisted.content

    def test_build_context_replaces_older_tool_results_with_context_stub(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            for index in range(4):
                call_id = f"tc{index}"
                store.append_message(
                    info.id,
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(id=call_id, name="read_file", arguments={"path": f"file{index}.py"})],
                    ),
                )
                result = tool_result(
                    f"preview {index}",
                    {"path": f"file{index}.py", "line_count": 1, "char_count": 9, "total_chars": 9},
                )
                store.append_message(
                    info.id,
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=call_id,
                        name="read_file",
                        metadata=build_tool_result_metadata("read_file", {"path": f"file{index}.py"}, result),
                    ),
                )

            messages = store.load_messages(info.id)
            tool_messages = [message for message in messages if message.role == "tool"]

            self.assertIn("omitted from older context", tool_messages[0].content)
            self.assertIn("file0.py", tool_messages[0].content)
            self.assertEqual(tool_messages[-1].content, "preview 3")

    def test_append_custom_entries_follow_branch_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            root_id = store.append_message(info.id, Message(role="user", content="root"))
            custom_id = store.append_custom(info.id, {"customType": "plan_state", "mode": "plan"}, parent_id=root_id)
            other_id = store.append_message(info.id, Message(role="assistant", content="other"), parent_id=root_id)

            custom_path = store.path_entries(info.id, leaf_id=custom_id)
            other_path = store.path_entries(info.id, leaf_id=other_id)

            self.assertEqual([entry.id for entry in custom_path], [root_id, custom_id])
            self.assertEqual([entry.id for entry in other_path], [root_id, other_id])

    def test_list_and_latest_are_scoped_to_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            first_cwd = Path(directory) / "one"
            second_cwd = Path(directory) / "two"
            first = store.create(cwd=first_cwd, provider="openai", model="gpt-4o-mini")
            second = store.create(cwd=second_cwd, provider="gemini", model="gemini-2.5-flash")
            store.append_message(first.id, Message(role="user", content="first"))
            store.append_message(second.id, Message(role="user", content="second"))

            self.assertEqual(store.latest(cwd=first_cwd).id, first.id)
            self.assertEqual(store.list(cwd=second_cwd)[0].id, second.id)

    def test_append_message_writes_parent_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")

            first_id = store.append_message(info.id, Message(role="user", content="first"))
            second_id = store.append_message(info.id, Message(role="assistant", content="second"), parent_id=first_id)

            entries = store.entries(info.id)

            self.assertEqual(entries[0].id, first_id)
            self.assertIsNone(entries[0].parent_id)
            self.assertEqual(entries[1].id, second_id)
            self.assertEqual(entries[1].parent_id, first_id)

    def test_build_context_from_specific_leaf_uses_branch_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            root_id = store.append_message(info.id, Message(role="user", content="root"))
            left_id = store.append_message(info.id, Message(role="assistant", content="left"), parent_id=root_id)
            right_id = store.append_message(info.id, Message(role="assistant", content="right"), parent_id=root_id)

            left_context = store.build_context(info.id, leaf_id=left_id)
            right_context = store.build_context(info.id, leaf_id=right_id)

            self.assertEqual([message.content for message in left_context.messages], ["root", "left"])
            self.assertEqual([message.content for message in right_context.messages], ["root", "right"])

    def test_set_leaf_rewinds_active_path_and_next_append_branches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            root_id = store.append_message(info.id, Message(role="user", content="root"))
            old_id = store.append_message(info.id, Message(role="assistant", content="old"), parent_id=root_id)

            marker_id = store.set_leaf(info.id, root_id)
            new_id = store.append_message(info.id, Message(role="assistant", content="new"))

            self.assertEqual(store.info(info.id).leaf_id, new_id)
            self.assertEqual([message.content for message in store.load_messages(info.id)], ["root", "new"])
            entries = {entry.id: entry for entry in store.entries(info.id)}
            self.assertEqual(entries[new_id].parent_id, root_id)
            self.assertEqual(entries[marker_id].type, "leaf")
            self.assertIn(old_id, entries)

    def test_resumed_todo_state_only_reads_the_active_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions", cwd=root)
            info = store.create(cwd=root, provider="openai", model="gpt-5.1")
            root_id = store.append_message(info.id, Message(role="user", content="root"))
            store.append_custom(
                info.id,
                {
                    "customType": "todo_state",
                    "items": [{"id": "old", "content": "discarded branch", "status": "pending"}],
                },
                parent_id=root_id,
            )
            store.set_leaf(info.id, root_id)

            resumed = AgentSession(cwd=root, session_id=info.id, session_store=store)

            self.assertEqual(resumed.todo_state.read(), [])

    def test_compaction_context_skips_entries_before_first_kept_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content='{"partial": false, "kept": "old"}'))
            kept_id = store.append_message(info.id, Message(role="assistant", content="kept"))
            compact_id = store.append_compaction(
                info.id,
                summary="Old JSON was summarized without preserving raw content.",
                first_kept_entry_id=kept_id,
                tokens_before=100,
            )
            new_id = store.append_message(info.id, Message(role="user", content="new"), parent_id=compact_id)

            messages = store.load_messages(info.id)
            entries = {entry.id: entry for entry in store.entries(info.id)}

            self.assertEqual(entries[compact_id].parent_id, kept_id)
            self.assertEqual(entries[new_id].parent_id, compact_id)
            self.assertEqual(messages[0].role, "system")
            self.assertIn("Old JSON was summarized", messages[0].content)
            self.assertEqual([message.content for message in messages[1:]], ["kept", "new"])
            self.assertNotIn("partial", "\n".join(message.content for message in messages))

    def test_read_skips_corrupt_jsonl_entries_after_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            good_id = store.append_message(info.id, Message(role="user", content="good"))
            path = store.path_for(info.id)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("{not json}\n")

            with self.assertLogs("spice.agent.sessions", level="WARNING") as logs:
                second_id = store.append_message(info.id, Message(role="assistant", content="still good"))

            self.assertEqual([entry.id for entry in store.entries(info.id)], [good_id, second_id])
            self.assertIn("Skipping invalid session JSONL row", "\n".join(logs.output))
            self.assertIn(str(path), "\n".join(logs.output))

    def test_session_info_current_model_comes_from_header_and_model_changes_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store.append_message(
                info.id,
                Message(role="assistant", content="old answer", provider="anthropic", model="claude-sonnet-4-5"),
            )

            initial_info = store.info(info.id)

            self.assertEqual(initial_info.provider, "openai")
            self.assertEqual(initial_info.model, "gpt-4o-mini")

            store.append_model_change(info.id, provider="gemini", model="gemini-2.5-flash")
            changed_info = store.info(info.id)

            self.assertEqual(changed_info.provider, "gemini")
            self.assertEqual(changed_info.model, "gemini-2.5-flash")

    def test_resolve_supports_id_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")

            resolved = store.resolve(info.id[:6], cwd=Path.cwd())

            self.assertEqual(resolved.id, info.id)

    def test_resolve_exact_id_uses_direct_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")

            original_list = store.list

            def fail_list(*args, **kwargs):
                raise AssertionError("exact id resolution should not scan session list")

            store.list = fail_list
            try:
                resolved = store.resolve(info.id, cwd=Path.cwd())
            finally:
                store.list = original_list

            self.assertEqual(resolved.id, info.id)

    def test_old_linear_entries_are_loaded_as_a_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store._append_json(
                store.path_for(info.id),
                {
                    "type": "message",
                    "id": "legacy1",
                    "timestamp": "now",
                    "message": {"role": "user", "content": "legacy root"},
                },
            )
            store._append_json(
                store.path_for(info.id),
                {
                    "type": "message",
                    "id": "legacy2",
                    "timestamp": "now",
                    "message": {"role": "assistant", "content": "legacy child"},
                },
            )

            entries = store.entries(info.id)
            messages = store.load_messages(info.id)

            self.assertIsNone(entries[0].parent_id)
            self.assertEqual(entries[1].parent_id, "legacy1")
            self.assertEqual([message.content for message in messages], ["legacy root", "legacy child"])

    def test_list_omits_empty_header_only_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            empty = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            active = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store.append_message(active.id, Message(role="user", content="hello"))

            rows = store.list(cwd=Path.cwd())

            self.assertEqual([row.id for row in rows], [active.id])
            self.assertEqual(store.info(empty.id).message_count, 0)

    def test_reset_keeps_session_id_and_removes_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content="hello"))
            store.append_message(info.id, Message(role="assistant", content="hi"))

            reset_info = store.reset(info.id)

            self.assertEqual(reset_info.id, info.id)
            self.assertEqual(reset_info.message_count, 0)
            self.assertIsNone(reset_info.leaf_id)
            self.assertEqual(store.load_messages(info.id), [])
            self.assertEqual(len(store.path_for(info.id).read_text(encoding="utf-8").splitlines()), 1)

    def test_delete_removes_session_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")

            store.delete(info.id)

            self.assertFalse(store.path_for(info.id).exists())
            with self.assertRaises(ValueError):
                store.info(info.id)


class AgentSessionPersistenceTests(unittest.TestCase):
    def test_round_complete_persists_before_prompt_finishes(self) -> None:
        async def fake_run_turn(**kwargs):
            kwargs["messages"].extend(
                [
                    Message(role="user", content=kwargs["prompt"]),
                    Message(role="assistant", content="", tool_calls=[ToolCall("tc1", "demo", {})]),
                    Message(role="tool", content="ok", tool_call_id="tc1", name="demo"),
                ]
            )
            yield RoundCompleteEvent(1)
            await asyncio.Event().wait()

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                events = session.prompt("hello")
                await anext(events)  # AgentStartEvent
                event = await anext(events)
                persisted_roles = [entry.data["message"]["role"] for entry in store.path_entries(session.session_id) if entry.type == "message"]
                await events.aclose()
                return event, persisted_roles


        event, roles = asyncio.run(run())

        self.assertIsInstance(event, RoundCompleteEvent)
        self.assertEqual(roles, ["user", "assistant", "tool"])

    def test_cancellation_persists_interrupted_tool_result(self) -> None:
        async def fake_run_turn(**kwargs):
            kwargs["messages"].extend(
                [
                    Message(role="user", content=kwargs["prompt"]),
                    Message(role="assistant", content="", tool_calls=[ToolCall("tc1", "bash", {"command": "sleep 10"})]),
                ]
            )
            await asyncio.Event().wait()
            if False:
                yield TurnEndEvent("")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)

                async def consume():
                    return [event async for event in session.prompt("run")]

                task = asyncio.create_task(consume())
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                task.cancel()
                events = await task
                messages = store.load_messages(session.session_id)
                return events, messages


        events, messages = asyncio.run(run())
        interrupted = next(message for message in messages if message.role == "tool")

        self.assertTrue(any(isinstance(event, AgentErrorEvent) and event.kind == "user_interrupted" for event in events))
        self.assertTrue(interrupted.is_error)
        self.assertEqual(interrupted.metadata["tool_result"]["error_code"], "user_interrupted")
    def setUp(self) -> None:
        self.original_run_turn = session_module.run_turn
        self.original_generate_summary = session_module.generate_summary

    def tearDown(self) -> None:
        session_module.run_turn = self.original_run_turn
        session_module.generate_summary = self.original_generate_summary

    def test_resume_loads_messages_after_fresh_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content="remember this"))

            session = AgentSession(cwd=Path.cwd(), session_id=info.id, session_store=store)

            self.assertEqual(session.session_id, info.id)
            self.assertEqual(session.messages[0].role, "system")
            self.assertEqual(session.messages[1].content, "remember this")

    def test_resume_and_reset_rebuild_system_prompt_with_current_spice_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            workspace.mkdir()
            spice_file = workspace / "SPICE.md"
            spice_file.write_text("Initial project rule.\n", encoding="utf-8")
            store = SessionStore(base / "sessions", cwd=workspace)
            info = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
            store.append_message(info.id, Message(role="user", content="remember this"))

            session = AgentSession(cwd=workspace, session_id=info.id, session_store=store)
            self.assertIn("Initial project rule.", session.messages[0].content)

            spice_file.write_text("Updated project rule.\n", encoding="utf-8")
            session.reset()

            self.assertIn("Updated project rule.", session.messages[0].content)
            self.assertNotIn("Initial project rule.", session.messages[0].content)

    def test_resume_uses_current_config_model_not_session_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            info = store.create(cwd=Path.cwd(), provider="openai", model="gpt-4o-mini")

            original_load_config = session_module.load_config
            session_module.load_config = lambda: SpiceConfig(provider="openai", model="gpt-5.1")
            try:
                session = AgentSession(cwd=Path.cwd(), session_id=info.id, session_store=store)
            finally:
                session_module.load_config = original_load_config

            self.assertEqual(session.session_id, info.id)
            self.assertEqual(session.model.provider, "openai")
            self.assertEqual(session.model.id, "gpt-5.1")

    def test_set_model_appends_model_change_to_session(self) -> None:
        async def fake_run_turn(**kwargs):
            messages = kwargs["messages"]
            messages.append(Message(role="user", content=kwargs["prompt"]))
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn


        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = AgentSession(cwd=Path.cwd(), session_store=store)
            asyncio.run(_drain(session.prompt("hello")))

            session.set_model(Model(id="gpt-4o", provider="openai"))
            info = store.info(session.session_id)

            self.assertEqual(session.model.id, "gpt-4o")
            self.assertEqual(info.model, "gpt-4o")

    def test_new_session_is_not_persisted_until_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = AgentSession(cwd=Path.cwd(), session_store=store)

            session.set_model(Model(id="gpt-4o", provider="openai"))

            self.assertEqual(session.session_label, "new")
            self.assertEqual(store.list(cwd=Path.cwd()), [])

    def test_prompt_fans_out_events_to_multiple_consumers_and_persists(self) -> None:
        async def fake_run_turn(**kwargs):
            messages = kwargs["messages"]
            messages.append(Message(role="user", content=kwargs["prompt"]))
            messages.append(Message(role="assistant", content="done"))
            yield TextDeltaEvent("done")
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = SessionStore(Path(directory))
                original_load_config = session_module.load_config
                session_module.load_config = lambda: SpiceConfig(provider="openai", model="gpt-5.1")
                try:
                    session = AgentSession(cwd=Path.cwd(), session_store=store)
                finally:
                    session_module.load_config = original_load_config
                first_seen = []
                second_seen = []

                session.subscribe(lambda event: first_seen.append(type(event).__name__))
                session.subscribe(lambda event: second_seen.append(type(event).__name__))

                yielded = [type(event).__name__ async for event in session.prompt("hello")]
                persisted = store.load_messages(session.session_id)
                return yielded, first_seen, second_seen, persisted


        yielded, first_seen, second_seen, persisted = asyncio.run(run())

        self.assertEqual(yielded[0], "AgentStartEvent")
        self.assertEqual(yielded[-1], "AgentEndEvent")
        self.assertEqual(first_seen, yielded)
        self.assertEqual(second_seen, yielded)
        self.assertEqual([message.role for message in persisted], ["user", "assistant"])
        self.assertEqual(persisted[1].provider, "openai")
        self.assertEqual(persisted[1].model, "gpt-5.1")

    def test_prompt_expands_file_references_before_running_turn_and_persisting(self) -> None:
        seen_prompts = []

        async def fake_run_turn(**kwargs):
            seen_prompts.append(kwargs["prompt"])
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "README.md").write_text("# Project\n", encoding="utf-8")
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)

                yielded = [type(event).__name__ async for event in session.prompt("@README.md summarize")]
                persisted = store.load_messages(session.session_id)
                return yielded, persisted


        yielded, persisted = asyncio.run(run())

        self.assertIn("TurnEndEvent", yielded)
        self.assertIn("Referenced files:", seen_prompts[0])
        self.assertIn("# Project", seen_prompts[0])
        self.assertIn("Referenced files:", persisted[0].content)

    def test_prompt_rejects_image_reference_when_model_lacks_vision(self) -> None:
        called = False

        async def fake_run_turn(**kwargs):
            nonlocal called
            called = True
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "ui.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                events = [event async for event in session.prompt("@ui.png analyze")]
                return events, session


        events, session = asyncio.run(run())

        self.assertEqual([type(event).__name__ for event in events], ["AgentErrorEvent"])
        self.assertIn("does not support image input", events[0].message)
        self.assertFalse(called)
        self.assertEqual(session.session_label, "new")

    def test_plan_mode_uses_read_only_tools_and_restores_from_session(self) -> None:
        seen_tools = []

        async def fake_run_turn(**kwargs):
            seen_tools.append([tool.name for tool in kwargs["tools"]])
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            kwargs["messages"].append(Message(role="assistant", content="Plan:\n1. Read files\n2. Explain changes"))
            yield TurnEndEvent("Plan:\n1. Read files\n2. Explain changes")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                session.start_plan("inspect project")
                await _drain(session.prompt("inspect project"))
                resumed = AgentSession(cwd=root, session_id=session.session_id, session_store=store)
                entries = session.session_store.path_entries(session.session_id)
                return session, resumed, entries


        session, resumed, entries = asyncio.run(run())

        self.assertNotIn("write_file", seen_tools[0])
        self.assertIn("read_file", seen_tools[0])
        self.assertEqual(session.plan_state.steps, ["Read files", "Explain changes"])
        self.assertEqual(resumed.plan_state.mode, "edit")
        self.assertEqual(resumed.plan_state.objective, "inspect project")
        self.assertEqual(resumed.plan_state.steps, ["Read files", "Explain changes"])

        plan_index = max(index for index, entry in enumerate(entries) if entry.type == "custom" and entry.data.get("customType") == "plan_state")
        assistant_index = next(index for index, entry in enumerate(entries) if entry.type == "message" and entry.data.get("message", {}).get("role") == "assistant")
        self.assertGreater(plan_index, assistant_index)

    def test_update_todo_custom_entry_is_persisted_after_tool_messages(self) -> None:
        async def fake_run_turn(**kwargs):
            update_todo = next(tool for tool in kwargs["tools"] if tool.name == "update_todo")
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            kwargs["messages"].append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="update_todo",
                            arguments={
                                "todos": [
                                    {"id": "1", "content": "Inspect state", "status": "in_progress"},
                                ]
                            },
                        )
                    ],
                )
            )
            result = await update_todo.execute(
                {
                    "todos": [
                        {"id": "1", "content": "Inspect state", "status": "in_progress"},
                    ]
                },
                ToolContext(cwd=kwargs["cwd"]),
            )
            kwargs["messages"].append(
                Message(role="tool", content=result.content, tool_call_id="tc1", name="update_todo", is_error=result.is_error)
            )
            kwargs["messages"].append(Message(role="assistant", content="done"))
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                await _drain(session.prompt("complex analysis"))
                entries = session.session_store.path_entries(session.session_id)
                leaf_id = session.session_store.info(session.session_id).leaf_id
                return entries, leaf_id


        entries, leaf_id = asyncio.run(run())
        todo_index = next(index for index, entry in enumerate(entries) if entry.type == "custom" and entry.data.get("customType") == "todo_state")
        tool_index = next(index for index, entry in enumerate(entries) if entry.type == "message" and entry.data.get("message", {}).get("role") == "tool")
        final_assistant_index = max(
            index
            for index, entry in enumerate(entries)
            if entry.type == "message" and entry.data.get("message", {}).get("role") == "assistant"
        )

        self.assertGreater(todo_index, tool_index)
        self.assertGreater(todo_index, final_assistant_index)
        self.assertEqual(leaf_id, entries[todo_index].id)

    def test_runtime_context_is_not_persisted_as_user_message(self) -> None:
        seen_runtime_contexts = []

        async def fake_run_turn(**kwargs):
            seen_runtime_contexts.append(kwargs.get("runtime_context"))
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            kwargs["messages"].append(Message(role="assistant", content="done"))
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                session.start_long_task("ship the feature")
                await _drain(session.prompt("continue"))
                entries = session.session_store.path_entries(session.session_id)
                user_messages = [
                    entry.data["message"]["content"]
                    for entry in entries
                    if entry.type == "message" and entry.data.get("message", {}).get("role") == "user"
                ]
                return user_messages


        user_messages = asyncio.run(run())

        self.assertEqual(user_messages, ["continue"])
        self.assertTrue(seen_runtime_contexts)
        self.assertIn("Active sustained goal", seen_runtime_contexts[0])
        self.assertNotIn("Active sustained goal", user_messages[0])

    def test_deferred_custom_state_is_discarded_when_message_persist_fails(self) -> None:
        async def fake_run_turn(**kwargs):
            update_todo = next(tool for tool in kwargs["tools"] if tool.name == "update_todo")
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            result = await update_todo.execute(
                {
                    "todos": [
                        {"id": "1", "content": "Inspect state", "status": "in_progress"},
                    ]
                },
                ToolContext(cwd=kwargs["cwd"]),
            )
            kwargs["messages"].append(Message(role="tool", content=result.content, tool_call_id="tc1", name="update_todo"))
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)

                def fail_persist_messages(_index: int) -> str | None:
                    raise OSError("disk full")

                session._persist_messages_from = fail_persist_messages  # type: ignore[method-assign]
                try:
                    await _drain(session.prompt("complex analysis"))
                except OSError as exc:
                    return session, str(exc)
                raise AssertionError("Expected message persistence to fail")


        session, error = asyncio.run(run())

        self.assertIn("disk full", error)
        self.assertFalse(session._defer_custom_state)
        self.assertFalse(session._dirty_plan_state)
        self.assertFalse(session._dirty_todo_state)

    def test_update_todo_tool_is_edit_only_and_persists_across_resume(self) -> None:
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                edit_tools = session.get_active_tools()
                tool = next(tool for tool in session._edit_tools if tool.name == "update_todo")
                await tool.execute(
                    {
                        "todos": [
                            {"id": "1", "content": "Read files", "status": "completed"},
                            {"id": "2", "content": "Run tests", "status": "in_progress"},
                        ]
                    },
                    ToolContext(cwd=root),
                )
                session.start_plan("inspect only")
                plan_tools = session.get_active_tools()
                resumed = AgentSession(cwd=root, session_id=session.session_id, session_store=store)
                return edit_tools, plan_tools, resumed


        edit_tools, plan_tools, resumed = asyncio.run(run())

        self.assertIn("update_todo", edit_tools)
        self.assertNotIn("update_todo", plan_tools)
        self.assertEqual(resumed.todo_state.read()[1]["content"], "Run tests")
        self.assertNotIn("Current todo list:", resumed.messages[0].content)
        self.assertNotIn("Run tests", resumed.messages[0].content)

    def test_approve_plan_initializes_todo_from_plan_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions", cwd=root)
            session = AgentSession(cwd=root, session_store=store)
            session.start_plan("inspect project")
            session.plan_state.steps = ["Read files", "Explain changes"]

            prompt = session.approve_plan("manual")

            self.assertIn("Execute the approved plan", prompt)
            self.assertEqual(
                session.todo_state.read(),
                [
                    {"id": "1", "content": "Read files", "status": "pending"},
                    {"id": "2", "content": "Explain changes", "status": "pending"},
                ],
            )
            self.assertNotIn("Current todo list:", session.messages[0].content)

    def test_approve_plan_without_steps_leaves_todo_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions", cwd=root)
            session = AgentSession(cwd=root, session_store=store)
            session.start_plan("inspect project")

            session.approve_plan("manual")

            self.assertEqual(session.todo_state.read(), [])

    def test_compaction_reloads_todo_state_without_system_injection(self) -> None:
        async def fake_generate_summary(**kwargs):
            return "Old work summarized."

        async def fake_run_turn(**kwargs):
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            text = "done " * 10000
            kwargs["messages"].append(Message(role="assistant", content=text))
            yield TurnEndEvent(text)

        session_module.generate_summary = fake_generate_summary
        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                tool = next(tool for tool in session._edit_tools if tool.name == "update_todo")
                await tool.execute(
                    {"todos": [{"id": "1", "content": "Keep going", "status": "in_progress"}]},
                    ToolContext(cwd=root),
                )
                for index in range(4):
                    await _drain(session.prompt(f"turn {index}"))
                session.session_store.append_custom(
                    session.session_id,
                    {
                        "customType": "todo_state",
                        "items": [{"id": "1", "content": "Disk latest todo", "status": "in_progress"}],
                    },
                )
                await session.compact(force=True)
                return session


        session = asyncio.run(run())

        self.assertIn("Previous conversation summary:", session.messages[1].content)
        self.assertEqual(session.todo_state.read()[0]["content"], "Disk latest todo")
        self.assertNotIn("Current todo list:", session.messages[0].content)
        self.assertNotIn("Disk latest todo", session.messages[0].content)
        self.assertNotIn("Keep going", session.messages[0].content)

    def test_prompt_auto_compacts_before_running_turn(self) -> None:
        seen_message_counts = []
        seen_system_prompts = []

        async def fake_generate_summary(**kwargs):
            return "Summarized old context."

        async def fake_run_turn(**kwargs):
            messages = kwargs["messages"]
            seen_message_counts.append(len(messages))
            seen_system_prompts.append(messages[0].content)
            messages.append(Message(role="user", content=kwargs["prompt"]))
            messages.append(Message(role="assistant", content="done"))
            yield TurnEndEvent("done")

        session_module.generate_summary = fake_generate_summary
        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                workspace = base / "workspace"
                workspace.mkdir()
                (workspace / "SPICE.md").write_text("Compact-visible project rule.\n", encoding="utf-8")
                store = SessionStore(base / "sessions", cwd=workspace)
                info = store.create(cwd=workspace, provider="openai", model="gpt-4o-mini")
                for index in range(12):
                    store.append_message(info.id, Message(role="user", content=f"old {index} " + ("x" * 10000)))
                session = AgentSession(cwd=workspace, session_id=info.id, session_store=store)
                session.model = Model(id="tiny", provider="openai", context_window=10_000, output_tokens=500)

                yielded = [type(event).__name__ async for event in session.prompt("new prompt")]
                entries = store.entries(info.id)
                loaded = store.load_messages(info.id)
                return yielded, entries, loaded


        yielded, entries, loaded = asyncio.run(run())

        self.assertIn("TurnEndEvent", yielded)
        self.assertTrue(any(entry.type == "compaction" for entry in entries))
        self.assertIn("Summarized old context.", loaded[0].content)
        self.assertLess(seen_message_counts[0], 14)
        self.assertIn("Compact-visible project rule.", seen_system_prompts[0])

    def test_listener_errors_are_isolated(self) -> None:
        async def fake_run_turn(**kwargs):
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = SessionStore(Path(directory))
                session = AgentSession(cwd=Path.cwd(), session_store=store)
                seen = []

                def bad_listener(event):
                    raise RuntimeError("listener failed")

                session.subscribe(bad_listener)
                session.subscribe(lambda event: seen.append(type(event).__name__))
                yielded = [type(event).__name__ async for event in session.prompt("hello")]
                return yielded, seen, session.listener_errors


        yielded, seen, errors = asyncio.run(run())

        self.assertIn("AgentStartEvent", yielded)
        self.assertEqual(seen, yielded)
        self.assertTrue(errors)
        self.assertIn("listener failed", errors[0])

    def test_active_long_task_context_is_injected_into_next_turn(self) -> None:
        seen_prompts = []
        seen_runtime_contexts = []

        async def fake_run_turn(**kwargs):
            seen_prompts.append(kwargs["prompt"])
            seen_runtime_contexts.append(kwargs.get("runtime_context"))
            kwargs["messages"].append(Message(role="user", content=kwargs["prompt"]))
            yield TurnEndEvent("done")

        session_module.run_turn = fake_run_turn

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SessionStore(root / "sessions", cwd=root)
                session = AgentSession(cwd=root, session_store=store)
                session.start_long_task("Restore Spice fully")
                await _drain(session.prompt("continue"))
                resumed = AgentSession(cwd=root, session_id=session.session_id, session_store=store)
                entries = store.path_entries(session.session_id)
                refs = [entry.data for entry in entries if entry.type == "custom" and entry.data.get("customType") == "long_task_ref"]
                task_id = resumed.long_task_state.task_id
                task_dir = root / "tasks" / task_id
                files_exist = [
                    (task_dir / "state.json").exists(),
                    (task_dir / "checkpoint.json").exists(),
                    (task_dir / "events.jsonl").exists(),
                ]
                return resumed.long_task_state.objective, refs, files_exist


        objective, refs, files_exist = asyncio.run(run())

        self.assertEqual(objective, "Restore Spice fully")
        self.assertTrue(refs)
        self.assertEqual(files_exist, [True, True, True])
        self.assertEqual(seen_prompts[0], "continue")
        self.assertIn("Active sustained goal:", seen_runtime_contexts[0])
        self.assertIn("Restore Spice fully", seen_runtime_contexts[0])
        self.assertIn("has no todo list yet", seen_runtime_contexts[0])

    def test_long_task_start_tool_is_hidden_and_completion_tool_is_active_only_after_task_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = AgentSession(cwd=root, session_store=SessionStore(root / "sessions", cwd=root))

            self.assertNotIn("long_task", session.get_active_tools())
            self.assertNotIn("complete_long_task", session.get_active_tools())

            session.start_long_task("Restore Spice fully")

            self.assertNotIn("long_task", session.get_active_tools())
            self.assertIn("complete_long_task", session.get_active_tools())

    def test_long_task_tools_remain_available_only_while_task_is_active(self) -> None:
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                session = AgentSession(cwd=root, session_store=SessionStore(root / "sessions", cwd=root))
                session.start_long_task("Restore Spice fully")
                tools = {tool.name: tool for tool in session._edit_tools}
                active_tools = session.get_active_tools()
                await tools["complete_long_task"].execute({"note": "verified"}, ToolContext(cwd=root))
                completed_tools = session.get_active_tools()
                return active_tools, completed_tools


        active_tools, completed_tools = asyncio.run(run())

        self.assertIn("complete_long_task", active_tools)
        self.assertNotIn("long_task", completed_tools)
        self.assertNotIn("complete_long_task", completed_tools)

    def test_long_task_without_todo_can_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = AgentSession(cwd=root, session_store=SessionStore(root / "sessions", cwd=root))
            session.start_long_task("Restore Spice fully")

            state = session.complete_long_task(note="verified")

            self.assertEqual(state.status, "completed")

    def test_long_task_with_active_todo_rejects_completion_until_done(self) -> None:
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                session = AgentSession(cwd=root, session_store=SessionStore(root / "sessions", cwd=root))
                session.start_long_task("Restore Spice fully")
                update_todo = next(tool for tool in session._edit_tools if tool.name == "update_todo")
                await update_todo.execute(
                    {
                        "todos": [
                            {"id": "1", "content": "Inspect code", "status": "pending"},
                        ]
                    },
                    ToolContext(cwd=root),
                )
                rejected = None
                try:
                    session.complete_long_task(note="too early")
                except ValueError as exc:
                    rejected = str(exc)
                await update_todo.execute(
                    {
                        "todos": [
                            {"id": "1", "content": "Inspect code", "status": "completed"},
                        ]
                    },
                    ToolContext(cwd=root),
                )
                completed = session.complete_long_task(note="verified")
                return rejected, completed.status, completed.completion_candidate


        rejected, status, completion_candidate = asyncio.run(run())

        self.assertIn("todo items are still pending", rejected)
        self.assertEqual(status, "completed")
        self.assertFalse(completion_candidate)


if __name__ == "__main__":
    unittest.main()
