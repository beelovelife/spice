"""Command, selector, session, and long-task behavior for the legacy TUI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from prompt_toolkit.filters import Condition
from rich.table import Table

from spice.agent.session_history import HistoryError, session_history
from spice.cli.commands import SlashCommandRegistry
from spice.cli.completion import SpiceInputCompleter
from spice.interactive.confirm import ConfirmPolicy
from spice.interactive.sessions import (
    entry_preview,
)
from spice.interactive.types import CommandContext
from spice.llm.config import load_config
from spice.llm.model_registry import ModelRegistry
from spice.skills.loader import load_skills, read_skill_content
from spice.tools.tool_registry import READ_ONLY_TOOLS, TOOLSETS, create_all_tools
from spice.tui.legacy_ports import TuiConsole, TuiInteractivePort


class LegacyCommandMixin:
    def _create_agent_session(self, **kwargs):
        # Resolve through the compatibility module so existing integrations
        # that patch spice.tui.app.AgentSession continue to work.
        from spice.tui import app as legacy_app

        return legacy_app.AgentSession(**kwargs)

    async def _run_slash_command(self, text: str) -> None:
        try:
            await self._handle_slash_command(text)
        except Exception as exc:  # noqa: BLE001
            self._ensure_newline()
            self._append(f"Command failed: {exc}\n")
        finally:
            self._app.invalidate()

    async def _handle_slash_command(self, text: str) -> bool:
        if self._agent_session is None or not text.startswith("/"):
            return False

        from spice.interactive.commands import SlashCommandRegistry as CoreRegistry

        registry = CoreRegistry(self._agent_session.extensions)
        port = TuiInteractivePort(self)
        ctx = CommandContext(
            session=self._agent_session,
            cwd=Path.cwd(),
            port=port,
            raw=text,
            args="",
            confirm_policy=self.confirm_policy,
            extras={
                "extension_context": SimpleNamespace(
                    console=TuiConsole(self),
                    agent_session=self._agent_session,
                    cwd=Path.cwd(),
                ),
                "registry": registry,
            },
        )
        result = await registry.execute(text, ctx)
        if result.exit_requested:
            self._app.exit()
            return True

        previous_session = self._agent_session
        new_session = result.session
        replacing_session = (
            new_session is not None and new_session is not previous_session
        )
        if result.clear_requested or (
            replacing_session and result.replay_session_history
        ):
            self._clear_conversation()
        if new_session is not None and new_session is not previous_session:
            self.confirm_policy = ConfirmPolicy()
            set_confirm = getattr(new_session, "set_confirm", None)
            if set_confirm is not None:
                set_confirm(self.confirm)
            else:
                new_session.confirm = self.confirm
            self._agent_session = new_session
            self._bind_trace_session()
            self._refresh_completer()
            close = getattr(previous_session, "aclose", None)
            if close is not None:
                await close()

        self._paint_command_result(result)
        if replacing_session and new_session is not None:
            self._ensure_newline()
            self._append(f"Session: {new_session.session_label}\n")
            self._append(f"Model: {new_session.runtime_model_label}\n")
            if result.replay_session_history:
                self._display_session_history()
        if result.followup_prompt:
            await self._run_prompt(result.followup_prompt)
        return True

    def _show_settings(self) -> None:
        if self._agent_session is None:
            return
        config = load_config()
        table = Table("Setting", "Value")
        table.add_row("session", self._agent_session.session_label)
        table.add_row("cwd", str(Path.cwd()))
        table.add_row("model", self._model_label())
        table.add_row(
            "protocol", self._agent_session.model.protocol or "provider default"
        )
        table.add_row("default model", f"{config.provider}/{config.model}")
        table.add_row("temperature", str(config.temperature))
        table.add_row("memory.enabled", str(config.memory_enabled).lower())
        table.add_row(
            "subagents.enabled", str(self._agent_session.subagents_enabled).lower()
        )
        table.add_row(
            "subagents.max_concurrent",
            str(self._agent_session.subagent_manager.max_concurrent),
        )
        table.add_row("logging.retention_days", str(config.logging_retention_days))
        table.add_row(
            "tools.max_concurrency", str(config.tools.get("max_concurrency", 4))
        )
        table.add_row(
            "tools.default_timeout_seconds",
            str(config.tools.get("default_timeout_seconds", 120)),
        )
        table.add_row(
            "model retry",
            str(config.model_routing["retry"].get("enabled", True)).lower(),
        )
        table.add_row(
            "model retry attempts",
            str(config.model_routing["retry"].get("maxAttempts", 3)),
        )
        table.add_row(
            "model fallback",
            str(config.model_routing["fallback"].get("enabled", False)).lower(),
        )
        table.add_row(
            "fallback profiles",
            ", ".join(config.model_routing["fallback"].get("profiles", [])) or "(none)",
        )
        table.add_row("output tokens", str(self._agent_session.model.output_tokens))
        table.add_row("mode", self._agent_session.plan_state.mode)
        if self._agent_session.todo_state.status_line():
            table.add_row("todo", self._agent_session.todo_state.status_line())
        table.add_row(
            "tool approvals",
            "all tools allowed for session" if self._allow_all_tools else "ask",
        )
        table.add_row(
            "file edit approvals",
            "session allowed" if self._allow_file_edits else "ask",
        )
        self._append_table(table)

    def _show_tools(self) -> None:
        memory_enabled = load_config().memory_enabled
        subagents_enabled = bool(
            self._agent_session and self._agent_session.subagents_enabled
        )
        registry = create_all_tools(
            memory_enabled=memory_enabled, subagents_enabled=subagents_enabled
        )
        table = Table("Toolset", "Tools")
        for name, tools in TOOLSETS.items():
            if name == "memory" and not memory_enabled:
                continue
            if name == "subagent" and not subagents_enabled:
                continue
            table.add_row(name, ", ".join(tools))
        table.add_row(
            "read-only", ", ".join(name for name in READ_ONLY_TOOLS if name in registry)
        )
        table.add_row(
            "parallel",
            ", ".join(
                name
                for name, tool in registry.items()
                if tool.concurrency == "parallel"
            ),
        )
        self._append_table(table)

    def _handle_subagent_command(self, arg: str) -> None:
        if self._agent_session is None:
            return
        action = (arg or "status").strip().lower()
        if action in {"on", "enable", "enabled"}:
            self._agent_session.set_subagents_enabled(True)
            self._append("Subagents enabled for this session.\n")
            self._show_subagent_status()
            return
        if action in {"off", "disable", "disabled"}:
            self._agent_session.set_subagents_enabled(False)
            self._append("Subagents disabled for this session.\n")
            self._show_subagent_status()
            return
        if action in {"", "status"}:
            self._show_subagent_status()
            return
        self._append("Usage: /subagent [on|off|status]\n")

    def _show_subagent_status(self) -> None:
        if self._agent_session is None:
            return
        active_tools = self._agent_session.get_active_tools()
        table = Table("Setting", "Value")
        table.add_row(
            "session enabled", str(self._agent_session.subagents_enabled).lower()
        )
        table.add_row(
            "max concurrent", str(self._agent_session.subagent_manager.max_concurrent)
        )
        table.add_row("mode", self._agent_session.plan_state.mode)
        table.add_row("tool available", str("spawn_subagents" in active_tools).lower())
        self._append_table(table)

    def _show_models(self, selector: str = "") -> None:
        if self._agent_session is None:
            return
        registry = ModelRegistry()
        if selector:
            provider, _, model_id = selector.partition("/")
            if not model_id:
                model_id = provider
                provider = self._agent_session.model.provider
            model = registry.find(provider, model_id)
            if not model:
                self._ensure_newline()
                self._append(f"Model not found: {selector}\n")
                return
            self._set_model(model)
            return

        self._start_model_selector()

    def _show_sessions(self) -> None:
        self._start_session_selector()

    def _show_delete_sessions(self) -> None:
        self._start_session_selector(mode="delete")

    def _clear_conversation(self) -> None:
        self._message_follow_tail = True
        self._set_message_text("")

    async def _reset_session(self) -> None:
        if self._agent_session is None:
            return
        if self._agent_session.session is None:
            self._clear_conversation()
            self._append("No session has been created yet.\n")
            return
        confirmed = await self._confirm_reset()
        if not confirmed:
            self._ensure_newline()
            self._append("Reset cancelled.\n")
            return

        try:
            self._agent_session.reset()
        except (RuntimeError, ValueError) as exc:
            self._ensure_newline()
            self._append(f"Reset failed: {exc}\n")
            return

        self._clear_conversation()
        self._append(f"Reset session: {self._agent_session.session_label}\n")
        self._append(f"Session: {self._agent_session.session_label}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        self._refresh_completer()

    async def _delete_session(self, target: str) -> None:
        if self._agent_session is None:
            return
        current_session_id = (
            self._agent_session.session_id if self._agent_session.session else None
        )
        session_id = target
        if target == "current":
            if not current_session_id:
                self._ensure_newline()
                self._append("No current session has been created yet.\n")
                return
            session_id = current_session_id
        elif not target:
            self._show_delete_sessions()
            return
        else:
            try:
                session_id = self._agent_session.session_store.resolve(
                    target, cwd=Path.cwd()
                ).id
            except ValueError as exc:
                self._ensure_newline()
                self._append(f"Delete failed: {exc}\n")
                return

        confirmed = await self._confirm_delete_session(
            session_id, current=session_id == current_session_id
        )
        if not confirmed:
            self._ensure_newline()
            self._append("Delete cancelled.\n")
            return

        try:
            self._agent_session.session_store.delete(session_id)
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Delete failed: {exc}\n")
            return

        if session_id != current_session_id:
            self._ensure_newline()
            self._append(f"Deleted session: {session_id}\n")
            return

        current_model = self._agent_session.model
        self._agent_session = self._create_agent_session(
            cwd=Path.cwd(),
            provider=current_model.provider,
            model_id=current_model.id,
            confirm=self.confirm,
            session_store=self._agent_session.session_store,
            extension_manager=self._agent_session.extensions,
        )
        self.confirm_policy = ConfirmPolicy()
        self._bind_trace_session()
        self._clear_conversation()
        self._append(f"Deleted session: {session_id}\n")
        self._append(f"Session: {self._agent_session.session_label}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        self._refresh_completer()

    def _resume_session(self, session_id: str) -> None:
        if not session_id:
            self._start_session_selector()
            return
        try:
            self._agent_session = self._create_agent_session(
                cwd=Path.cwd(),
                provider=self._provider,
                model_id=self._model,
                session_id=session_id,
                confirm=self.confirm,
            )
        except (RuntimeError, ValueError) as exc:
            self._ensure_newline()
            self._append(f"Resume failed: {exc}\n")
            return
        self.confirm_policy = ConfirmPolicy()
        self._bind_trace_session()
        # A resumed session replaces the current conversation view. Resetting
        # follow-tail is essential when the previous view was scrolled up.
        self._clear_conversation()
        self._append(f"Resumed session: {self._agent_session.session_label}\n")
        self._append(f"Model: {self._agent_session.runtime_model_label}\n")
        self._display_session_history()
        self._refresh_completer()

    def _display_session_history(self) -> None:
        """Render session messages like re-entering the conversation."""
        if self._agent_session is None:
            return
        messages = session_history(self._agent_session)
        if not messages:
            return
        self._append("\n")
        for message in messages:
            if isinstance(message, HistoryError):
                self._append(f"{message.display_text}\n\n")
                continue
            if message.role == "user":
                content = (message.content or "").strip()
                if content:
                    self._append(f"> {content}\n\n")
            elif message.role == "assistant":
                content = (message.content or "").strip()
                if content:
                    self._append(self._render_markdown_text(content))
                    self._append("\n")
                if message.tool_calls:
                    for call in message.tool_calls:
                        self._append(f"  tool: {call.name}()\n")
            elif message.role == "tool":
                content = (message.content or "").strip()
                name = message.name or "tool"
                if content:
                    preview = content[:300] + "..." if len(content) > 300 else content
                    self._append(f"  {name} -> {preview}\n")

    def _show_history(self, args: str) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        store = self._agent_session.session_store
        session_id = self._agent_session.session_id
        try:
            if args == "--raw":
                path = store.path_for(session_id)
                lines = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        lines.append(line)
                    else:
                        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
                self._ensure_newline()
                self._append("\n".join(lines) + "\n")
                return
            if args == "--tree":
                self._show_history_tree()
                return
            if args:
                self._ensure_newline()
                self._append("Usage: /history [--tree|--raw]\n")
                return
            info = store.info(session_id)
            active_context = store.build_context(session_id)
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return

        table = Table("Field", "Value")
        table.add_row("ID", info.id)
        table.add_row("Model", f"{info.provider}/{info.model}")
        table.add_row("CWD", info.cwd)
        table.add_row("Messages", str(info.message_count))
        table.add_row("Leaf", active_context.leaf_id or "")
        self._append_table(table)
        for message in active_context.messages:
            title = (
                message.role if not message.name else f"{message.role}:{message.name}"
            )
            body = message.content or "<empty>"
            if message.tool_calls:
                calls = ", ".join(
                    f"{call.name}({call.id})" for call in message.tool_calls
                )
                body = f"{body}\n\nTool calls: {calls}".strip()
            self._ensure_newline()
            self._append(f"{title}\n{body}\n")

    def _show_history_tree(self) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        store = self._agent_session.session_store
        session_id = self._agent_session.session_id
        try:
            info = store.info(session_id)
            active_path_ids = (
                {entry.id for entry in store.path_entries(session_id)}
                if info.leaf_id
                else set()
            )
            entries = store.entries(session_id)
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return
        table = Table("Entry", "Parent", "Type", "Active", "Preview")
        for entry in entries:
            active = (
                "yes"
                if entry.id == info.leaf_id
                else ("path" if entry.id in active_path_ids else "")
            )
            table.add_row(
                entry.id,
                entry.parent_id or "",
                entry.type,
                active,
                entry_preview(entry.data),
            )
        self._append_table(table)

    def _rewind_session(self, entry_id: str) -> None:
        if self._agent_session is None or self._agent_session.session is None:
            self._ensure_newline()
            self._append("No session has been created yet.\n")
            return
        if not entry_id:
            self._start_rewind_selector()
            return
        try:
            self._agent_session.rewind(entry_id)
        except (RuntimeError, ValueError) as exc:
            self._ensure_newline()
            self._append(f"Error: {exc}\n")
            return
        self._ensure_newline()
        self._append(
            f"Rewound session: {self._agent_session.session_id} -> {entry_id}\n"
        )

    async def _show_compact(self, args: str = "") -> None:
        session = self._agent_session
        self._ensure_newline()
        if session is None or session.session is None:
            self._append("No session has been created yet.\n")
            return
        if args in {"status", "--status"}:
            status = session.compaction_status()
            threshold = (
                str(status.threshold_tokens)
                if status.threshold_tokens is not None
                else "n/a"
            )
            self._append(
                f"Compaction status: estimated={status.estimated_tokens} threshold={threshold} "
                f"reserve={status.reserve_tokens} auto={'yes' if status.should_compact else 'no'} "
                f"reason={status.reason}\n"
            )
            return
        try:
            result = await session.compact(
                focus=args or None, reason="manual", force=True
            )
        except Exception as exc:
            self._append(f"Compaction failed: {exc}\n")
            return
        self._append(
            f"Compacted session: ~{result.tokens_before} -> ~{result.tokens_after} tokens\n"
        )

    async def _handle_memory_command(self, args: str) -> None:
        session = self._agent_session
        if session is None:
            return
        words = args.split()
        action = words[0].lower() if words else "project"
        store = session.memory_store
        if action == "status":
            status = store.status()
            table = Table("Memory", "Usage")
            table.add_row("Project", status.get("project_usage") or "n/a")
            table.add_row("User", status["user_usage"])
            table.add_row("Global", status["memory_usage"])
            table.add_row("Unprocessed history", str(status["unprocessed_count"]))
            table.add_row("Workspace", status.get("workspace") or "n/a")
            self._append_table(table)
            return
        if action == "show":
            scope = words[1].lower() if len(words) > 1 else "project"
            target = {"user": "user", "global": "memory", "project": "project"}.get(
                scope
            )
            if target is None:
                self._append("Usage: /memory show [user|global|project]\n")
                return
            entries = store.read_entries(target)
            self._append(
                f"{scope} memory:\n"
                + ("\n\n".join(entries) if entries else "(empty)")
                + "\n"
            )
            return
        if not session.config.memory_enabled:
            self._append(
                "Long-term memory is disabled. Run `spice memory enable` first.\n"
            )
            return
        if action not in {"distill", "all", "project", "global"}:
            self._append(
                "Usage: /memory [project|global|all|status|show [user|global|project]]\n"
            )
            return
        distill_scope: Literal["all", "global", "project"] = (
            "project"
            if action == "project"
            else "global"
            if action == "global"
            else "all"
        )
        try:
            result = await session.distill_current_memory(scope=distill_scope)
        except Exception as exc:
            self._append(f"Memory distillation failed: {exc}\n")
            return
        if result.get("processed", 0) == 0:
            self._append(
                str(result.get("message", "No conversation content to distill.")) + "\n"
            )
        elif result.get("success"):
            self._append(
                "Memory updated: "
                f"adds={result.get('adds', 0)} replacements={result.get('replacements', 0)} "
                f"removals={result.get('removals', 0)}\n"
            )
        else:
            self._append(
                str(
                    result.get(
                        "message", "Memory review completed with skipped changes."
                    )
                )
                + "\n"
            )

    def _show_skills(self) -> None:
        result = load_skills(cwd=Path.cwd())
        if not result.skills:
            self._ensure_newline()
            self._append("No skills found.\n")
            return
        table = Table("Skill", "Description")
        for skill in result.skills:
            table.add_row(skill.name, skill.description or "")
        self._append_table(table)

    def _show_skill(self, name: str) -> None:
        if not name:
            self._ensure_newline()
            self._append("Usage: /skill:<name>\n")
            return
        try:
            content = read_skill_content(name, cwd=Path.cwd())
        except ValueError as exc:
            self._ensure_newline()
            self._append(f"{exc}\n")
            return
        self._ensure_newline()
        self._append(f"# Skill: {name}\n")
        self._append(self._render_markdown_text(content))

    def _refresh_completer(self) -> None:
        if self._agent_session is None:
            return
        slash_completer = SlashCommandRegistry(
            self._agent_session.extensions
        ).completer()
        completer = SpiceInputCompleter(slash_completer, cwd=Path.cwd())
        self._input_buffer.completer = completer
        self._input_buffer.complete_while_typing = Condition(lambda: True)

    def _toggle_interaction_mode(self) -> None:
        if self._agent_session is None:
            return
        mode = self._agent_session.toggle_interaction_mode()
        self._show_edit_mode_hint = mode == "edit"

    async def _handle_plan_command(self, arg: str) -> None:
        if self._agent_session is None:
            return
        if arg == "cancel":
            self._agent_session.cancel_plan()
            self._show_edit_mode_hint = False
            self._ensure_newline()
            self._append("Switched to edit mode. Plan cleared.\n")
            return
        if arg == "execute":
            prompt = self._agent_session.approve_plan("manual")
            self._show_edit_mode_hint = False
            self._ensure_newline()
            self._append("Switched to edit mode. Executing the approved plan.\n")
            await self._run_prompt(prompt)
            return
        self._agent_session.start_plan(arg)
        self._ensure_newline()
        self._append("Plan mode on. Read-only tools are active.\n")
        if arg:
            await self._run_prompt(arg)

    async def _handle_sustained_goal_command(self, arg: str, *, command: str) -> None:
        if self._agent_session is None:
            return
        objective = arg.strip()
        if not objective:
            self._ensure_newline()
            self._append(f"Usage: /{command} <objective|status|cancel|complete>\n")
            return
        action, _, rest = objective.partition(" ")
        action = action.lower()
        if action == "status":
            state = self._agent_session.long_task_status()
            self._ensure_newline()
            if not state.objective:
                self._append(f"No sustained {command} is active.\n")
                return
            self._append(
                f"Sustained {command}: {state.status} {state.task_id or 'legacy'}\n"
                f"Objective: {state.objective}\n"
                f"Continuations: {state.continuation_rounds}/{state.max_continuation_rounds}, remaining {state.remaining_continuations}\n"
                f"Needs attention: {str(state.needs_user_attention).lower()}\n"
                f"Completion candidate: {str(state.completion_candidate).lower()}\n"
            )
            return
        if action == "cancel":
            try:
                self._agent_session.cancel_long_task(note=rest.strip())
            except ValueError as exc:
                self._ensure_newline()
                self._append(f"{exc}\n")
                return
            self._ensure_newline()
            self._append(f"Sustained {command} cancelled.\n")
            return
        if action == "complete":
            from spice.interactive.commands import parse_force_note

            force, note = parse_force_note(rest)
            try:
                self._agent_session.complete_long_task(note=note, force=force)
            except ValueError as exc:
                self._ensure_newline()
                self._append(f"{exc}\n")
                return
            self._ensure_newline()
            self._append(f"Sustained {command} completed.\n")
            return
        self._agent_session.set_interaction_mode("edit")
        self._agent_session.start_long_task(objective)
        self._ensure_newline()
        self._append(f"Sustained {command} started.\n")
        from spice.interactive.commands import sustained_goal_prompt

        await self._run_prompt(sustained_goal_prompt(objective, command=command))
