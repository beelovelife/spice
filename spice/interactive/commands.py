"""Slash command registry with handlers bound on each command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spice.interactive.confirm import ConfirmChoice, ConfirmRequest
from spice.interactive.sessions import (
    entry_preview,
    list_model_choices,
    list_rewind_choices,
    list_session_choices,
    replace_session,
)
from spice.interactive.types import (
    CommandContext,
    CommandHandler,
    CommandResult,
    CommandView,
    PanelView,
    TableView,
    TextView,
)
from spice.llm.config import get_api_key, load_config, save_config
from spice.llm.model_registry import ModelRegistry
from spice.llm.usage import aggregate_session_usage
from spice.skills.loader import load_skills, read_skill_content
from spice.tools.tool_registry import READ_ONLY_TOOLS, TOOLSETS, create_all_tools

if TYPE_CHECKING:
    from spice.extensions.manager import ExtensionManager


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    usage: str
    handler: CommandHandler

    @property
    def trigger(self) -> str:
        return f"/{self.name}"


def ok(*views: CommandView, **kwargs) -> CommandResult:
    return CommandResult(views=list(views), **kwargs)


def err(message: str, **kwargs) -> CommandResult:
    return CommandResult(views=[TextView(message, "error")], **kwargs)


def info(message: str, **kwargs) -> CommandResult:
    return CommandResult(views=[TextView(message, "plain")], **kwargs)


def success(message: str, **kwargs) -> CommandResult:
    return CommandResult(views=[TextView(message, "success")], **kwargs)


def warn(message: str, **kwargs) -> CommandResult:
    return CommandResult(views=[TextView(message, "warning")], **kwargs)


def dim(message: str, **kwargs) -> CommandResult:
    return CommandResult(views=[TextView(message, "dim")], **kwargs)


# --- handlers ---


async def handle_help(ctx: CommandContext) -> CommandResult:
    registry = ctx.extras.get("registry")
    commands: list[SlashCommand] = (
        list(registry.list_commands()) if registry is not None else []
    )
    rows = tuple((cmd.trigger, cmd.description, cmd.usage) for cmd in commands)
    return ok(TableView("Commands", ("Command", "Description", "Usage"), rows))


async def handle_quit(ctx: CommandContext) -> CommandResult:
    return CommandResult(exit_requested=True)


async def handle_clear(ctx: CommandContext) -> CommandResult:
    return CommandResult(clear_requested=True)


async def handle_tools(ctx: CommandContext) -> CommandResult:
    config = load_config()
    memory_enabled = config.memory_enabled
    subagents_enabled = bool(
        getattr(ctx.session, "subagents_enabled", config.subagents_enabled)
    )
    registry = create_all_tools(
        memory_enabled=memory_enabled, subagents_enabled=subagents_enabled
    )
    rows: list[tuple[str, ...]] = []
    for toolset, names in TOOLSETS.items():
        if toolset == "memory" and not memory_enabled:
            continue
        if toolset == "subagent" and not subagents_enabled:
            continue
        for name in names:
            tool = registry.get(name)
            if not tool:
                continue
            risk = (
                "read-only"
                if name in READ_ONLY_TOOLS and not tool.requires_confirmation
                else "write/exec"
            )
            timeout = (
                f"{tool.timeout_seconds:g}s"
                if tool.timeout_seconds is not None
                else "default"
            )
            rows.append(
                (name, toolset, risk, tool.concurrency, timeout, tool.description)
            )
    return ok(
        TableView(
            None,
            ("Tool", "Toolset", "Risk", "Execution", "Timeout", "Description"),
            tuple(rows),
        )
    )


async def handle_settings(ctx: CommandContext) -> CommandResult:
    config = load_config()
    session = ctx.session
    manager = getattr(session, "subagent_manager", None)
    plan_state = getattr(session, "plan_state", None)
    todo_state = getattr(session, "todo_state", None)
    rows: list[tuple[str, str]] = [
        ("session", session.session_label),
        ("cwd", str(ctx.cwd)),
        ("active model", f"{session.model.provider}/{session.model.id}"),
        ("protocol", session.model.protocol or "provider default"),
        ("default model", config.default_model),
        ("temperature", str(config.temperature)),
        ("memory.enabled", str(config.memory_enabled).lower()),
        (
            "subagents.enabled",
            str(
                getattr(session, "subagents_enabled", config.subagents_enabled)
            ).lower(),
        ),
        (
            "subagents.max_concurrent",
            str(getattr(manager, "max_concurrent", config.max_concurrent_subagents)),
        ),
        ("logging.retention_days", str(config.logging_retention_days)),
        ("tools.max_concurrency", str(config.tools.get("max_concurrency", 4))),
        (
            "tools.default_timeout_seconds",
            str(config.tools.get("default_timeout_seconds", 120)),
        ),
        (
            "model retry",
            str(config.model_routing["retry"].get("enabled", True)).lower(),
        ),
        (
            "model retry attempts",
            str(config.model_routing["retry"].get("maxAttempts", 3)),
        ),
        (
            "model fallback",
            str(config.model_routing["fallback"].get("enabled", False)).lower(),
        ),
        (
            "fallback profiles",
            ", ".join(config.model_routing["fallback"].get("profiles", [])) or "(none)",
        ),
        ("output_tokens", str(session.model.output_tokens)),
        ("mode", str(getattr(plan_state, "mode", "edit"))),
    ]
    if todo_state is not None and todo_state.status_line():
        rows.append(("todo", todo_state.status_line()))
    rows.append(("tools", ", ".join(session.get_active_tools())))
    rows.append(("session save", "enabled"))
    policy = ctx.confirm_policy
    rows.append(
        (
            "tool approvals",
            "all tools allowed for session" if policy.allow_all_tools else "ask",
        )
    )
    rows.append(
        ("file edit approvals", "session allowed" if policy.allow_file_edits else "ask")
    )
    return ok(TableView(None, ("Setting", "Value"), tuple(rows)))


async def handle_cost(ctx: CommandContext) -> CommandResult:
    session = ctx.session
    if session.session is None:
        summary_rows = (
            ("Model calls", "0"),
            ("Input tokens", "0"),
            ("Output tokens", "0"),
            ("Total tokens", "0"),
            ("Estimated cost", "unavailable"),
        )
        return ok(TableView("Session usage", ("Metric", "Value"), summary_rows))
    summary = aggregate_session_usage(session.session_store.entries(session.session_id))
    rate = summary.cache_hit_rate
    if rate is None:
        cache_hit = "unavailable"
    else:
        cache_hit = f"{rate * 100:.1f}%"
        if summary.cache_unavailable_calls:
            cache_hit += (
                f" (observed {summary.cache_metrics_calls}/{summary.model_calls} calls)"
            )
    cost = (
        "unavailable"
        if summary.estimated_cost_usd is None
        else f"${summary.estimated_cost_usd}"
    )
    if summary.estimated_cost_usd is not None and summary.unpriced_calls:
        cost = f">= {cost} ({summary.unpriced_calls} calls unavailable)"
    rows = (
        ("Model calls", f"{summary.model_calls:,}"),
        ("Input tokens", f"{summary.input_tokens:,}"),
        ("Cache hit", cache_hit),
        ("Cache read", f"{summary.cache_read_tokens:,}"),
        ("Cache write", f"{summary.cache_write_tokens:,}"),
        ("Output tokens", f"{summary.output_tokens:,}"),
        ("Total tokens", f"{summary.total_tokens:,}"),
        ("Estimated cost", cost),
    )
    return ok(TableView("Session usage", ("Metric", "Value"), rows))


async def handle_mcp(ctx: CommandContext) -> CommandResult:
    action = ctx.args.strip()
    if action == "reload":
        await ctx.session.reload_mcp()
    elif action:
        return err("Usage: /mcp [reload]")
    statuses = ctx.session.mcp.status()
    if not statuses:
        return info("No MCP servers configured.")
    rows = tuple(
        (
            status.name,
            status.state,
            status.transport,
            status.source,
            str(status.tool_count),
            status.error or "",
        )
        for status in statuses
    )
    return ok(
        TableView(
            None, ("Server", "State", "Transport", "Source", "Tools", "Error"), rows
        ),
    )


async def handle_subagent(ctx: CommandContext) -> CommandResult:
    action = (ctx.args or "status").strip().lower()
    views: list[CommandView] = []
    if action in {"on", "enable", "enabled"}:
        ctx.session.set_subagents_enabled(True)
        views.append(TextView("Subagents enabled for this session.", "success"))
    elif action in {"off", "disable", "disabled"}:
        ctx.session.set_subagents_enabled(False)
        views.append(TextView("Subagents disabled for this session.", "warning"))
    elif action not in {"", "status"}:
        return err("Usage: /subagent [on|off|status]")
    enabled = bool(getattr(ctx.session, "subagents_enabled", False))
    plan_mode = bool(getattr(ctx.session.plan_state, "is_plan_mode", False))
    active_tools = ctx.session.get_active_tools()
    manager = getattr(ctx.session, "subagent_manager", None)
    max_concurrent = getattr(manager, "max_concurrent", "n/a")
    views.append(
        TableView(
            None,
            ("Setting", "Value"),
            (
                ("session enabled", str(enabled).lower()),
                ("max concurrent", str(max_concurrent)),
                ("mode", "plan" if plan_mode else "edit"),
                ("tool available", str("spawn_subagents" in active_tools).lower()),
            ),
        )
    )
    return CommandResult(views=views)


async def handle_skills(ctx: CommandContext) -> CommandResult:
    result = load_skills(cwd=ctx.cwd)
    views: list[CommandView] = []
    if not result.skills:
        views.append(TextView("No skills found in ~/.spice/skills or ./.spice/skills."))
    else:
        rows = tuple(
            (skill.name, skill.source, skill.description) for skill in result.skills
        )
        views.append(TableView(None, ("Skill", "Source", "Description"), rows))
    for diagnostic in result.diagnostics:
        views.append(TextView(f"{diagnostic.type}: {diagnostic.message}", "warning"))
    return CommandResult(views=views)


async def handle_skill(ctx: CommandContext) -> CommandResult:
    name = ctx.args.strip()
    if not name:
        return err("Usage: /skill:<name>")
    try:
        content = read_skill_content(name, cwd=ctx.cwd)
    except ValueError as exc:
        return err(str(exc))
    return ok(TextView(content))


async def handle_compact(ctx: CommandContext) -> CommandResult:
    session = ctx.session
    if session.session is None:
        return info("No session has been created yet.")
    args = ctx.args.strip()
    if args in {"status", "--status"}:
        status = session.compaction_status()
        threshold = (
            str(status.threshold_tokens)
            if status.threshold_tokens is not None
            else "n/a"
        )
        body = (
            f"Estimated tokens: {status.estimated_tokens}\n"
            f"Threshold: {threshold}\n"
            f"Reserve: {status.reserve_tokens}\n"
            f"Auto compact: {'yes' if status.should_compact else 'no'}\n"
            f"Reason: {status.reason}"
        )
        return ok(PanelView("/compact status", body))
    focus = args or None
    try:
        result = await session.compact(focus=focus, reason="manual", force=True)
    except Exception as exc:
        return err(f"Compaction failed: {exc}")
    return success(
        f"Compacted session: ~{result.tokens_before} -> ~{result.tokens_after} tokens",
    )


async def handle_memory(ctx: CommandContext) -> CommandResult:
    words = ctx.args.split()
    action = words[0].lower() if words else "project"
    store = ctx.session.memory_store
    if action == "status":
        status = store.status()
        rows = (
            ("Project", status.get("project_usage") or "n/a"),
            ("User", status["user_usage"]),
            ("Global", status["memory_usage"]),
            ("Unprocessed history", str(status["unprocessed_count"])),
            ("Workspace", status.get("workspace") or "n/a"),
        )
        return ok(TableView(None, ("Memory", "Usage"), rows))
    if action == "show":
        scope = words[1].lower() if len(words) > 1 else "project"
        target = {"user": "user", "global": "memory", "project": "project"}.get(scope)
        if target is None:
            return err("Usage: /memory show [user|global|project]")
        entries = store.read_entries(target)
        body = "\n\n".join(entries) if entries else "(empty)"
        return ok(PanelView(f"{scope} memory", body))
    if not ctx.session.config.memory_enabled:
        return warn("Long-term memory is disabled. Run `spice memory enable` first.")
    if action not in {"distill", "all", "project", "global"}:
        return err(
            "Usage: /memory [project|global|all|status|show [user|global|project]]"
        )
    distill_scope: Literal["all", "global", "project"] = (
        "project" if action == "project" else "global" if action == "global" else "all"
    )
    try:
        result = await ctx.session.distill_current_memory(scope=distill_scope)
    except Exception as exc:
        return err(f"Memory distillation failed: {exc}")
    if result.get("processed", 0) == 0:
        return info(result.get("message", "No conversation content to distill."))
    if result.get("success"):
        return success(
            f"Memory updated: adds={result.get('adds', 0)} "
            f"replacements={result.get('replacements', 0)} removals={result.get('removals', 0)}",
        )
    return warn(result.get("message", "Memory review completed with skipped changes."))


async def handle_models(ctx: CommandContext) -> CommandResult:
    selector = ctx.args.strip()
    registry = ModelRegistry()
    if selector:
        provider, _, model_id = selector.partition("/")
        if not model_id:
            model_id = provider
            provider = ctx.session.model.provider
        model = registry.find(provider, model_id)
        if not model:
            return err(f"Model not found: {selector}")
        return _apply_model(ctx, model)

    request = list_model_choices(ctx.session.model)
    selected = await ctx.port.choose(request)
    if not selected:
        return CommandResult()
    provider, model_id = selected.split("/", 1)
    model = registry.find(provider, model_id)
    if not model:
        return err(f"Model not found: {selected}")
    return _apply_model(ctx, model)


def _apply_model(ctx: CommandContext, model) -> CommandResult:
    ctx.session.set_model(model)
    config = load_config()
    config.default_model = model.profile_key or model.id
    config.provider = model.provider
    config.model = model.id
    config.protocol = model.protocol
    config.base_url = model.base_url
    if model.temperature is not None:
        config.temperature = model.temperature
    save_config(config)
    views: list[CommandView] = [
        TextView(
            f"Set model to {model.provider}/{model.id} and saved as your default for new sessions",
            "success",
        )
    ]
    if not get_api_key(model.provider, env_names=model.api_key_envs):
        views.append(TextView(f"No API key found for {model.provider}.", "warning"))
    return CommandResult(views=views)


async def handle_sessions(ctx: CommandContext) -> CommandResult:
    return await _resume_like(ctx, title="Sessions")


async def handle_resume(ctx: CommandContext) -> CommandResult:
    arg = ctx.args.strip()
    if arg:
        try:
            new_session = replace_session(
                ctx.session,
                session_id=arg,
                confirm=getattr(ctx.session, "confirm", None),
            )
        except (RuntimeError, ValueError) as exc:
            return err(str(exc))
        return success(
            f"Resumed session: {new_session.session_id}",
            session=new_session,
            replay_session_history=True,
        )
    return await _resume_like(ctx, title="Sessions")


async def _resume_like(ctx: CommandContext, *, title: str) -> CommandResult:
    active_id = None
    if getattr(ctx.session, "session", None) is not None:
        active_id = getattr(ctx.session, "session_id", None)
    request = list_session_choices(
        cwd=ctx.cwd,
        active_id=active_id,
        title=title,
    )
    if not request.items:
        return info("No sessions found for this workspace.")
    selected = await ctx.port.choose(request)
    if not selected:
        return CommandResult()
    try:
        new_session = replace_session(
            ctx.session,
            session_id=selected,
            confirm=getattr(ctx.session, "confirm", None),
        )
    except (RuntimeError, ValueError) as exc:
        return err(str(exc))
    return success(
        f"Resumed session: {new_session.session_id}",
        session=new_session,
        replay_session_history=True,
    )


async def handle_reset(ctx: CommandContext) -> CommandResult:
    if ctx.session.session is None:
        return CommandResult(
            clear_requested=True, views=[TextView("No session has been created yet.")]
        )
    decision = await ctx.port.confirm(
        ConfirmRequest(
            question="Reset current session? This clears all messages but keeps the session id.",
            choices=(
                ConfirmChoice(
                    "yes", "Yes", "Clear all messages but keep the current session id."
                ),
                ConfirmChoice("no", "No", "Keep the current session."),
            ),
            current_id="yes",
        )
    )
    if decision != "yes":
        return dim("Reset cancelled.")
    try:
        ctx.session.reset()
    except (RuntimeError, ValueError) as exc:
        return err(str(exc))
    return success(
        f"Reset session: {ctx.session.session_id}",
        clear_requested=True,
    )


async def handle_delete(ctx: CommandContext) -> CommandResult:
    store = ctx.session.session_store
    current_session_id = ctx.session.session_id if ctx.session.session else None
    target = ctx.args.strip()
    selected_id = target
    if target == "current":
        if not current_session_id:
            return info("No current session has been created yet.")
        selected_id = current_session_id
    elif not target:
        request = list_session_choices(
            cwd=ctx.cwd,
            active_id=current_session_id,
            title="Sessions to delete",
            store=store,
        )
        if not request.items:
            return info("No sessions found for this workspace.")
        selected_id = await ctx.port.choose(request) or ""
        if not selected_id:
            return CommandResult()
    else:
        try:
            selected_id = store.resolve(target, cwd=ctx.cwd).id
        except ValueError as exc:
            return err(str(exc))

    is_current = selected_id == current_session_id
    detail = (
        "Delete this current session and all of its content. A fresh session will start."
        if is_current
        else "Delete this session and all of its content."
    )
    decision = await ctx.port.confirm(
        ConfirmRequest(
            question=f"Delete session {selected_id}?",
            choices=(
                ConfirmChoice("yes", "Yes", detail),
                ConfirmChoice("no", "No", "Keep the session."),
            ),
            current_id="yes",
        )
    )
    if decision != "yes":
        return dim("Delete cancelled.")
    try:
        store.delete(selected_id)
    except ValueError as exc:
        return err(str(exc))
    views: list[CommandView] = [TextView(f"Deleted session: {selected_id}", "success")]
    if not is_current:
        return CommandResult(views=views)
    new_session = replace_session(
        ctx.session,
        fresh=True,
        confirm=getattr(ctx.session, "confirm", None),
        reuse_store=True,
        reuse_extensions=True,
    )
    # New policy for the fresh session is the caller's responsibility when re-binding confirm;
    # replace_session keeps the same confirm fn for tools.
    views.append(TextView("Started a fresh session.", "success"))
    return CommandResult(views=views, session=new_session, clear_requested=True)


async def handle_history(ctx: CommandContext) -> CommandResult:
    if ctx.session.session is None:
        return info("No session has been created yet.")
    store = ctx.session.session_store
    session_id = ctx.session.session_id
    args = ctx.args.strip()
    try:
        if args == "--raw":
            path = store.path_for(session_id)
            lines: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                    lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    lines.append(line)
            return ok(TextView("\n".join(lines) if lines else "(empty)"))
        if args == "--tree":
            session_info = store.info(session_id)
            active_path_ids = (
                {entry.id for entry in store.path_entries(session_id)}
                if session_info.leaf_id
                else set()
            )
            rows = tuple(
                (
                    entry.id,
                    entry.parent_id or "",
                    entry.type,
                    "yes"
                    if entry.id == session_info.leaf_id
                    else ("path" if entry.id in active_path_ids else ""),
                    entry_preview(entry.data),
                )
                for entry in store.entries(session_id)
            )
            return ok(
                TableView(None, ("Entry", "Parent", "Type", "Active", "Preview"), rows)
            )
        if args:
            return err("Usage: /history [--tree|--raw]")
        session_info = store.info(session_id)
        active_context = store.build_context(session_id)
    except ValueError as exc:
        return err(str(exc))
    views: list[CommandView] = [
        PanelView(
            "Session",
            f"ID: {session_info.id}\n"
            f"Model: {session_info.provider}/{session_info.model}\n"
            f"CWD: {session_info.cwd}\n"
            f"Messages: {session_info.message_count}\n"
            f"Leaf: {active_context.leaf_id or ''}",
        )
    ]
    for message in active_context.messages:
        title = message.role
        if message.name:
            title += f":{message.name}"
        body = message.content or ""
        if message.tool_calls:
            calls = ", ".join(f"{call.name}({call.id})" for call in message.tool_calls)
            body = f"{body}\n\nTool calls: {calls}".strip()
        views.append(PanelView(title, body or "<empty>"))
    return CommandResult(views=views)


async def handle_rewind(ctx: CommandContext) -> CommandResult:
    if ctx.session.session is None:
        return info("No session has been created yet.")
    entry_id = ctx.args.strip()
    if not entry_id:
        try:
            request = list_rewind_choices(ctx.session)
        except ValueError as exc:
            return err(str(exc))
        if not request.items:
            return info("No rewind targets.")
        entry_id = await ctx.port.choose(request) or ""
        if not entry_id:
            return CommandResult()
    try:
        ctx.session.rewind(entry_id)
    except (RuntimeError, ValueError) as exc:
        return err(str(exc))
    return success(f"Rewound session: {ctx.session.session_id} -> {entry_id}")


async def handle_plan(ctx: CommandContext) -> CommandResult:
    args = ctx.args.strip()
    if args == "cancel":
        ctx.session.cancel_plan()
        return success("Switched to edit mode. Plan cleared.")
    if args == "execute":
        prompt = ctx.session.approve_plan("manual")
        return success(
            "Switched to edit mode. Executing the approved plan.",
            followup_prompt=prompt,
        )
    ctx.session.start_plan(args)
    if args:
        return CommandResult(
            views=[TextView("Plan mode on. Read-only tools are active.", "plain")],
            followup_prompt=args,
        )
    return info(
        "Plan mode on. Send the task you want to plan. Shift+Tab toggles back to edit mode."
    )


async def handle_task(ctx: CommandContext) -> CommandResult:
    return await _handle_sustained_goal(ctx, command="task")


async def handle_goal(ctx: CommandContext) -> CommandResult:
    return await _handle_sustained_goal(ctx, command="goal")


async def _handle_sustained_goal(ctx: CommandContext, *, command: str) -> CommandResult:
    objective = ctx.args.strip()
    if not objective:
        return err(f"Usage: /{command} <objective|status|cancel|complete>")
    action, _, rest = objective.partition(" ")
    action = action.lower()
    if action == "status":
        return _long_task_status(ctx, command=command)
    if action == "cancel":
        note = rest.strip()
        try:
            ctx.session.cancel_long_task(note=note)
        except ValueError as exc:
            return err(str(exc))
        return warn(f"Sustained {command} cancelled.")
    if action == "complete":
        force, note = parse_force_note(rest)
        try:
            ctx.session.complete_long_task(note=note, force=force)
        except ValueError as exc:
            return err(str(exc))
        return success(f"Sustained {command} completed.")
    ctx.session.set_interaction_mode("edit")
    if hasattr(ctx.session, "start_long_task"):
        ctx.session.start_long_task(objective)
    prompt = sustained_goal_prompt(objective, command=command)
    return CommandResult(
        views=[TextView(f"Sustained {command} started.", "plain")],
        followup_prompt=prompt,
    )


def _long_task_status(ctx: CommandContext, *, command: str) -> CommandResult:
    state = ctx.session.long_task_status()
    if not state.objective:
        return info(f"No sustained {command} is active.")
    rows = (
        ("task id", state.task_id or "legacy"),
        ("status", state.status),
        ("objective", state.objective),
        (
            "continuations",
            f"{state.continuation_rounds}/{state.max_continuation_rounds}",
        ),
        ("remaining", str(state.remaining_continuations)),
        ("needs attention", str(state.needs_user_attention).lower()),
        ("completion candidate", str(state.completion_candidate).lower()),
    )
    if state.last_stop_reason:
        rows = (*rows, ("last stop", state.last_stop_reason))
    return ok(TableView(None, ("Field", "Value"), rows))


def sustained_goal_prompt(objective: str, *, command: str) -> str:
    label = "goal" if command == "goal" else "task"
    return f"""Start or continue this sustained {label}:

{objective}

This is long-running execution mode, not read-only planning mode.

The sustained goal has already been persisted by the /{command} command. Work toward the objective using the available tools. Break work into concrete steps when useful, keep the todo list updated for immediate execution progress, and continue until the objective is actually done or you need user input.

When the objective is fully done and verified, call complete_long_task with a concise completion note. Do not call complete_long_task merely because you produced a plan.
"""


def parse_force_note(text: str) -> tuple[bool, str]:
    parts = text.split()
    force = False
    kept: list[str] = []
    for part in parts:
        if part == "--force":
            force = True
        else:
            kept.append(part)
    return force, " ".join(kept)


def is_execute_request(message: str) -> bool:
    normalized = message.strip().lower()
    return normalized in {
        "执行",
        "执行吧",
        "开始执行",
        "开始改",
        "按这个做",
        "可以执行",
        "go",
        "go ahead",
        "execute",
        "run it",
        "do it",
    }


def build_core_commands() -> list[SlashCommand]:
    return [
        SlashCommand(
            "models",
            "Choose the current provider/model.",
            "/models [provider/model]",
            handle_models,
        ),
        SlashCommand(
            "sessions",
            "List recent sessions for this workspace.",
            "/sessions",
            handle_sessions,
        ),
        SlashCommand(
            "resume",
            "Choose and resume a previous session.",
            "/resume [session-id]",
            handle_resume,
        ),
        SlashCommand(
            "clear",
            "Clear the visible conversation without deleting the session.",
            "/clear",
            handle_clear,
        ),
        SlashCommand(
            "reset",
            "Clear all messages from the current session after confirmation.",
            "/reset",
            handle_reset,
        ),
        SlashCommand(
            "delete",
            "Delete a session after confirmation.",
            "/delete [session-id|current]",
            handle_delete,
        ),
        SlashCommand(
            "history",
            "Show current session history.",
            "/history [--tree|--raw]",
            handle_history,
        ),
        SlashCommand(
            "rewind",
            "Move the current session leaf to an entry.",
            "/rewind [entry-id]",
            handle_rewind,
        ),
        SlashCommand(
            "tools", "Show built-in tools and toolsets.", "/tools", handle_tools
        ),
        SlashCommand(
            "mcp", "Show or reload MCP server connections.", "/mcp [reload]", handle_mcp
        ),
        SlashCommand(
            "settings",
            "Show current interactive settings.",
            "/settings",
            handle_settings,
        ),
        SlashCommand(
            "cost",
            "Show model usage and estimated cost for this session.",
            "/cost",
            handle_cost,
        ),
        SlashCommand("usage", "Alias for /cost.", "/usage", handle_cost),
        SlashCommand(
            "subagent",
            "Control subagent tools for this session.",
            "/subagent [on|off|status]",
            handle_subagent,
        ),
        SlashCommand(
            "compact",
            "Compact the current session context.",
            "/compact [status|focus]",
            handle_compact,
        ),
        SlashCommand(
            "memory",
            "Distill or inspect long-term memory.",
            "/memory [project|global|all|status|show [user|global|project]]",
            handle_memory,
        ),
        SlashCommand(
            "plan",
            "Switch to read-only plan mode or plan a task.",
            "/plan [task|execute|cancel]",
            handle_plan,
        ),
        SlashCommand(
            "task",
            "Start, inspect, cancel, or complete a sustained task.",
            "/task <objective|status|cancel|complete>",
            handle_task,
        ),
        SlashCommand(
            "goal",
            "Start, inspect, cancel, or complete a sustained goal.",
            "/goal <objective|status|cancel|complete>",
            handle_goal,
        ),
        SlashCommand("skills", "List installed skills.", "/skills", handle_skills),
        SlashCommand("skill", "Show a skill by name.", "/skill:<name>", handle_skill),
        SlashCommand("help", "Show slash commands.", "/help", handle_help),
        SlashCommand("quit", "Exit the interactive session.", "/quit", handle_quit),
    ]


class SlashCommandRegistry:
    def __init__(self, extensions: ExtensionManager | None = None) -> None:
        self.extensions = extensions
        self._commands = build_core_commands()
        self._by_name = {cmd.name: cmd for cmd in self._commands}

    def list_commands(self) -> list[SlashCommand]:
        commands = list(self._commands)
        extensions = self.extensions
        if extensions:
            for name, ext_cmd in sorted(extensions.commands().items()):
                if name in self._by_name:
                    continue

                async def _ext_handler(
                    ctx: CommandContext, _name: str = name
                ) -> CommandResult:
                    # Extension handlers may expect a legacy console context; adapters can
                    # supply extras["extension_context"]. Prefer structured text when possible.
                    extension_context = ctx.extras.get("extension_context", ctx)
                    result = await extensions.handle_command(
                        _name, ctx.args, extension_context
                    )
                    if result is None:
                        return CommandResult()
                    return ok(TextView(str(result)))

                commands.append(
                    SlashCommand(
                        name,
                        ext_cmd.description or "Extension command.",
                        f"/{name}",
                        _ext_handler,
                    )
                )
        return commands

    def completion_items(self) -> list[tuple[str, str, str]]:
        items = [
            (cmd.trigger, cmd.usage, cmd.description) for cmd in self.list_commands()
        ]
        items.append(("/exit", "/exit", "Exit the interactive session."))
        return items

    async def execute(self, raw: str, ctx: CommandContext) -> CommandResult:
        message = raw.strip()
        if not message.startswith("/"):
            return CommandResult(handled=False)

        if message in {"/exit", "/quit"}:
            return await handle_quit(ctx)

        # /skill:name special form
        if message.startswith("/skill:"):
            skill_ctx = CommandContext(
                session=ctx.session,
                cwd=ctx.cwd,
                port=ctx.port,
                raw=message,
                args=message.removeprefix("/skill:").strip(),
                confirm_policy=ctx.confirm_policy,
                extras={**ctx.extras, "registry": self},
            )
            return await handle_skill(skill_ctx)

        body = message[1:]
        name, sep, args = body.partition(" ")
        name = name.strip()
        args = args.strip() if sep else ""

        command = self._by_name.get(name)
        if command is None and self.extensions and name in self.extensions.commands():
            for cmd in self.list_commands():
                if cmd.name == name:
                    command = cmd
                    break

        if command is None:
            return CommandResult(
                views=[
                    TextView(f"Unknown command: {message}", "error"),
                    TextView("Run /help to see available commands.", "plain"),
                ]
            )

        handler_ctx = CommandContext(
            session=ctx.session,
            cwd=ctx.cwd,
            port=ctx.port,
            raw=message,
            args=args,
            confirm_policy=ctx.confirm_policy,
            extras={**ctx.extras, "registry": self},
        )
        return await command.handler(handler_ctx)
