from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import StringIO
import unittest

import spice.cli.main as cli_main
import spice.cli.commands as cli_commands
import spice.cli.run_interactive as run_interactive
from spice.cli.commands import InteractiveCommandContext, SlashCommandRegistry
from spice.agent.tool_results import build_tool_result_metadata
from spice.agent.sessions import SessionEntry, SessionInfo
from spice.llm.config import SpiceConfig
from spice.llm.messages import Message
from spice.llm.models import Model
from spice.tools.base import tool_result
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.console import Console
from typer.testing import CliRunner


def _session_info(session_id: str, *, updated_at: str = "2026-06-19T00:00:00+00:00", cwd: str | None = None) -> SessionInfo:
    return SessionInfo(
        id=session_id,
        path=cli_main.Path(f"{session_id}.jsonl"),
        cwd=cwd or str(cli_main.Path.cwd()),
        provider="openai",
        model="gpt-5.1",
        created_at=updated_at,
        updated_at=updated_at,
        leaf_id="leaf",
        message_count=2,
        preview=f"preview {session_id}",
    )


class SlashCommandRegistryTests(unittest.TestCase):
    def test_core_interactive_commands_are_registered(self) -> None:
        registry = SlashCommandRegistry()
        triggers = {command.trigger for command in registry.commands}

        self.assertIn("/models", triggers)
        self.assertIn("/sessions", triggers)
        self.assertIn("/resume", triggers)
        self.assertIn("/clear", triggers)
        self.assertIn("/reset", triggers)
        self.assertIn("/delete", triggers)
        self.assertIn("/history", triggers)
        self.assertIn("/rewind", triggers)
        self.assertIn("/tools", triggers)
        self.assertIn("/settings", triggers)
        self.assertIn("/subagent", triggers)
        self.assertIn("/compact", triggers)
        self.assertIn("/memory", triggers)
        self.assertIn("/plan", triggers)
        self.assertIn("/task", triggers)
        self.assertIn("/goal", triggers)
        self.assertIn("/cost", triggers)
        self.assertIn("/usage", triggers)

    def test_skill_completion_does_not_duplicate_colon_entry(self) -> None:
        completions = list(SlashCommandRegistry().completer().get_completions(Document("/skill"), None))
        self.assertEqual([completion.text for completion in completions], ["/skills", "/skill"])


def test_memory_slash_command_defaults_to_current_project() -> None:
    calls = []

    class FakeMemoryStore:
        pass

    class FakeAgentSession:
        config = SpiceConfig(memory_enabled=True)
        memory_store = FakeMemoryStore()

        async def distill_current_memory(self, *, scope):
            calls.append(scope)
            return {"success": True, "processed": 1, "adds": 1, "replacements": 0, "removals": 0}

    context = InteractiveCommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=FakeAgentSession(),
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(SlashCommandRegistry().execute("/memory", context))

    assert result.handled is True
    assert calls == ["project"]


def test_chat_and_tui_accept_trace_options(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "set_process_title", lambda: None)

    chat = CliRunner().invoke(cli_main.app, ["chat", "--help"])
    tui = CliRunner().invoke(cli_main.app, ["tui", "--help"])

    assert chat.exit_code == 0
    assert "--trace" in chat.output
    assert "--trace-file" in chat.output
    assert tui.exit_code == 0
    assert "--trace" in tui.output
    assert "--trace-file" in tui.output


def test_interactive_entrypoint_uses_async_prompt(monkeypatch) -> None:
    class FakeExtensions:
        errors = []

        def commands(self) -> dict:
            return {}

    class FakeAgentSession:
        session_id = "test-session"
        session_label = "new"
        extensions = FakeExtensions()
        plan_state = type("PlanState", (), {"mode": "edit", "is_plan_mode": False, "steps": []})()

    class FakePromptSession:
        kwargs = None

        def __init__(self, *args, **kwargs) -> None:
            FakePromptSession.kwargs = kwargs
            self.completer = kwargs.get("completer")

        def prompt(self, *args, **kwargs) -> str:
            raise AssertionError("interactive CLI must not call synchronous prompt() inside the async loop")

        async def prompt_async(self, *args, **kwargs) -> str:
            return "/quit"

    monkeypatch.setattr(run_interactive, "AgentSession", lambda **kwargs: FakeAgentSession())
    monkeypatch.setattr(run_interactive, "UpwardCompletionPromptSession", FakePromptSession)
    monkeypatch.setattr(run_interactive, "preserve_cursor_blink", lambda: None)
    monkeypatch.setattr(run_interactive, "print_welcome", lambda console: None)
    monkeypatch.setattr(run_interactive.sys.stdin, "isatty", lambda: True)

    asyncio.run(run_interactive.run_conversation(cli_main.console))

    assert FakePromptSession.kwargs["reserve_space_for_menu"] == run_interactive.COMPLETION_MENU_RESERVED_ROWS
    assert FakePromptSession.kwargs["reserve_space_for_menu"] == 16
    assert FakePromptSession.kwargs["mouse_support"] is run_interactive.has_completions


def test_interactive_cli_adds_blank_line_before_response(monkeypatch) -> None:
    class FakeExtensions:
        errors = []

        def commands(self) -> dict:
            return {}

    class FakeAgentSession:
        session_label = "new"
        extensions = FakeExtensions()
        plan_state = type("PlanState", (), {"mode": "edit", "is_plan_mode": False, "steps": []})()

    class FakePromptSession:
        def __init__(self, *args, **kwargs) -> None:
            self.completer = kwargs.get("completer")
            self.bottom_toolbar = None
            self._messages = iter(["hello", "quit"])

        async def prompt_async(self, *args, **kwargs) -> str:
            return next(self._messages)

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    output_before_response = []

    async def fake_render_prompt(*args, **kwargs) -> bool:
        output_before_response.append(output.getvalue())
        return True

    monkeypatch.setattr(run_interactive, "AgentSession", lambda **kwargs: FakeAgentSession())
    monkeypatch.setattr(run_interactive, "UpwardCompletionPromptSession", FakePromptSession)
    monkeypatch.setattr(run_interactive, "preserve_cursor_blink", lambda: None)
    monkeypatch.setattr(run_interactive, "print_compact_welcome", lambda console: None)
    monkeypatch.setattr(run_interactive, "render_prompt_interruptible", fake_render_prompt)
    monkeypatch.setattr(run_interactive.sys.stdin, "isatty", lambda: True)

    asyncio.run(run_interactive.run_conversation(console))

    assert output_before_response[0].endswith("Session: new\n\n")


def test_interactive_clear_alias_routes_to_clear_command(monkeypatch) -> None:
    class FakeExtensions:
        errors = []

        def commands(self) -> dict:
            return {}

    class FakeAgentSession:
        session_id = "test-session"
        session_label = "new"
        extensions = FakeExtensions()
        plan_state = type("PlanState", (), {"mode": "edit", "is_plan_mode": False})()

    class FakePromptSession:
        def __init__(self, *args, **kwargs) -> None:
            self.completer = kwargs.get("completer")
            self._messages = iter(["clear", "/quit"])

        async def prompt_async(self, *args, **kwargs) -> str:
            return next(self._messages)

    seen = []

    async def fake_execute(self, raw_message, context):
        seen.append(raw_message)
        if raw_message == "/clear":
            return cli_commands.SlashCommandResult(clear_requested=True)
        return cli_commands.SlashCommandResult(exit_requested=True)

    monkeypatch.setattr(run_interactive, "AgentSession", lambda **kwargs: FakeAgentSession())
    monkeypatch.setattr(run_interactive, "UpwardCompletionPromptSession", FakePromptSession)
    monkeypatch.setattr(run_interactive, "preserve_cursor_blink", lambda: None)
    monkeypatch.setattr(run_interactive, "print_welcome", lambda console: None)
    monkeypatch.setattr(cli_commands.SlashCommandRegistry, "execute", fake_execute)
    monkeypatch.setattr(run_interactive.sys.stdin, "isatty", lambda: True)

    asyncio.run(run_interactive.run_conversation(cli_main.console))

    assert seen[0] == "/clear"


def test_config_set_storage_backend(monkeypatch) -> None:
    saved_configs = []
    monkeypatch.setattr(cli_main, "load_config", lambda: SpiceConfig())
    monkeypatch.setattr(cli_main, "save_config", lambda config: saved_configs.append(config))
    monkeypatch.setattr(cli_main.console, "print", lambda *args, **kwargs: None)

    cli_main.config_set("storage.backend", "sqlite")

    assert saved_configs[0].storage["backend"] == "sqlite"


def test_config_set_storage_sqlite_path(monkeypatch) -> None:
    saved_configs = []
    monkeypatch.setattr(cli_main, "load_config", lambda: SpiceConfig())
    monkeypatch.setattr(cli_main, "save_config", lambda config: saved_configs.append(config))
    monkeypatch.setattr(cli_main.console, "print", lambda *args, **kwargs: None)

    cli_main.config_set("storage.sqlitePath", "/tmp/spice.db")

    assert saved_configs[0].storage["sqlitePath"] == "/tmp/spice.db"


def test_config_get_storage_backend(monkeypatch) -> None:
    printed = []
    monkeypatch.setattr(cli_main, "load_config", lambda: SpiceConfig(storage={"backend": "sqlite", "sqlitePath": "x.db"}))
    monkeypatch.setattr(cli_main.console, "print", lambda value, *args, **kwargs: printed.append(value))

    cli_main.config_get("storage.backend")

    assert printed == ["sqlite"]


def test_clear_slash_command_requests_visible_clear() -> None:
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=object(),
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/clear", context))

    assert result.clear_requested
    assert result.session is None


def test_plan_slash_command_enters_plan_mode_and_returns_prompt() -> None:
    class FakeSession:
        def __init__(self):
            self.started = None
            self.plan_state = type("PlanState", (), {"mode": "edit"})()

        def start_plan(self, objective=""):
            self.started = objective
            self.plan_state.mode = "plan"

    session = FakeSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/plan implement mode", context))

    assert session.started == "implement mode"
    assert result.prompt == "implement mode"


def test_task_slash_command_starts_sustained_goal_prompt() -> None:
    class FakeSession:
        def __init__(self):
            self.mode = None
            self.started = None

        def set_interaction_mode(self, mode):
            self.mode = mode

        def start_long_task(self, objective):
            self.started = objective

    session = FakeSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/task restore spice", context))

    assert session.mode == "edit"
    assert session.started == "restore spice"
    assert result.prompt is not None
    assert "First call long_task" not in result.prompt
    assert "call long_task" not in result.prompt
    assert "complete_long_task" in result.prompt
    assert "not read-only planning mode" in result.prompt


def test_task_slash_subcommands_do_not_start_new_task() -> None:
    class FakeState:
        objective = "restore spice"
        task_id = "lt_1"
        status = "active"
        continuation_rounds = 1
        max_continuation_rounds = 12
        remaining_continuations = 11
        needs_user_attention = False
        completion_candidate = False
        last_stop_reason = ""

    class FakeSession:
        def __init__(self):
            self.started = None
            self.completed = None

        def long_task_status(self):
            return FakeState()

        def complete_long_task(self, *, note="", force=False):
            self.completed = (note, force)
            return FakeState()

        def cancel_long_task(self, *, note=""):
            return FakeState()

    session = FakeSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=session,
        cwd=cli_main.Path.cwd(),
    )

    status_result = asyncio.run(registry.execute("/task status", context))
    complete_result = asyncio.run(registry.execute("/task complete --force verified", context))

    assert status_result.prompt is None
    assert complete_result.prompt is None
    assert session.started is None
    assert session.completed == ("verified", True)


def test_bottom_toolbar_shows_plan_or_transient_edit_hint() -> None:
    class FakeSession:
        def __init__(self, mode):
            self.plan_state = type("PlanState", (), {"mode": mode})()

    assert run_interactive._bottom_toolbar(FakeSession("edit"), object(), show_edit_hint=False) is None
    assert run_interactive._bottom_toolbar(FakeSession("plan"), object(), show_edit_hint=False) is not None
    assert run_interactive._bottom_toolbar(FakeSession("edit"), object(), show_edit_hint=True) is not None


def test_interactive_choice_picker_uses_taller_scrollable_window(monkeypatch) -> None:
    captured = {}

    class FakeApplication:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run_async(self):
            return None

    monkeypatch.setattr(cli_commands, "Application", FakeApplication)
    choices = [(f"value-{index}", f"Label {index}", f"Description {index}") for index in range(40)]

    result = asyncio.run(cli_commands._select_interactive(title="Model", choices=choices))

    picker_window = captured["layout"].container.children[0]
    assert result is None
    assert captured["mouse_support"] is True
    assert picker_window.height.max == cli_commands.CHOICE_PICKER_MAX_HEIGHT
    assert picker_window.right_margins


def test_interactive_choice_picker_scrolls_from_any_visible_row(monkeypatch) -> None:
    captured = {}
    invalidated = []

    class FakeApplication:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run_async(self):
            return None

    class FakeApp:
        def invalidate(self) -> None:
            invalidated.append(True)

    monkeypatch.setattr(cli_commands, "Application", FakeApplication)
    monkeypatch.setattr(cli_commands, "get_app", lambda: FakeApp())
    choices = [(f"value-{index}", f"Label {index}", f"Description {index}") for index in range(40)]

    asyncio.run(cli_commands._select_interactive(title="Session", choices=choices, current_value="value-20"))

    control = captured["layout"].container.children[0].content
    before = control.get_cursor_position().y
    result = control.mouse_handler(
        MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.SCROLL_UP,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
    )

    assert result is None
    assert invalidated
    assert control.get_cursor_position().y != before


def test_reset_slash_command_keeps_session_id_and_clears_messages(monkeypatch) -> None:
    class FakeCurrentSession:
        session = object()
        session_id = "session-1"
        model = Model(id="gpt-4o-mini", provider="openai")

        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    async def fake_select(*args, **kwargs):
        return "yes"

    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)
    current_session = FakeCurrentSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=current_session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/reset", context))

    assert current_session.reset_called
    assert result.clear_requested
    assert result.session is None


def test_reset_slash_command_cancel_keeps_current_session(monkeypatch) -> None:
    class FakeCurrentSession:
        session = object()
        session_id = "session-1"
        model = Model(id="gpt-4o-mini", provider="openai")

        def reset(self):
            raise AssertionError("cancelled reset must not reset the session")

    async def fake_select(*args, **kwargs):
        return "no"

    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=FakeCurrentSession(),
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/reset", context))

    assert not result.clear_requested
    assert result.session is None


def test_delete_current_slash_command_deletes_session_and_starts_fresh(monkeypatch) -> None:
    class FakeStore:
        def __init__(self):
            self.deleted = []

        def delete(self, session_id):
            self.deleted.append(session_id)

    class FakeCurrentSession:
        session = object()
        session_id = "session-1"
        session_store = FakeStore()
        model = Model(id="gpt-4o-mini", provider="openai")
        extensions = object()

    class FakeNewSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.extensions = kwargs["extension_manager"]

    async def fake_select(*args, **kwargs):
        return "yes"

    import spice.interactive.sessions as interactive_sessions

    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)
    monkeypatch.setattr(interactive_sessions, "AgentSession", FakeNewSession)
    current_session = FakeCurrentSession()
    current_session.confirm = None
    registry = SlashCommandRegistry()
    renderer = type("Renderer", (), {"confirm": None})()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=renderer,
        agent_session=current_session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/delete current", context))

    assert current_session.session_store.deleted == ["session-1"]
    assert result.clear_requested
    assert isinstance(result.session, FakeNewSession)
    assert result.session.kwargs["provider"] == "openai"
    assert result.session.kwargs["model_id"] == "gpt-4o-mini"
    assert result.session.kwargs["confirm"] is None
    assert result.session.kwargs["session_store"] is current_session.session_store
    assert result.session.kwargs["extension_manager"] is current_session.extensions


def test_delete_specific_slash_command_keeps_current_chat(monkeypatch) -> None:
    class FakeResolved:
        id = "session-2"

    class FakeStore:
        def __init__(self):
            self.deleted = []

        def resolve(self, session_id, cwd=None):
            assert session_id == "session"
            return FakeResolved()

        def delete(self, session_id):
            self.deleted.append(session_id)

    class FakeCurrentSession:
        session = object()
        session_id = "session-1"
        session_store = FakeStore()
        model = Model(id="gpt-4o-mini", provider="openai")
        extensions = object()

    async def fake_select(*args, **kwargs):
        return "yes"

    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)
    current_session = FakeCurrentSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=current_session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/delete session", context))

    assert current_session.session_store.deleted == ["session-2"]
    assert not result.clear_requested
    assert result.session is None


def test_resume_slash_command_uses_current_model(monkeypatch) -> None:
    @dataclass
    class FakeRow:
        id: str = "session-1"
        updated_at: str = "now"
        provider: str = "openai"
        model: str = "gpt-4o-mini"
        message_count: int = 1
        preview: str = "hello"

    class FakeStore:
        def list(self, *, limit, cwd, include_empty=False):
            assert include_empty is True
            return [FakeRow()]

    class FakeCurrentSession:
        model = Model(id="claude-sonnet-4-5", provider="anthropic")

    created_kwargs = {}

    class FakeNewSession:
        session_id = "session-1"
        messages = []

        def __init__(self, **kwargs) -> None:
            created_kwargs.update(kwargs)

    async def fake_select(*args, **kwargs):
        return "session-1"

    import spice.interactive.sessions as interactive_sessions

    monkeypatch.setattr(interactive_sessions, "create_session_store_for_config", lambda **_kwargs: FakeStore())
    monkeypatch.setattr(interactive_sessions, "AgentSession", FakeNewSession)
    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)

    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=FakeCurrentSession(),
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/resume", context))

    assert result.session is not None
    assert created_kwargs["provider"] == "anthropic"
    assert created_kwargs["model_id"] == "claude-sonnet-4-5"


def test_sessions_slash_command_includes_reset_empty_sessions(monkeypatch) -> None:
    @dataclass
    class FakeRow:
        id: str = "session-1"
        updated_at: str = "now"
        provider: str = "openai"
        model: str = "gpt-4o-mini"
        message_count: int = 0
        preview: str = ""

    calls = []

    class FakeStore:
        def list(self, *, limit, cwd, include_empty=False):
            calls.append(include_empty)
            return [FakeRow()]

    class FakeCurrentSession:
        model = Model(id="gpt-4o-mini", provider="openai")

    class FakeNewSession:
        session_id = "session-1"
        messages = []

        def __init__(self, **kwargs) -> None:
            pass

    async def fake_select(*args, **kwargs):
        return None

    import spice.interactive.sessions as interactive_sessions

    monkeypatch.setattr(interactive_sessions, "create_session_store_for_config", lambda **_kwargs: FakeStore())
    monkeypatch.setattr(interactive_sessions, "AgentSession", FakeNewSession)
    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)

    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=FakeCurrentSession(),
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/sessions", context))

    assert result.handled
    assert calls == [True]


def test_models_slash_command_keeps_current_session_reference(monkeypatch) -> None:
    class FakeRegistry:
        model = Model(id="deepseek-v4-pro", provider="deepseek", temperature=0.1)

        def all(self):
            return [self.model]

        def find(self, provider, model_id):
            if provider == self.model.provider and model_id == self.model.id:
                return self.model
            return None

    class FakeCurrentSession:
        model = Model(id="gpt-5.1", provider="openai")
        extensions = type("Extensions", (), {"commands": lambda self: {}})()

        def __init__(self):
            self.selected_model = None

        def set_model(self, model):
            self.selected_model = model
            self.model = model

    class FakeConsole:
        def print(self, *args, **kwargs):
            pass

    async def fake_select(*args, **kwargs):
        return "deepseek/deepseek-v4-pro"

    import spice.interactive.commands as interactive_commands
    import spice.interactive.sessions as interactive_sessions

    saved_configs = []
    monkeypatch.setattr(interactive_commands, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(interactive_sessions, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)
    monkeypatch.setattr(interactive_commands, "load_config", lambda: SpiceConfig(provider="openai", model="gpt-5.1"))
    monkeypatch.setattr(interactive_commands, "save_config", lambda config: saved_configs.append(config))
    monkeypatch.setattr(interactive_commands, "get_api_key", lambda *args, **kwargs: "key")
    monkeypatch.setattr("spice.llm.config.get_api_key", lambda *args, **kwargs: "key")

    current_session = FakeCurrentSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=FakeConsole(),
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=current_session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/models", context))

    assert result.session is None
    assert current_session.selected_model == FakeRegistry.model
    assert saved_configs[0].default_model == "deepseek-v4-pro"
    assert saved_configs[0].provider == "deepseek"
    assert saved_configs[0].model == "deepseek-v4-pro"
    assert saved_configs[0].temperature == 0.1


def test_rewind_slash_command_selects_entry_when_id_is_omitted(monkeypatch) -> None:
    class FakeStore:
        def info(self, session_id):
            return SessionInfo(
                id=session_id,
                path=cli_main.Path("session.jsonl"),
                cwd=str(cli_main.Path.cwd()),
                provider="openai",
                model="gpt-4o-mini",
                created_at="",
                updated_at="",
                leaf_id="a1",
            )

        def path_entries(self, session_id):
            return [SessionEntry(id="u1", parent_id=None, timestamp="", type="message", data={"type": "message", "message": {"role": "user", "content": "hello"}})]

        def entries(self, session_id):
            return [
                SessionEntry(id="u1", parent_id=None, timestamp="", type="message", data={"type": "message", "message": {"role": "user", "content": "hello"}}),
                SessionEntry(id="a1", parent_id="u1", timestamp="", type="message", data={"type": "message", "message": {"role": "assistant", "content": "hi"}}),
                SessionEntry(id="leaf1", parent_id=None, timestamp="", type="leaf", data={"type": "leaf", "targetId": "a1"}),
            ]

    class FakeCurrentSession:
        session = object()
        session_id = "session-1"
        session_store = FakeStore()
        model = Model(id="gpt-4o-mini", provider="openai")

        def __init__(self):
            self.rewound_to = None

        def rewind(self, entry_id):
            self.rewound_to = entry_id

    async def fake_select(*args, **kwargs):
        return "u1"

    monkeypatch.setattr(cli_commands, "_select_from_choices", fake_select)
    current_session = FakeCurrentSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=current_session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/rewind", context))

    assert result.session is None
    assert current_session.rewound_to == "u1"


def test_rewind_slash_command_uses_entry_id_directly(monkeypatch) -> None:
    async def fail_select(*args, **kwargs):
        raise AssertionError("/rewind <entry-id> should not prompt")

    class FakeCurrentSession:
        session = object()
        session_id = "session-1"
        model = Model(id="gpt-4o-mini", provider="openai")

        def __init__(self):
            self.rewound_to = None

        def rewind(self, entry_id):
            self.rewound_to = entry_id

    monkeypatch.setattr(cli_commands, "_select_from_choices", fail_select)
    current_session = FakeCurrentSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=cli_main.console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=current_session,
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/rewind a1", context))

    assert result.session is None
    assert current_session.rewound_to == "a1"


def test_settings_slash_command_prints_with_context_console() -> None:
    class FakeConsole:
        def __init__(self):
            self.items = []

        def print(self, item):
            self.items.append(item)

    class FakeAgentSession:
        session_label = "new"
        model = Model(id="gpt-4o-mini", provider="openai")

        def get_active_tools(self):
            return ["read_file", "write_file"]

    fake_console = FakeConsole()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=fake_console,
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=FakeAgentSession(),
        cwd=cli_main.Path.cwd(),
    )

    result = asyncio.run(registry.execute("/settings", context))

    assert result.handled
    assert result.views
    assert fake_console.items == []


def test_subagent_slash_command_toggles_current_session() -> None:
    class FakeConsole:
        def __init__(self):
            self.items = []

        def print(self, item):
            self.items.append(item)

    class FakePlanState:
        mode = "edit"
        is_plan_mode = False

    class FakeManager:
        max_concurrent = 3

    class FakeAgentSession:
        session_label = "new"
        model = Model(id="gpt-4o-mini", provider="openai")
        plan_state = FakePlanState()
        subagent_manager = FakeManager()

        def __init__(self):
            self.subagents_enabled = True

        def set_subagents_enabled(self, enabled):
            self.subagents_enabled = enabled

        def get_active_tools(self):
            tools = ["read_file"]
            if self.subagents_enabled:
                tools.append("spawn_subagents")
            return tools

    session = FakeAgentSession()
    registry = SlashCommandRegistry()
    context = InteractiveCommandContext(
        console=FakeConsole(),
        input_session=None,
        renderer=type("Renderer", (), {"confirm": None})(),
        agent_session=session,
        cwd=cli_main.Path.cwd(),
    )

    off_result = asyncio.run(registry.execute("/subagent off", context))
    assert off_result.session is None
    assert session.subagents_enabled is False
    assert "spawn_subagents" not in session.get_active_tools()

    status_result = asyncio.run(registry.execute("/subagent status", context))
    assert status_result.session is None

    on_result = asyncio.run(registry.execute("/subagent on", context))
    assert on_result.session is None
    assert session.subagents_enabled is True
    assert "spawn_subagents" in session.get_active_tools()


def test_sessions_delete_requires_confirmation(monkeypatch) -> None:
    class FakeStore:
        def __init__(self, **_kwargs):
            self.deleted = []

        def resolve(self, session_id, cwd=None):
            return _session_info(session_id)

        def delete(self, session_id):
            self.deleted.append(session_id)

    stores = []

    def fake_store(_config, **kwargs):
        store = FakeStore(**kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(cli_main, "create_session_store", fake_store)
    monkeypatch.setattr(cli_main.typer, "confirm", lambda *args, **kwargs: False)

    cli_main.sessions_delete("session-1")

    assert stores[0].deleted == []


def test_sessions_delete_yes_skips_confirmation(monkeypatch) -> None:
    class FakeStore:
        def __init__(self, **_kwargs):
            self.deleted = []

        def resolve(self, session_id, cwd=None):
            return _session_info(session_id)

        def delete(self, session_id):
            self.deleted.append(session_id)

    stores = []

    def fake_store(_config, **kwargs):
        store = FakeStore(**kwargs)
        stores.append(store)
        return store

    def fail_confirm(*args, **kwargs):
        raise AssertionError("--yes should not prompt for single-session delete")

    monkeypatch.setattr(cli_main, "create_session_store", fake_store)
    monkeypatch.setattr(cli_main.typer, "confirm", fail_confirm)

    cli_main.sessions_delete("session-1", yes=True)

    assert stores[0].deleted == ["session-1"]


def test_tool_display_for_session_history_uses_message_metadata() -> None:
    message = Message(
        role="tool",
        content="full output",
        name="read_file",
        metadata=build_tool_result_metadata("read_file", {"path": "README.md"}, tool_result("full output")),
    )

    assert cli_main.tool_display_text(message) == "read_file: README.md"


def test_tool_display_without_metadata_falls_back_to_content_preview() -> None:
    message = Message(role="tool", content="full output", name="read_file")

    assert cli_main.tool_display_text(message) == ""


def test_tool_display_ignores_plain_tool_name_summary() -> None:
    message = Message(
        role="tool",
        content="full output",
        name="bash",
        metadata=build_tool_result_metadata("bash", {}, tool_result("full output")),
    )

    assert cli_main.tool_display_text(message) == ""


def test_session_tree_entry_preview_uses_stored_field_names() -> None:
    assert cli_main._entry_preview({"type": "model_change", "provider": "openai", "model": "gpt-5.1"}) == "openai/gpt-5.1"
    assert cli_main._entry_preview({"type": "leaf", "leaf_id": "entry-1"}) == "target=entry-1"


def test_sessions_prune_dry_run_does_not_delete(monkeypatch) -> None:
    class FakeStore:
        base_root = cli_main.Path("/tmp/sessions")

        def __init__(self, **_kwargs):
            self.deleted = []

        def list(self, *, limit, cwd, include_empty=False):
            return [
                _session_info("new", updated_at="2026-06-19T00:00:00+00:00"),
                _session_info("old", updated_at="2026-06-18T00:00:00+00:00"),
            ]

        def delete(self, session_id):
            self.deleted.append(session_id)

    stores = []

    def fake_store(_config, **kwargs):
        store = FakeStore(**kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(cli_main, "create_session_store", fake_store)

    cli_main.sessions_prune(keep_recent=1)

    assert stores[0].deleted == []


def test_sessions_prune_yes_deletes_candidates_after_confirmation(monkeypatch) -> None:
    class FakeStore:
        base_root = cli_main.Path("/tmp/sessions")

        def __init__(self, **_kwargs):
            self.deleted = []

        def list(self, *, limit, cwd, include_empty=False):
            return [
                _session_info("new", updated_at="2026-06-19T00:00:00+00:00"),
                _session_info("old", updated_at="2026-06-18T00:00:00+00:00"),
            ]

        def delete(self, session_id):
            self.deleted.append(session_id)

    stores = []

    def fake_store(_config, **kwargs):
        store = FakeStore(**kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(cli_main, "create_session_store", fake_store)
    monkeypatch.setattr(cli_main.typer, "confirm", lambda *args, **kwargs: True)

    cli_main.sessions_prune(keep_recent=1, yes=True)

    assert stores[0].deleted == ["old"]


if __name__ == "__main__":
    unittest.main()
