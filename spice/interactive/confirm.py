"""Per-session tool confirmation policy and pure rule helpers."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from spice.interactive.types import ConfirmChoice, ConfirmRequest, InteractivePort
from spice.tools.base import ConfirmFn

if TYPE_CHECKING:
    pass

FILE_EDIT_TOOLS = frozenset({"write_file", "edit_file", "apply_patch"})


def is_file_edit_tool(tool_name: str) -> bool:
    return tool_name in FILE_EDIT_TOOLS


def confirmation_question(tool_name: str, args: dict) -> str:
    path = str(args.get("path") or "").strip()
    target = path or "the target file"
    if tool_name == "write_file":
        return f"Do you want to write {target}?"
    if tool_name == "edit_file":
        return f"Do you want to edit {target}?"
    if tool_name == "bash":
        if bash_may_modify_files(args):
            return (
                "Do you want to run this command? It may modify files; "
                "prefer edit_file or apply_patch when possible."
            )
        return "Do you want to run this command?"
    return f"Do you want to run {tool_name}?"


def bash_may_modify_files(args: dict) -> bool:
    command = str(args.get("command") or "").strip().lower()
    if not command:
        return False
    write_markers = (
        "sed -i",
        "perl -pi",
        "python -c",
        "python3 -c",
        " write_text(",
        ".write(",
        "open(",
        ">>",
        ">",
    )
    if not any(marker in command for marker in write_markers):
        return False
    return any(
        marker in command
        for marker in (
            "open(",
            ".write(",
            "write_text(",
            "sed -i",
            "perl -pi",
            ">>",
            ">",
            "mode='w'",
            'mode="w"',
            "', 'w'",
            '", "w"',
            "', 'a'",
            '", "a"',
        )
    )


def is_read_only_bash_command(args: dict) -> bool:
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    try:
        parts = _shell_command_tokens(command)
    except ValueError:
        return False
    if not parts:
        return False
    if any(part in {";", "&", "||", ">", ">>", "<", "<<", "<<<"} for part in parts):
        return False
    for command_parts in _split_shell_commands(parts):
        if not command_parts:
            return False
        program = command_parts[0].rsplit("/", 1)[-1]
        if program == "cd":
            if not _cd_stays_in_workspace(command_parts):
                return False
            continue
        if not _bash_paths_stay_in_workspace(command_parts):
            return False
        if not _is_read_only_program_call(command_parts):
            return False
    return True


def _shell_command_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _split_shell_commands(parts: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    current: list[str] = []
    for part in parts:
        if part in {"&&", "|"}:
            commands.append(current)
            current = []
        else:
            current.append(part)
    commands.append(current)
    return commands


def _is_read_only_program_call(parts: list[str]) -> bool:
    program = parts[0].rsplit("/", 1)[-1]
    if program in {"rg", "grep", "cat", "head", "tail", "ls", "pwd", "wc", "echo"}:
        return True
    if program == "sort":
        return not any(
            part == "-o" or part == "--output" or part.startswith("--output=")
            for part in parts[1:]
        )
    if program == "uniq":
        # POSIX uniq accepts INPUT then OUTPUT; a second positional operand writes.
        positional = [part for part in parts[1:] if not part.startswith("-")]
        return len(positional) <= 1
    if program == "awk":
        script = " ".join(parts[1:]).lower()
        return "system(" not in script and ">" not in script and "|" not in script
    if program == "sed":
        return not any(
            part == "-i" or part.startswith("-i") or part == "--in-place"
            for part in parts[1:]
        )
    if program == "find":
        forbidden = {
            "-delete",
            "-exec",
            "-execdir",
            "-ok",
            "-okdir",
            "-fprint",
            "-fprint0",
            "-fprintf",
            "-fls",
        }
        return not any(part in forbidden for part in parts[1:])
    if program == "git":
        if len(parts) < 2:
            return False
        subcommand = parts[1]
        subcommand_args = parts[2:]
        if any(
            part == "--output" or part.startswith("--output=")
            for part in subcommand_args
        ):
            return False
        if subcommand == "branch":
            return not subcommand_args or all(
                part
                in {
                    "--all",
                    "--list",
                    "--remotes",
                    "--show-current",
                    "--verbose",
                    "-a",
                    "-l",
                    "-r",
                    "-v",
                    "-vv",
                }
                for part in subcommand_args
            )
        if subcommand == "remote":
            return not subcommand_args or subcommand_args in (["-v"], ["--verbose"])
        return subcommand in {
            "describe",
            "diff",
            "grep",
            "log",
            "ls-files",
            "rev-parse",
            "show",
            "status",
        }
    return False


def _cd_stays_in_workspace(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    return _bash_paths_stay_in_workspace(parts)


def _bash_paths_stay_in_workspace(parts: list[str]) -> bool:
    cwd = Path.cwd().resolve()
    for token in parts[1:]:
        if not token or token.startswith("-"):
            continue
        if token in {".", "./"}:
            continue
        if ".." in Path(token).parts:
            return False
        path = Path(token).expanduser()
        if path.is_absolute():
            try:
                resolved = path.resolve()
            except OSError:
                return False
            if resolved != cwd and cwd not in resolved.parents:
                return False
    return True


def build_tool_confirm_request(tool_name: str, args: dict) -> ConfirmRequest:
    choices: list[ConfirmChoice] = [
        ConfirmChoice("allow", "Yes", "allow once"),
        ConfirmChoice(
            "allow_all_tools",
            "Yes, allow all tools during this session",
            "no more permission prompts",
        ),
    ]
    if is_file_edit_tool(tool_name):
        choices.append(
            ConfirmChoice(
                "allow_all_edits", "Yes, allow all edits during this session", ""
            ),
        )
    if tool_name == "bash" and is_read_only_bash_command(args):
        choices.append(
            ConfirmChoice(
                "allow_read_only_bash",
                "Yes, allow read-only shell commands this session",
                "grep/sed/rg/cat/ls",
            ),
        )
    choices.append(ConfirmChoice("deny", "No", "do not run this tool"))
    return ConfirmRequest(
        question=confirmation_question(tool_name, args),
        choices=tuple(choices),
        current_id="allow",
    )


class ConfirmPolicy:
    """Session-scoped confirmation state. One instance per AgentSession."""

    def __init__(self) -> None:
        self.allow_all_tools = False
        self.allow_file_edits = False
        self.allow_read_only_bash = False
        self._lock = asyncio.Lock()

    def precheck(self, tool_name: str, args: dict) -> bool | None:
        """True = allow, False = deny, None = ask UI."""
        if self.allow_all_tools:
            return True
        if is_file_edit_tool(tool_name) and self.allow_file_edits:
            return True
        if (
            tool_name == "bash"
            and self.allow_read_only_bash
            and is_read_only_bash_command(args)
        ):
            return True
        return None

    def apply_decision(self, decision: str) -> None:
        if decision == "allow_all_tools":
            self.allow_all_tools = True
        elif decision == "allow_all_edits":
            self.allow_file_edits = True
        elif decision == "allow_read_only_bash":
            self.allow_read_only_bash = True

    async def confirm(self, tool_name: str, args: dict, port: InteractivePort) -> bool:
        async with self._lock:
            pre = self.precheck(tool_name, args)
            if pre is not None:
                return pre
            request = build_tool_confirm_request(tool_name, args)
            decision = await port.confirm(request)
            if not decision or decision == "deny":
                return False
            self.apply_decision(decision)
            return decision in {
                "allow",
                "allow_all_tools",
                "allow_all_edits",
                "allow_read_only_bash",
            }

    def as_confirm_fn(self, port: InteractivePort) -> ConfirmFn:
        async def _confirm(tool_name: str, args: dict) -> bool:
            return await self.confirm(tool_name, args, port)

        return _confirm
