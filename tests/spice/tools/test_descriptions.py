from __future__ import annotations

from spice.tools.bash import create_bash_tools
from spice.tools.file import create_file_tools


def _tools_by_name():
    return {tool.name: tool for tool in [*create_file_tools(), *create_bash_tools()]}


def test_file_tool_descriptions_route_common_file_tasks_away_from_bash() -> None:
    tools = _tools_by_name()

    assert "instead of bash ls" in tools["list_dir"].description
    assert "instead of bash cat/head/tail/sed" in tools["read_file"].description
    assert "multi-file analysis" in tools["read_files"].description
    assert "instead of bash grep/rg/find" in tools["search_files"].description


def test_write_file_description_requires_explicit_save_intent() -> None:
    write_file = _tools_by_name()["write_file"]

    assert "explicitly asks to save/create/overwrite a file" in write_file.description
    assert "Before overwriting an existing file, first read it with read_file/read_files" in write_file.description
    assert "Do not use for pure examples" in write_file.description
    assert "use edit_file for targeted changes" in write_file.description


def test_editing_descriptions_tell_model_to_read_existing_files_first() -> None:
    tools = _tools_by_name()

    assert "Read the target file first with read_file/read_files" in tools["edit_file"].description
    assert "Before replace/delete operations on existing files, read the target files first" in tools["apply_patch"].description


def test_bash_description_reserves_shell_for_shell_tasks() -> None:
    bash = _tools_by_name()["bash"]

    assert "Use for builds, tests, package managers, git, scripts, processes" in bash.description
    assert "complex shell pipelines" in bash.description
    assert "For simple file reading, listing, or searching, prefer read_file, list_dir, or search_files" in bash.description
    assert "prefer edit_file or apply_patch over shell scripts" in bash.description
