from spice.interactive.tool_presentation import tool_icon, tool_presentation


def test_builtin_tools_have_distinct_semantic_icons() -> None:
    assert tool_icon("read_file") == "◇"
    assert tool_icon("search_files") == "⌕"
    assert tool_icon("bash") == "⌘"
    assert tool_icon("web_search") == "◎"
    assert tool_icon("update_todo") == "☑"


def test_unknown_extension_tool_has_safe_fallback() -> None:
    presentation = tool_presentation("deploy_preview")

    assert presentation.icon == "·"
    assert presentation.label == "Deploy Preview"
    assert presentation.category == "generic"

