from __future__ import annotations

from spice.llm.config import SpiceConfig
from spice.tools import catalog


def test_tool_catalog_reports_feature_and_dependency_status(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "get_api_key", lambda provider: None)
    config = SpiceConfig(memory_enabled=False, subagents_enabled=False)

    entries = {entry.name: entry for entry in catalog.build_tool_catalog(config)}

    assert entries["read_file"].status == "enabled"
    assert entries["bash"].risk == "write/exec"
    assert entries["bash"].confirmation == "required"
    assert entries["web_search"].status == "unavailable"
    assert "TAVILY_API_KEY" in entries["web_search"].note
    assert entries["memory"].status == "disabled"
    assert entries["spawn_subagents"].status == "disabled"


def test_tool_catalog_reports_browser_dependency(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "browser_dependency_available", lambda: False)
    entries = {entry.name: entry for entry in catalog.build_tool_catalog(SpiceConfig())}
    assert entries["browser_check"].status == "unavailable"
    assert "browser extra" in entries["browser_check"].note
    assert entries["browser_check"].confirmation == "required"
    assert entries["view_image"].status == "enabled"
    assert entries["view_image"].risk == "read-only"


def test_find_tool_entry_normalizes_hyphens(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "get_api_key", lambda provider: "configured")
    entries = catalog.build_tool_catalog(SpiceConfig())

    assert catalog.find_tool_entry(entries, "read-file").name == "read_file"
    assert catalog.find_tool_entry(entries, "missing") is None


def test_tool_summary_uses_first_sentence_and_limit() -> None:
    assert (
        catalog.tool_summary("First sentence. More detail follows.") == "First sentence"
    )
    assert catalog.tool_summary("x" * 100, limit=12) == "xxxxxxxxxxx…"
