from __future__ import annotations

import json

import pytest

from spice.mcp import config as mcp_config


def test_project_config_overrides_global_and_expands_secret(tmp_path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings.write_text(json.dumps({"mcpServers": {"shared": {"url": "https://global.example/mcp"}}}))
    secrets.write_text(json.dumps({"MCP_TOKEN": "secret-value"}))
    (workspace / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "shared": {
                        "command": "server",
                        "args": ["--stdio"],
                        "env": {"TOKEN": "${MCP_TOKEN}"},
                        "tools": {"include": ["read"]},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(mcp_config, "SETTINGS_PATH", settings)
    monkeypatch.setattr(mcp_config, "SECRETS_PATH", secrets)

    result = mcp_config.load_mcp_config(workspace)

    server = result.servers["shared"]
    assert server.source == "project"
    assert server.transport == "stdio"
    assert server.env == {"TOKEN": "secret-value"}
    assert server.tool_filter.allows("read")
    assert not server.tool_filter.allows("write")


def test_invalid_transport_and_missing_variable_are_server_scoped(tmp_path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    settings.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "both": {"command": "x", "url": "https://example.com/mcp"},
                    "missing": {"url": "https://example.com/${NOT_CONFIGURED}"},
                }
            }
        )
    )
    secrets.write_text("{}")
    monkeypatch.delenv("NOT_CONFIGURED", raising=False)
    monkeypatch.setattr(mcp_config, "SETTINGS_PATH", settings)
    monkeypatch.setattr(mcp_config, "SECRETS_PATH", secrets)

    result = mcp_config.load_mcp_config(tmp_path)

    assert not result.servers
    assert any("both:" in error for error in result.errors)
    assert any("NOT_CONFIGURED" in error for error in result.errors)


@pytest.mark.parametrize(
    ("tools", "allowed", "denied"),
    [
        ({"exclude": ["delete"]}, "read", "delete"),
        ({"include": ["read"], "exclude": ["read"]}, "read", "write"),
    ],
)
def test_tool_filter_semantics(tmp_path, monkeypatch, tools, allowed, denied) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    settings.write_text(json.dumps({"mcpServers": {"one": {"url": "https://example.com/mcp", "tools": tools}}}))
    secrets.write_text("{}")
    monkeypatch.setattr(mcp_config, "SETTINGS_PATH", settings)
    monkeypatch.setattr(mcp_config, "SECRETS_PATH", secrets)

    server = mcp_config.load_mcp_config(tmp_path).servers["one"]

    assert server.tool_filter.allows(allowed)
    assert not server.tool_filter.allows(denied)
