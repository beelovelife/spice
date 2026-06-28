from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spice.llm.config as config_module
import spice.tools.web as web_module
from spice.tools.base import ToolContext
from spice.tools.web import web_search


class FakeAsyncTavilyClient:
    api_keys: list[str | None] = []
    closed = False

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.api_keys.append(api_key)

    async def search(self, **kwargs: object) -> dict:
        return {
            "results": [
                {
                    "title": "Spice result",
                    "url": "https://example.com/spice",
                    "content": f"query={kwargs['query']} max={kwargs['max_results']}",
                }
            ]
        }

    async def close(self) -> None:
        self.closed = True


class WebToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_search_reads_tavily_key_from_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_path = Path(directory) / "secrets.json"
            secrets_path.write_text(json.dumps({"TAVILY_API_KEY": "from-secret"}), encoding="utf-8")
            FakeAsyncTavilyClient.api_keys = []

            with (
                patch.object(config_module, "SECRETS_PATH", secrets_path),
                patch.dict("os.environ", {}, clear=True),
                patch.object(web_module, "AsyncTavilyClient", FakeAsyncTavilyClient),
            ):
                result = await web_search({"query": "spice", "max_results": 1}, ToolContext(cwd=Path(directory)))

        self.assertFalse(result.is_error)
        self.assertEqual(FakeAsyncTavilyClient.api_keys, ["from-secret"])
        self.assertIn("Spice result", result.content)
        self.assertIn("query=spice max=1", result.content)

    async def test_web_search_reports_missing_tavily_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            FakeAsyncTavilyClient.api_keys = []
            with (
                patch.object(config_module, "SECRETS_PATH", Path(directory) / "missing.json"),
                patch.dict("os.environ", {}, clear=True),
                patch.object(web_module, "AsyncTavilyClient", FakeAsyncTavilyClient),
            ):
                result = await web_search({"query": "spice"}, ToolContext(cwd=Path(directory)))

        self.assertTrue(result.is_error)
        self.assertIn("TAVILY_API_KEY", result.content)
        self.assertEqual(FakeAsyncTavilyClient.api_keys, [])


if __name__ == "__main__":
    unittest.main()
