from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spice.llm.model_registry import ModelRegistry
from spice.llm.provider_registry import get_provider


class ModelRegistryTests(unittest.TestCase):
    def test_current_anthropic_models_are_built_in(self) -> None:
        registry = ModelRegistry()

        self.assertIsNotNone(registry.find("anthropic", "claude-haiku-4-5"))
        self.assertIsNotNone(registry.find("anthropic", "claude-sonnet-4-6"))
        self.assertIsNotNone(registry.find("anthropic", "claude-opus-4-8"))

    def test_models_settings_json_registers_profiles_by_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "deepseek-flash": {
                                "provider": "deepseek",
                                "model": "deepseek-v4-flash",
                                "baseUrl": "https://api.deepseek.com",
                                "protocol": "openai-completions",
                                "contextWindow": 1000000,
                                "outputTokens": 32000,
                                "temperature": 0.1,
                            },
                            "deepseek-pro": {
                                "provider": "deepseek",
                                "model": "deepseek-v4-pro",
                                "baseUrl": "https://api.deepseek.com",
                                "protocol": "openai-completions",
                                "contextWindow": 1000000,
                                "outputTokens": 60000,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry(path)

            flash = registry.find(None, "deepseek-flash")
            pro = registry.find(None, "deepseek-pro")

            self.assertIsNotNone(flash)
            self.assertIsNotNone(pro)
            self.assertEqual(flash.id, "deepseek-v4-flash")
            self.assertEqual(flash.profile_key, "deepseek-flash")
            self.assertEqual(flash.context_window, 1000000)
            self.assertEqual(flash.output_tokens, 32000)
            self.assertEqual(flash.temperature, 0.1)
            self.assertEqual(pro.output_tokens, 60000)

    def test_deepseek_provider_uses_openai_compatible_base_url(self) -> None:
        registry = ModelRegistry()
        model = registry.find("deepseek", "deepseek-v4-pro")
        provider = get_provider("deepseek", protocol=model.protocol, model=model)

        self.assertIsNotNone(provider)
        self.assertEqual(getattr(provider, "default_base_url", None), "https://api.deepseek.com")
        self.assertFalse(getattr(provider, "use_responses", False))

    def test_custom_anthropic_profile_uses_configured_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "work-claude": {
                                "provider": "anthropic",
                                "model": "claude-sonnet-4-5",
                                "baseUrl": "https://anthropic-proxy.example.com",
                                "protocol": "anthropic-messages",
                                "outputTokens": 1234,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry(path)
            model = registry.find(None, "work-claude")
            provider = get_provider("anthropic", protocol=model.protocol, model=model)

            self.assertIsNotNone(provider)
            self.assertEqual(model.base_url, "https://anthropic-proxy.example.com")
            self.assertEqual(getattr(provider, "default_base_url", None), "https://anthropic-proxy.example.com")

    def test_official_openai_provider_uses_responses_api(self) -> None:
        registry = ModelRegistry()
        model = registry.find("openai", "gpt-5.1")
        provider = get_provider("openai", protocol=model.protocol, model=model)

        self.assertIsNotNone(provider)
        self.assertTrue(getattr(provider, "use_responses", False))

    def test_custom_openai_compatible_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "local-qwen": {
                                "provider": "local-openai",
                                "model": "qwen-coder",
                                "name": "Qwen Coder",
                                "baseUrl": "http://localhost:1234/v1",
                                "protocol": "openai-completions",
                                "contextWindow": 64000,
                                "outputTokens": 4096,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry(path)
            model = registry.find(None, "local-qwen")
            provider = get_provider("local-openai", protocol=model.protocol, model=model)

            self.assertEqual(model.id, "qwen-coder")
            self.assertEqual(model.base_url, "http://localhost:1234/v1")
            self.assertEqual(getattr(provider, "default_base_url", None), "http://localhost:1234/v1")
            self.assertFalse(getattr(provider, "use_responses", False))

    def test_custom_profile_can_define_pricing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "priced": {
                                "provider": "local-openai",
                                "model": "priced-model",
                                "protocol": "openai-completions",
                                "pricing": {
                                    "inputPerMillionUsd": "1.5",
                                    "outputPerMillionUsd": "3",
                                    "cacheReadPerMillionUsd": "0.15",
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            model = ModelRegistry(path).find(None, "priced")

            self.assertEqual(model.pricing.input_per_million_usd, "1.5")
            self.assertEqual(model.pricing.output_per_million_usd, "3")
            self.assertEqual(model.pricing.cache_read_per_million_usd, "0.15")

    def test_profile_can_override_builtin_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "gpt-5.1": {
                                "provider": "openai",
                                "model": "gpt-5.1",
                                "baseUrl": "https://proxy.example.com/v1",
                                "protocol": "openai-completions",
                                "outputTokens": 1234,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry(path)
            model = registry.find("openai", "gpt-5.1")

            self.assertEqual(model.base_url, "https://proxy.example.com/v1")
            self.assertEqual(model.output_tokens, 1234)

    def test_legacy_max_tokens_is_read_as_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "local-qwen": {
                                "provider": "local-openai",
                                "model": "qwen-coder",
                                "protocol": "openai-completions",
                                "contextWindow": 64000,
                                "maxTokens": 4096,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry(path)
            model = registry.find(None, "local-qwen")

            self.assertEqual(model.output_tokens, 4096)

    def test_model_supports_vision_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "vision-model": {
                                "provider": "openai",
                                "model": "vision-model",
                                "protocol": "openai-completions",
                                "contextWindow": 128000,
                                "supportsVision": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry(path)
            model = registry.find("openai", "vision-model")

            self.assertTrue(model.supports_vision)


if __name__ == "__main__":
    unittest.main()
