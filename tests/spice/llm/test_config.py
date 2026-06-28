from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spice.llm.config as config_module
from spice.llm.config import SpiceConfig, get_api_key, load_config, load_secrets, save_config, save_secret
from spice.storage.factory import create_long_task_store, create_memory_store, create_session_store
from spice.storage.sqlite_long_task import SqliteLongTaskStore
from spice.storage.sqlite_memory import SqliteMemoryHistoryBackend
from spice.storage.sqlite_sessions import SqliteSessionStore


class ConfigTests(unittest.TestCase):
    def test_load_config_reads_model_settings_from_settings_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "defaultModel": "claude-sonnet",
                        "models": {
                            "claude-sonnet": {
                                "provider": "anthropic",
                                "model": "claude-sonnet-4-5",
                                "protocol": "anthropic-messages",
                                "temperature": 0.7,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertEqual(config.default_model, "claude-sonnet")
            self.assertEqual(config.provider, "anthropic")
            self.assertEqual(config.model, "claude-sonnet-4-5")
            self.assertEqual(config.protocol, "anthropic-messages")
            self.assertEqual(config.temperature, 0.7)
            self.assertFalse(config.memory_enabled)
            self.assertEqual(config.logging_retention_days, 7)

    def test_load_config_does_not_parse_old_provider_model_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "defaultModel": "deepseek/deepseek-v4-pro",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertEqual(config.default_model, "deepseek/deepseek-v4-pro")
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model, "gpt-5.1")
            self.assertEqual(config.temperature, 0.2)

    def test_load_config_reads_debug_trace_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(json.dumps({"debug": {"trace": True}}), encoding="utf-8")

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertTrue(config.debug_trace)

    def test_load_config_reads_memory_enabled_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(json.dumps({"memory": {"enabled": True}}), encoding="utf-8")

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertTrue(config.memory_enabled)

    def test_load_config_reads_storage_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(
                json.dumps({"storage": {"backend": "sqlite", "sqlitePath": str(Path(directory) / "spice.db")}}),
                encoding="utf-8",
            )

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertEqual(config.storage["backend"], "sqlite")
            self.assertEqual(config.storage["sqlitePath"], str(Path(directory) / "spice.db"))

    def test_load_config_rejects_unknown_storage_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(json.dumps({"storage": {"backend": "postgres"}}), encoding="utf-8")

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertEqual(config.storage["backend"], "file")

    def test_create_session_store_uses_sqlite_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SpiceConfig(storage={"backend": "sqlite", "sqlitePath": str(Path(directory) / "spice.db")})

            store = create_session_store(config, cwd=Path(directory))

            self.assertIsInstance(store, SqliteSessionStore)

    def test_create_application_stores_use_sqlite_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SpiceConfig(storage={"backend": "sqlite", "sqlitePath": str(Path(directory) / "spice.db")})

            task_store = create_long_task_store(config)
            memory_store = create_memory_store(config)

            self.assertIsInstance(task_store, SqliteLongTaskStore)
            self.assertIsInstance(memory_store.history_backend, SqliteMemoryHistoryBackend)

    def test_load_config_reads_logging_retention_days_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(json.dumps({"logging": {"retention_days": 14}}), encoding="utf-8")

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                config = load_config()

            self.assertEqual(config.logging_retention_days, 14)

    def test_save_config_writes_models_profiles_and_no_api_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "local-qwen": {
                                "provider": "local-openai",
                                "model": "qwen-coder",
                                "baseUrl": "http://localhost:1234/v1",
                                "protocol": "openai-completions",
                                "context_window": 64000,
                                "output_tokens": 4096,
                                "apiKey": "should-not-persist",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                save_config(
                    SpiceConfig(
                        default_model="local-qwen",
                        provider="local-openai",
                        model="qwen-coder",
                        temperature=0.1,
                        protocol="openai-completions",
                        base_url="http://localhost:1234/v1",
                        model_profiles={
                            "local-qwen": {
                                "provider": "local-openai",
                                "model": "qwen-coder",
                                "baseUrl": "http://localhost:1234/v1",
                                "protocol": "openai-completions",
                                "context_window": 64000,
                                "output_tokens": 4096,
                                "apiKey": "should-not-persist",
                                "temperature": 0.1,
                            }
                        },
                        debug_trace=True,
                        memory_enabled=True,
                        logging_retention_days=14,
                    )
                )

            data = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(data["defaultModel"], "local-qwen")
            self.assertEqual(data["debug"], {"trace": True})
            self.assertEqual(data["memory"], {"enabled": True})
            self.assertEqual(data["logging"], {"retention_days": 14})
            self.assertNotIn("model", data)
            self.assertNotIn("providers", data)
            self.assertEqual(
                data["models"]["local-qwen"],
                {
                    "provider": "local-openai",
                    "model": "qwen-coder",
                    "baseUrl": "http://localhost:1234/v1",
                    "protocol": "openai-completions",
                    "context_window": 64000,
                    "output_tokens": 4096,
                    "temperature": 0.1,
                },
            )

    def test_get_api_key_reads_secret_by_env_name_before_provider_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_path = Path(directory) / "secrets.json"
            secrets_path.write_text(json.dumps({"DEEPSEEK_API_KEY": "from-secret", "deepseek": "from-provider"}), encoding="utf-8")

            with (
                patch.object(config_module, "CONFIG_DIR", Path(directory)),
                patch.object(config_module, "SECRETS_PATH", secrets_path),
                patch.dict("os.environ", {}, clear=True),
            ):
                api_key = get_api_key("deepseek", env_names=["DEEPSEEK_API_KEY"])

            self.assertEqual(api_key, "from-secret")

    def test_save_secret_uses_default_env_name_for_builtin_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_path = Path(directory) / "secrets.json"

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SECRETS_PATH", secrets_path):
                save_secret("deepseek", "secret-value")

            data = json.loads(secrets_path.read_text(encoding="utf-8"))
            self.assertEqual(data, {"DEEPSEEK_API_KEY": "secret-value"})

    def test_save_secret_restricts_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_path = Path(directory) / "secrets.json"

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SECRETS_PATH", secrets_path):
                save_secret("deepseek", "secret-value")

            mode = stat.S_IMODE(secrets_path.stat().st_mode)
            self.assertEqual(mode, 0o600)

    def test_load_config_warns_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            settings_path.write_text("{invalid", encoding="utf-8")

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SETTINGS_PATH", settings_path):
                with self.assertLogs("spice.llm.config", level="WARNING") as logs:
                    loaded = load_config()

            self.assertEqual(loaded.provider, SpiceConfig.provider)
            self.assertIn("config_json_invalid", "\n".join(logs.output))

    def test_load_secrets_repairs_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secrets_path = Path(directory) / "secrets.json"
            secrets_path.write_text(json.dumps({"OPENAI_API_KEY": "secret"}), encoding="utf-8")
            secrets_path.chmod(0o644)

            with patch.object(config_module, "CONFIG_DIR", Path(directory)), patch.object(config_module, "SECRETS_PATH", secrets_path):
                with self.assertLogs("spice.llm.config", level="WARNING") as logs:
                    secrets = load_secrets()

            self.assertEqual(secrets["OPENAI_API_KEY"], "secret")
            self.assertEqual(stat.S_IMODE(secrets_path.stat().st_mode), 0o600)
            self.assertIn("secrets_permissions_restricted", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
