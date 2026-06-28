"""Spice configuration and secrets."""

from __future__ import annotations

import copy
import json
import logging
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".spice"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
SECRETS_PATH = CONFIG_DIR / "secrets.json"
logger = logging.getLogger(__name__)

ENV_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "tavily": ["TAVILY_API_KEY"],
    "brave": ["BRAVE_SEARCH_API_KEY"],
}


DEFAULT_SANDBOX_CONFIG: dict[str, Any] = {
    "mode": "workspace",
    "local": {
        "requires_confirmation": True,
    },
    "workspace": {
        "root": ".",
        "restrict": True,
        "max_write_bytes": 1_000_000,
        "protected_write": [".spice/**", ".git/**"],
        "secret_paths": [
            ".env",
            ".env.*",
            "**/*.pem",
            "**/*.key",
            "**/id_rsa",
            "**/id_ed25519",
        ],
    },
    "docker": {
        "image": "spice-sandbox:latest",
        "container_name": "",
        "container_workspace": "/workspace",
        "network": False,
        "memory": "2g",
        "cpus": 2,
        "pids_limit": 256,
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
        "persist": True,
        "fallback_to_local": False,
    },
}

DEFAULT_STORAGE_CONFIG: dict[str, Any] = {
    "backend": "file",
    "sqlitePath": "~/.spice/spice.db",
}


@dataclass
class SpiceConfig:
    default_model: str = "gpt-5.1"
    provider: str = "openai"
    model: str = "gpt-5.1"
    temperature: float = 0.2
    protocol: str | None = "openai-completions"
    base_url: str | None = None
    model_profiles: dict[str, Any] = field(default_factory=dict)
    debug_trace: bool = False
    memory_enabled: bool = False
    subagents_enabled: bool = True
    max_concurrent_subagents: int = 3
    logging_retention_days: int = 7
    sandbox: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_SANDBOX_CONFIG))
    storage: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_STORAGE_CONFIG))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "config_json_invalid path=%s line=%d column=%d error=%s",
            path,
            exc.lineno,
            exc.colno,
            exc.msg,
        )
        return {}
    except OSError as exc:
        logger.warning("config_read_failed path=%s error=%s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("config_json_ignored path=%s reason=top-level value is not an object", path)
        return {}
    return data


def _ensure_private_file(path: Path, *, label: str) -> None:
    if os.name == "nt" or not path.exists():
        return
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        logger.warning("%s_permission_check_failed path=%s error=%s", label, path, exc)
        return
    if mode & 0o077 == 0:
        return
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.warning("%s_permission_fix_failed path=%s mode=%o error=%s", label, path, mode, exc)
        return
    logger.warning("%s_permissions_restricted path=%s previous_mode=%o new_mode=600", label, path, mode)


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
    finally:
        if fd >= 0:
            os.close(fd)
    if os.name != "nt":
        os.chmod(path, 0o600)


def load_config() -> SpiceConfig:
    _ensure_private_file(SETTINGS_PATH, label="settings")
    data = _read_json(SETTINGS_PATH)
    defaults = asdict(SpiceConfig())
    defaults.update(_config_values_from_settings(data))
    return SpiceConfig(**defaults)


def save_config(config: SpiceConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_private_file(SETTINGS_PATH, label="settings")
    data = _read_json(SETTINGS_PATH)
    model_profiles = copy.deepcopy(config.model_profiles)
    profile_key = config.default_model or config.model
    existing_profile = model_profiles.get(profile_key)
    profile = copy.deepcopy(existing_profile) if isinstance(existing_profile, dict) else {}
    profile["provider"] = config.provider
    profile["model"] = config.model
    if config.base_url is not None:
        profile["baseUrl"] = config.base_url
    else:
        profile.pop("baseUrl", None)
        profile.pop("base_url", None)
    if config.protocol is not None:
        profile["protocol"] = config.protocol
    else:
        profile.pop("protocol", None)
    profile["temperature"] = config.temperature
    profile.pop("apiKey", None)
    profile.pop("api_key", None)
    model_profiles[profile_key] = profile
    payload: dict[str, Any] = {
        "defaultModel": profile_key,
        "models": model_profiles,
        "debug": {
            "trace": config.debug_trace,
        },
        "memory": {
            "enabled": config.memory_enabled,
        },
        "subagents": {
            "enabled": config.subagents_enabled,
            "max_concurrent": config.max_concurrent_subagents,
        },
        "logging": {
            "retention_days": config.logging_retention_days,
        },
        "sandbox": config.sandbox,
        "storage": config.storage,
    }
    _write_private_json(SETTINGS_PATH, payload)


def _config_values_from_settings(data: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    model_profiles = data.get("models")
    if isinstance(model_profiles, dict):
        values["model_profiles"] = copy.deepcopy(model_profiles)

    default_model = data.get("defaultModel") or data.get("default_model")
    if isinstance(default_model, str) and default_model:
        values["default_model"] = default_model
        if isinstance(model_profiles, dict):
            profile = model_profiles.get(default_model)
            if isinstance(profile, dict):
                provider = profile.get("provider")
                model_id = profile.get("model") or profile.get("id")
                if isinstance(provider, str) and provider:
                    values["provider"] = provider
                if isinstance(model_id, str) and model_id:
                    values["model"] = model_id
                protocol = profile.get("protocol") or profile.get("api")
                if isinstance(protocol, str) and protocol:
                    values["protocol"] = protocol
                base_url = profile.get("base_url") or profile.get("baseUrl")
                if isinstance(base_url, str) and base_url:
                    values["base_url"] = base_url
                if "temperature" in profile:
                    try:
                        values["temperature"] = float(profile["temperature"])
                    except (TypeError, ValueError):
                        pass

    debug_settings = data.get("debug")
    if isinstance(debug_settings, dict) and "trace" in debug_settings:
        values["debug_trace"] = bool(debug_settings.get("trace"))

    memory_settings = data.get("memory")
    if isinstance(memory_settings, dict) and "enabled" in memory_settings:
        values["memory_enabled"] = bool(memory_settings.get("enabled"))

    subagent_settings = data.get("subagents")
    if isinstance(subagent_settings, dict):
        if "enabled" in subagent_settings:
            values["subagents_enabled"] = bool(subagent_settings.get("enabled"))
        if "max_concurrent" in subagent_settings:
            try:
                max_concurrent = int(subagent_settings.get("max_concurrent"))
            except (TypeError, ValueError):
                max_concurrent = SpiceConfig.max_concurrent_subagents
            values["max_concurrent_subagents"] = min(max(max_concurrent, 1), 3)

    logging_settings = data.get("logging")
    if isinstance(logging_settings, dict) and "retention_days" in logging_settings:
        try:
            retention_days = int(logging_settings.get("retention_days"))
        except (TypeError, ValueError):
            retention_days = SpiceConfig.logging_retention_days
        values["logging_retention_days"] = max(retention_days, 1)

    sandbox_settings = data.get("sandbox")
    if isinstance(sandbox_settings, dict):
        values["sandbox"] = _deep_merge(DEFAULT_SANDBOX_CONFIG, sandbox_settings)

    storage_settings = data.get("storage")
    if isinstance(storage_settings, dict):
        values["storage"] = _normalize_storage_config(storage_settings)

    # Backward-compatible flat settings for older config.json-style files.
    for key, value in data.items():
        if key not in SpiceConfig.__dataclass_fields__:
            continue
        if key in {"sandbox", "storage", "model_profiles", "models", "defaultModel"}:
            continue
        if key == "model" and not isinstance(value, str):
            continue
        values[key] = value
    return values


def _normalize_storage_config(settings: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(DEFAULT_STORAGE_CONFIG)
    backend = settings.get("backend")
    if isinstance(backend, str) and backend.strip().lower() in {"file", "sqlite"}:
        result["backend"] = backend.strip().lower()
    sqlite_path = settings.get("sqlitePath") or settings.get("sqlite_path")
    if isinstance(sqlite_path, str) and sqlite_path.strip():
        result["sqlitePath"] = sqlite_path.strip()
    return result


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_secrets() -> dict[str, Any]:
    _ensure_private_file(SECRETS_PATH, label="secrets")
    return _read_json(SECRETS_PATH)


def save_secret(provider: str, api_key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    secrets = load_secrets()
    secret_name = (ENV_KEYS.get(provider) or [provider])[0]
    secrets[secret_name] = api_key
    _write_private_json(SECRETS_PATH, secrets)


def get_api_key(provider: str, env_names: list[str] | None = None) -> str | None:
    names = env_names or ENV_KEYS.get(provider, [])
    for env_name in names:
        value = os.getenv(env_name)
        if value:
            return value
    secrets = load_secrets()
    for env_name in names:
        value = secrets.get(env_name)
        if isinstance(value, str) and value:
            return value
    value = secrets.get(provider)
    return value if isinstance(value, str) and value else None
