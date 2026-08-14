"""Model registry and initial model resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spice.llm.config import SETTINGS_PATH, SpiceConfig
from spice.llm.models import Model, ModelPricing

BUILTIN_MODELS = [
    Model(
        id="gpt-4o-mini",
        provider="openai",
        context_window=128000,
        output_tokens=4096,
        protocol="openai-responses",
        api_key_envs=["OPENAI_API_KEY"],
    ),
    Model(
        id="gpt-4o",
        provider="openai",
        context_window=128000,
        output_tokens=4096,
        protocol="openai-responses",
        api_key_envs=["OPENAI_API_KEY"],
    ),
    Model(
        id="gpt-5.1",
        provider="openai",
        context_window=256000,
        output_tokens=8192,
        supports_reasoning=True,
        protocol="openai-responses",
        api_key_envs=["OPENAI_API_KEY"],
    ),
    Model(
        id="deepseek-v4-flash",
        provider="deepseek",
        context_window=1000000,
        output_tokens=60000,
        supports_reasoning=True,
        protocol="openai-responses",
        base_url="https://api.deepseek.com",
        api_key_envs=["DEEPSEEK_API_KEY"],
        provider_name="DeepSeek",
        pricing=ModelPricing(
            input_per_million_usd="0.14",
            output_per_million_usd="0.28",
            cache_read_per_million_usd="0.0028",
        ),
    ),
    Model(
        id="deepseek-v4-pro",
        provider="deepseek",
        context_window=1000000,
        output_tokens=60000,
        supports_reasoning=True,
        protocol="openai-responses",
        base_url="https://api.deepseek.com",
        api_key_envs=["DEEPSEEK_API_KEY"],
        provider_name="DeepSeek",
        pricing=ModelPricing(
            input_per_million_usd="0.435",
            output_per_million_usd="0.87",
            cache_read_per_million_usd="0.003625",
        ),
    ),
    Model(
        id="claude-haiku-4-5",
        provider="anthropic",
        context_window=200000,
        output_tokens=64000,
        protocol="anthropic-messages",
        api_key_envs=["ANTHROPIC_API_KEY"],
    ),
    Model(
        id="claude-sonnet-4-6",
        provider="anthropic",
        context_window=1000000,
        output_tokens=128000,
        protocol="anthropic-messages",
        api_key_envs=["ANTHROPIC_API_KEY"],
    ),
    Model(
        id="claude-opus-4-8",
        provider="anthropic",
        context_window=1000000,
        output_tokens=128000,
        protocol="anthropic-messages",
        api_key_envs=["ANTHROPIC_API_KEY"],
    ),
    Model(
        id="gemini-2.5-pro",
        provider="gemini",
        context_window=1000000,
        output_tokens=8192,
        protocol="google-generative-ai",
        api_key_envs=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    ),
    Model(
        id="gemini-2.5-flash",
        provider="gemini",
        context_window=1000000,
        output_tokens=8192,
        protocol="google-generative-ai",
        api_key_envs=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    ),
]


@dataclass
class InitialModelResult:
    model: Model | None
    message: str | None = None


class ModelRegistry:
    def __init__(self, settings_path: Path = SETTINGS_PATH) -> None:
        self._models: dict[str, Model] = {}
        self._model_profiles: dict[str, Model] = {}
        for model in BUILTIN_MODELS:
            self.register(model)
        self._load_custom_models(settings_path)

    def register(self, model: Model) -> None:
        self._models[f"{model.provider}/{model.id}"] = model
        if model.profile_key:
            self._model_profiles[model.profile_key] = model

    def all(self) -> list[Model]:
        return list(self._models.values())

    def providers(self) -> list[str]:
        return sorted({m.provider for m in self._models.values()})

    def find(self, provider: str | None, model_id: str | None) -> Model | None:
        if not model_id:
            return None
        if model_id in self._model_profiles:
            return self._model_profiles[model_id]
        if "/" in model_id:
            return self._models.get(model_id)
        if provider:
            found = self._models.get(f"{provider}/{model_id}")
            if found:
                return found
        matches = [m for m in self._models.values() if m.id == model_id]
        return matches[0] if len(matches) == 1 else None

    def _load_custom_models(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        models = data.get("models")
        if not isinstance(models, dict):
            return
        for profile_key, raw_model in models.items():
            if not isinstance(profile_key, str) or not isinstance(raw_model, dict):
                continue
            model = _model_from_profile(profile_key, raw_model)
            if model:
                self.register(model)


def _model_from_profile(profile_key: str, data: dict[str, Any]) -> Model | None:
    provider = data.get("provider")
    model_id = data.get("model") or data.get("id")
    if not isinstance(provider, str) or not isinstance(model_id, str):
        return None
    return Model(
        id=model_id,
        provider=provider,
        profile_key=profile_key,
        name=data.get("name"),
        context_window=_int_field(
            data, "context_window", aliases=["contextWindow"], default=128000
        ),
        output_tokens=_output_tokens_field(data, default=4096),
        supports_vision=_bool_field(
            data, "supports_vision", aliases=["supportsVision", "vision"], default=False
        ),
        supports_reasoning=_bool_field(
            data,
            "supports_reasoning",
            aliases=["supportsReasoning", "reasoning"],
            default=False,
        ),
        temperature=_float_field(data, "temperature"),
        protocol=_protocol_value(data),
        base_url=_optional_str(data.get("base_url") or data.get("baseUrl")),
        api_key_envs=_api_key_envs(data),
        provider_name=_optional_str(
            data.get("provider_name") or data.get("providerName")
        ),
        pricing=_pricing_value(data.get("pricing")),
    )


def _pricing_value(value: Any) -> ModelPricing | None:
    if not isinstance(value, dict):
        return None
    input_price = value.get("input_per_million_usd") or value.get("inputPerMillionUsd")
    output_price = value.get("output_per_million_usd") or value.get(
        "outputPerMillionUsd"
    )
    if input_price is None or output_price is None:
        return None
    return ModelPricing(
        input_per_million_usd=str(input_price),
        output_per_million_usd=str(output_price),
        cache_read_per_million_usd=_optional_price(
            value, "cache_read_per_million_usd", "cacheReadPerMillionUsd"
        ),
        cache_write_per_million_usd=_optional_price(
            value, "cache_write_per_million_usd", "cacheWritePerMillionUsd"
        ),
    )


def _optional_price(value: dict[str, Any], snake: str, camel: str) -> str | None:
    raw = value.get(snake)
    if raw is None:
        raw = value.get(camel)
    return str(raw) if raw is not None else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _protocol_value(data: dict[str, Any]) -> str | None:
    return _optional_str(data.get("protocol") or data.get("api"))


def _int_field(
    data: dict[str, Any], name: str, *, aliases: list[str], default: int
) -> int:
    raw = data.get(name)
    for alias in aliases:
        if raw is None:
            raw = data.get(alias)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _bool_field(
    data: dict[str, Any], name: str, *, aliases: list[str], default: bool
) -> bool:
    raw = data.get(name)
    for alias in aliases:
        if raw is None:
            raw = data.get(alias)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _float_field(data: dict[str, Any], name: str) -> float | None:
    raw = data.get(name)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _output_tokens_field(data: dict[str, Any], *, default: int) -> int:
    return _int_field(
        data,
        "output_tokens",
        aliases=["outputTokens", "max_tokens", "maxTokens"],
        default=default,
    )


def _api_key_envs(data: dict[str, Any]) -> list[str] | None:
    raw = data.get("api_key_envs") or data.get("apiKeyEnvs")
    if isinstance(raw, list):
        values = [item for item in raw if isinstance(item, str) and item]
        return values or None
    raw_key = data.get("api_key_env") or data.get("apiKeyEnv")
    if isinstance(raw_key, str) and raw_key:
        return [raw_key]
    return None


def find_initial_model(
    registry: ModelRegistry,
    config: SpiceConfig,
    provider: str | None = None,
    model_id: str | None = None,
) -> InitialModelResult:
    resolved_provider = provider or config.provider
    resolved_model = model_id or config.model
    model = registry.find(resolved_provider, resolved_model)
    if model:
        return InitialModelResult(model=model)
    return InitialModelResult(
        model=None,
        message=f"Model not found: provider={resolved_provider!r}, model={resolved_model!r}",
    )
