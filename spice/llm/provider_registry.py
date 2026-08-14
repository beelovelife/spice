"""Provider runtime registry."""

from __future__ import annotations

from collections.abc import Callable

from spice.llm.models import Model
from spice.llm.providers.anthropic import AnthropicProvider
from spice.llm.providers.base import Provider
from spice.llm.providers.gemini import GeminiProvider
from spice.llm.providers.openai import OpenAIProvider

ProviderFactory = Callable[[str, Model | None], Provider]


def _openai_compatible_provider(provider: str, model: Model | None) -> Provider:
    return OpenAIProvider(
        provider_name=(model.provider_name or model.provider) if model else provider,
        api_key_hint=(model.api_key_envs or [f"{provider.upper()}_API_KEY"])[0]
        if model
        else f"{provider.upper()}_API_KEY",
        default_base_url=model.base_url if model else None,
    )


def _openai_responses_provider(provider: str, model: Model | None) -> Provider:
    return OpenAIProvider(
        provider_name=(model.provider_name or model.provider) if model else provider,
        api_key_hint=(model.api_key_envs or [f"{provider.upper()}_API_KEY"])[0]
        if model
        else f"{provider.upper()}_API_KEY",
        default_base_url=model.base_url if model else None,
        use_responses=True,
    )


def _openai_provider(provider: str, model: Model | None) -> Provider:
    return OpenAIProvider(use_responses=True)


def _deepseek_provider(provider: str, model: Model | None) -> Provider:
    return OpenAIProvider(
        provider_name="DeepSeek",
        api_key_hint="DEEPSEEK_API_KEY",
        default_base_url="https://api.deepseek.com",
    )


def _anthropic_provider(provider: str, model: Model | None) -> Provider:
    return AnthropicProvider(default_base_url=model.base_url if model else None)


def _gemini_provider(provider: str, model: Model | None) -> Provider:
    return GeminiProvider()


PROTOCOL_FACTORIES: dict[str, ProviderFactory] = {
    "openai-responses": _openai_responses_provider,
    "openai-completions": _openai_compatible_provider,
    "openai-chat-completions": _openai_compatible_provider,
    "anthropic-messages": _anthropic_provider,
    "google-generative-ai": _gemini_provider,
}


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "openai": _openai_provider,
    "deepseek": _deepseek_provider,
    "anthropic": _anthropic_provider,
    "gemini": _gemini_provider,
}


def get_provider(
    provider: str, protocol: str | None = None, model: Model | None = None
) -> Provider | None:
    if protocol:
        protocol_factory = PROTOCOL_FACTORIES.get(protocol)
        if protocol_factory:
            return protocol_factory(provider, model)

    provider_factory = PROVIDER_FACTORIES.get(provider)
    if provider_factory:
        return provider_factory(provider, model)
    return None
