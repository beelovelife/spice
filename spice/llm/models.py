"""Model metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token USD prices captured in model metadata."""

    input_per_million_usd: str
    output_per_million_usd: str
    cache_read_per_million_usd: str | None = None
    cache_write_per_million_usd: str | None = None


@dataclass
class Model:
    id: str
    provider: str
    profile_key: str | None = None
    name: str | None = None
    context_window: int = 0
    output_tokens: int = 4096
    supports_vision: bool = False
    supports_reasoning: bool = False
    temperature: float | None = None
    protocol: str | None = None
    base_url: str | None = None
    api_key_envs: list[str] | None = None
    provider_name: str | None = None
    pricing: ModelPricing | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.id
