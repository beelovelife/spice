"""Model metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Model:
    id: str
    provider: str
    profile_key: str | None = None
    name: str | None = None
    context_window: int = 0
    output_tokens: int = 4096
    supports_vision: bool = False
    temperature: float | None = None
    protocol: str | None = None
    base_url: str | None = None
    api_key_envs: list[str] | None = None
    provider_name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.id
