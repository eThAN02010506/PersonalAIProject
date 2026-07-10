"""LLM runtime configuration.

This module keeps model names and provider settings outside the Agent, so any compatible model can
be swapped in by changing configuration instead of rewriting Planner, Executor, or Skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ModuleNotFoundError:  # pragma: no cover - used only before project deps are installed.
    BaseModel = object  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[misc]
        return default


if BaseModel is object:

    @dataclass(frozen=True)
    class LLMConfig:
        """Fallback config model used before pydantic is installed."""

        provider: str
        model: str
        base_url: str | None = None
        api_key: str | None = None
        timeout_seconds: float = 120.0
        extra: dict[str, Any] = field(default_factory=dict)

else:

    class LLMConfig(BaseModel):
        """Provider-neutral configuration for creating an LLM adapter."""

        # Reason: The model backend must be selected by configuration, not hard-coded in Agents.
        model_config = ConfigDict(frozen=True)

        # Role: Adapter key such as `local_mlx`, `openai_compatible`, `qwen`, or future providers.
        provider: str

        # Role: Runtime model identifier passed to the selected provider.
        model: str

        # Role: Optional API endpoint for local or remote OpenAI-compatible servers.
        base_url: str | None = None

        # Role: Optional auth token for providers that need one.
        api_key: str | None = None

        # Role: Network timeout for HTTP-based providers.
        timeout_seconds: float = 120.0

        # Role: Provider-specific settings without leaking them into Agent code.
        extra: dict[str, Any] = Field(default_factory=dict)
