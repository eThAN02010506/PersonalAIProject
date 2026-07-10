"""LLM provider registry.

The registry maps provider names to factories. New model backends can be added without changing
Agent, Skill, Planner, or Executor code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from qwopus_agent.llm.base import BaseLLM
from qwopus_agent.llm.config import LLMConfig
from qwopus_agent.llm.local_mlx import LocalMLXLLM
from qwopus_agent.llm.openai_compatible import OpenAICompatibleLLM

LLMFactory = Callable[[LLMConfig], BaseLLM]


@dataclass
class LLMRegistry:
    """Registry that creates concrete LLM adapters from provider-neutral config."""

    # Reason: Dependency injection keeps model backend selection outside business logic.
    _factories: dict[str, LLMFactory] = field(default_factory=dict)

    def register(self, provider: str, factory: LLMFactory) -> None:
        """Register one provider factory."""
        if provider in self._factories:
            raise ValueError(f"LLM provider already registered: {provider}")
        self._factories[provider] = factory

    def create(self, config: LLMConfig) -> BaseLLM:
        """Create an LLM adapter from runtime configuration."""
        try:
            factory = self._factories[config.provider]
        except KeyError as exc:
            raise KeyError(f"Unknown LLM provider: {config.provider}") from exc
        return factory(config)

    def list_providers(self) -> list[str]:
        """Return provider names in deterministic order."""
        return sorted(self._factories)


def create_default_llm_registry() -> LLMRegistry:
    """Create the built-in provider registry."""
    registry = LLMRegistry()

    # Reason: `local_mlx` is the current default for Apple Silicon MLX workflows.
    registry.register(
        "local_mlx",
        lambda config: LocalMLXLLM(
            model=config.model,
            base_url=config.base_url or "http://127.0.0.1:8080/v1",
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        ),
    )

    # Reason: this provider supports any server that exposes OpenAI-compatible chat completions.
    registry.register(
        "openai_compatible",
        lambda config: OpenAICompatibleLLM(
            model=config.model,
            base_url=config.base_url or "http://127.0.0.1:8080/v1",
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        ),
    )
    return registry
