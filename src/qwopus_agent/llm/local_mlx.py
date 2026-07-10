"""Local MLX LLM adapter."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.llm.openai_compatible import OpenAICompatibleLLM, OpenAICompatibleLLMError


class LocalMLXLLMError(OpenAICompatibleLLMError):
    """Raised when the local MLX server request fails or returns invalid data."""


@dataclass(frozen=True)
class LocalMLXLLM(OpenAICompatibleLLM):
    """Convenience adapter for `mlx_lm.server`."""

    # Reason: Local MLX is only a provider preset; it must still accept any MLX-served model name.
    model: str

    # Role: Default endpoint exposed by `python -m mlx_lm.server`.
    base_url: str = "http://127.0.0.1:8080/v1"
