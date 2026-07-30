"""LLM runtime configuration.

This module keeps model names and provider settings outside the Agent, so any compatible model can
be swapped in by changing configuration instead of rewriting Planner, Executor, or Skills.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class ModelCapabilities:
    """Runtime facts that affect Agent and context behavior."""

    context_window_tokens: int = 32768
    agent_mode: Literal["tool_calling", "code"] = "tool_calling"
    supports_structured_output: bool = False
    supports_vision: bool = False
    tokenizer_name: str | None = None

    def __post_init__(self) -> None:
        if self.context_window_tokens < 2048:
            raise ValueError("context_window_tokens must be at least 2048.")


@dataclass(frozen=True)
class ModelSettings:
    """Single model configuration shared by LLM adapters, smolagents, and the API."""

    model_id: str
    base_url: str
    provider: str = "openai_compatible"
    api_key: str = "sk-optiq-local"
    timeout_seconds: int = 120
    max_retries: int = 1
    run_timeout_seconds: int = 600
    temperature: float = 0.2
    max_tokens: int = 8192
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive.")
        if not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3.")
        if self.run_timeout_seconds < self.timeout_seconds:
            raise ValueError("run_timeout_seconds must not be shorter than one request timeout.")
        if self.max_tokens >= self.context_window_tokens:
            # 原因：输出上限占满上下文后，系统提示、历史和 Tool 证据没有任何可用空间。
            # 作用：在请求模型前拒绝不可能成立的运行配置。
            raise ValueError("max_tokens must be smaller than the context window.")

    @property
    def context_window_tokens(self) -> int:
        """Keep token-budget callers independent from capability storage shape."""
        return self.capabilities.context_window_tokens

    @classmethod
    def from_env(cls) -> ModelSettings:
        """Load one provider-neutral runtime configuration from environment variables."""
        return cls(
            model_id=os.getenv(
                "QWOPUS_MLX_MODEL",
                "gemma-4-12B-it-qat-OptiQ-4bit",
            ),
            base_url=os.getenv("QWOPUS_MLX_BASE_URL", "http://127.0.0.1:8080/v1"),
            provider=os.getenv("QWOPUS_LLM_PROVIDER", "openai_compatible"),
            api_key=os.getenv("QWOPUS_SMOLAGENTS_API_KEY", "sk-optiq-local"),
            timeout_seconds=int(os.getenv("QWOPUS_SMOLAGENTS_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("QWOPUS_SMOLAGENTS_MAX_RETRIES", "1")),
            run_timeout_seconds=int(
                os.getenv("QWOPUS_AGENT_RUN_TIMEOUT_SECONDS", "600")
            ),
            temperature=float(os.getenv("QWOPUS_SMOLAGENTS_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("QWOPUS_SMOLAGENTS_MAX_TOKENS", "8192")),
            capabilities=ModelCapabilities(
                context_window_tokens=int(
                    os.getenv("QWOPUS_SMOLAGENTS_CONTEXT_WINDOW_TOKENS", "32768")
                ),
                agent_mode=_agent_mode_from_env(),
                supports_structured_output=_env_flag(
                    "QWOPUS_MODEL_SUPPORTS_STRUCTURED_OUTPUT"
                ),
                supports_vision=_env_flag("QWOPUS_MODEL_SUPPORTS_VISION"),
                tokenizer_name=os.getenv("QWOPUS_MODEL_TOKENIZER") or None,
            ),
        )

    def to_llm_config(self) -> LLMConfig:
        """Convert runtime settings into the provider Registry contract."""
        return LLMConfig(
            provider=self.provider,
            model=self.model_id,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            extra={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_retries": self.max_retries,
                "capabilities": self.capabilities,
            },
        )


class LLMConfig(BaseModel):
    """Provider-neutral configuration for creating an LLM adapter."""

    # Reason: Pydantic is a required project dependency, so a second fallback contract is dead code.
    # Role: Every LLM adapter receives the same validated immutable configuration.
    model_config = ConfigDict(frozen=True)
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 120.0
    extra: dict[str, Any] = Field(default_factory=dict)


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().casefold() in {"1", "true", "yes", "on"}


def _agent_mode_from_env() -> Literal["tool_calling", "code"]:
    value = (os.getenv("QWOPUS_AGENT_MODE") or "tool_calling").strip().casefold()
    return "code" if value == "code" else "tool_calling"
