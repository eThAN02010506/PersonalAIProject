"""LLM runtime configuration.

This module keeps model names and provider settings outside the Agent, so any compatible model can
be swapped in by changing configuration instead of rewriting Planner, Executor, or Skills.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
