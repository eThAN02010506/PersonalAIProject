"""Base interfaces for model-agnostic LLM integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from qwopus_agent.utils.token_budget import estimate_tokens

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    """A model-agnostic chat message."""

    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM response returned by every adapter."""

    content: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for all language model backends."""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response for a list of chat messages."""

    @property
    def context_window(self) -> int:
        """Return a conservative default when a provider does not publish model metadata."""
        return 32768

    def count_tokens(self, text: str) -> int:
        """Use a model-neutral estimate unless an adapter supplies its exact tokenizer."""
        return estimate_tokens(text)
