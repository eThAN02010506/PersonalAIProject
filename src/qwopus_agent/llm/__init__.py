"""LLM abstractions and adapters."""

from qwopus_agent.llm.base import BaseLLM, ChatMessage, LLMResponse
from qwopus_agent.llm.config import LLMConfig
from qwopus_agent.llm.local_mlx import LocalMLXLLM
from qwopus_agent.llm.openai_compatible import OpenAICompatibleLLM
from qwopus_agent.llm.registry import LLMRegistry, create_default_llm_registry

__all__ = [
    "BaseLLM",
    "ChatMessage",
    "LLMConfig",
    "LLMRegistry",
    "LLMResponse",
    "LocalMLXLLM",
    "OpenAICompatibleLLM",
    "create_default_llm_registry",
]
