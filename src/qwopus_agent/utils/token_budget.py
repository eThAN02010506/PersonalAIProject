"""Model-neutral token estimates and deterministic context budgeting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil

_TOKEN_UNIT = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_MAX_TOOL_OBSERVATION_TOKENS = 6000
_MAX_SYNTHESIS_TOKENS = 12000


def estimate_tokens(text: str) -> int:
    """Estimate multilingual tokens conservatively when a model tokenizer is unavailable."""
    if not text:
        return 0
    lexical_units = len(_TOKEN_UNIT.findall(text))
    byte_estimate = ceil(len(text.encode("utf-8")) / 4)
    return max(1, lexical_units, byte_estimate)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return a prefix bounded by the same estimator used for context planning."""
    if max_tokens < 1:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


@dataclass(frozen=True)
class TokenBudgetManager:
    """Allocate evidence without assuming one specific local or remote model."""

    context_window: int = 32768
    output_reserve: int = 4096
    system_reserve: int = 4096
    history_reserve: int = 1024
    safety_reserve: int = 2048

    def __post_init__(self) -> None:
        if self.context_window < 2048:
            raise ValueError("context_window must be at least 2048 tokens")
        if min(
            self.output_reserve,
            self.system_reserve,
            self.history_reserve,
            self.safety_reserve,
        ) < 0:
            raise ValueError("token reserves cannot be negative")

    @property
    def input_budget(self) -> int:
        """Return the total prompt space after adaptive output and safety reserves."""
        output = min(self.output_reserve, self.context_window // 2)
        safety = min(self.safety_reserve, self.context_window // 8)
        return max(256, self.context_window - output - safety)

    @property
    def system_budget(self) -> int:
        """Cap system instructions when a model exposes a small context window."""
        return min(self.system_reserve, max(128, self.input_budget // 3))

    @property
    def history_budget(self) -> int:
        """Allocate recent conversation context from the same model window."""
        return min(self.history_reserve, max(128, self.input_budget // 3))

    @property
    def evidence_budget(self) -> int:
        # 原因：模型、Tool、历史和输出共享同一个上下文窗口，证据不能使用固定字符上限。
        # 作用：为每次运行计算稳定的剩余 token，模型窗口变化时无需修改文档工具。
        return max(
            256,
            self.input_budget - self.system_budget - self.history_budget,
        )

    @property
    def observation_budget(self) -> int:
        """Bound one Tool result so later calls and the final answer still fit."""
        return min(_MAX_TOOL_OBSERVATION_TOKENS, self.evidence_budget)

    @property
    def synthesis_budget(self) -> int:
        """Bound combined Agent evidence without using character-count slices."""
        return min(_MAX_SYNTHESIS_TOKENS, self.evidence_budget)
