"""Model-neutral token estimates and deterministic context budgeting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil

_TOKEN_UNIT = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")


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
    history_reserve: int = 4096
    safety_reserve: int = 2048

    @property
    def evidence_budget(self) -> int:
        # 原因：模型、Tool、历史和输出共享同一个上下文窗口，证据不能使用固定字符上限。
        # 作用：为每次运行计算稳定的剩余 token，模型窗口变化时无需修改文档工具。
        reserved = (
            self.output_reserve
            + self.system_reserve
            + self.history_reserve
            + self.safety_reserve
        )
        return max(512, self.context_window - reserved)
