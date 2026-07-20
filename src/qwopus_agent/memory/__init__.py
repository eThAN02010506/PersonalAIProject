"""Knowledge and memory facades loaded only when requested."""

from __future__ import annotations

from typing import Any

__all__ = ["MiniRAG"]


def __getattr__(name: str) -> Any:
    """Load the expensive MiniRAG stack only for callers that request it."""
    if name != "MiniRAG":
        raise AttributeError(name)
    # 原因：导入 graph_backend 不应顺带加载 embeddings、Transformers 和 Torch。
    # 作用：Skill 扫描与非 RAG 命令快速启动，真实 MiniRAG 调用行为保持不变。
    from qwopus_agent.memory.minirag import MiniRAG

    globals()[name] = MiniRAG
    return MiniRAG
