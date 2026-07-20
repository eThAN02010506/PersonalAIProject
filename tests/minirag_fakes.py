"""Deterministic embedding test double for MiniRAG unit tests."""

from __future__ import annotations

import re
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path

import numpy as np

from qwopus_agent.memory import MiniRAG


class TestEmbeddingBackend:
    """Small semantic-like encoder that keeps unit tests offline and deterministic."""

    model_name = "test-multilingual-embedding-v1"
    dimensions = 64

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.encode_calls += 1
        return np.asarray([self._encode_one(text) for text in texts], dtype=np.float32)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        aliases = {
            "automobile": "car",
            "vehicle": "car",
            "收入": "revenue",
            "收益": "revenue",
            "年度": "annual",
        }
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", lowered)
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", lowered))
        for token in tokens:
            canonical = aliases.get(token, token)
            digest = blake2b(canonical.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0 else vector / norm


def make_test_minirag(storage_path: Path) -> MiniRAG:
    """Create isolated MiniRAG without loading a downloaded production model."""
    # 原因：单元测试只验证知识层行为，不应依赖网络或 Hugging Face 缓存。
    # 作用：使用可预测向量快速覆盖持久化、排序和调用链。
    return MiniRAG(
        storage_path=storage_path,
        embedding_backend=TestEmbeddingBackend(),
    )
