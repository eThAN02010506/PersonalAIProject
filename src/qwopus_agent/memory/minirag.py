"""MiniRAG knowledge facade.

Only `insert(document)` and `search(query)` are exposed so the rest of the Agent never depends on
the retrieval implementation details.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from uuid import uuid4


DEFAULT_MINIRAG_STORE_PATH = Path("storage/minirag/documents.jsonl")
VECTOR_DIMENSIONS = 256


@dataclass
class MiniRAG:
    """Minimal MiniRAG facade for local knowledge operations."""

    # Reason: Public API remains insert/search while storage can evolve behind the facade.
    _documents: list[str] = field(default_factory=list)

    _vectors: list[list[float]] = field(default_factory=list)

    storage_path: Path = DEFAULT_MINIRAG_STORE_PATH

    def __post_init__(self) -> None:
        """Load persisted Markdown documents on startup."""
        self.storage_path = Path(self.storage_path)
        self._documents.extend(_load_documents(self.storage_path))
        self._vectors.extend(_document_vector(document) for document in self._documents)

    def insert(self, document: str) -> None:
        """Insert one Markdown-normalized document."""
        if not document.strip():
            raise ValueError("document must not be empty")
        if document in self._documents:
            return
        self._documents.append(document)
        self._vectors.append(_document_vector(document))
        _append_document(self.storage_path, document)

    def search(self, query: str) -> list[str]:
        """Search documents with an internal vector index."""
        if not query.strip():
            raise ValueError("query must not be empty")

        query_vector = _document_vector(query)
        scored = [
            (_cosine_similarity(query_vector, vector), document)
            for document, vector in zip(self._documents, self._vectors, strict=True)
        ]
        # 原因：MiniRAG 对外只暴露 search(query)，内部可以从关键词升级为向量排序。
        # 作用：返回和查询最接近的文档，同时保持旧调用方完全不变。
        return [
            document
            for score, document in sorted(scored, key=lambda item: item[0], reverse=True)
            if score > 0
        ]


def _document_vector(text: str) -> list[float]:
    """Build a small deterministic hashed vector for local retrieval."""
    vector = [0.0] * VECTOR_DIMENSIONS
    for token in _vector_tokens(text):
        index = _token_index(token)
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _vector_tokens(text: str) -> list[str]:
    """Tokenize English words, Chinese characters, and character n-grams."""
    lowered = text.lower()
    word_tokens = re.findall(r"[a-z0-9]+", lowered)
    chinese_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    compact = re.sub(r"\s+", "", lowered)
    # 原因：本地无 embedding 模型时，字符 n-gram 能提升相近拼写和中文短查询召回。
    # 作用：让 vector search 比纯关键词匹配更稳，同时保持零外部依赖。
    ngrams = [compact[index:index + 3] for index in range(max(len(compact) - 2, 0))]
    return list(dict.fromkeys(word_tokens + chinese_chars + ngrams))


def _token_index(token: str) -> int:
    """Map one token to a stable vector bucket."""
    digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % VECTOR_DIMENSIONS


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity for normalized vectors."""
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _load_documents(storage_path: Path) -> list[str]:
    """Load persisted MiniRAG documents from JSONL."""
    if not storage_path.exists():
        return []

    documents: list[str] = []
    seen: set[str] = set()
    for line in storage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        document = record.get("document")
        if isinstance(document, str) and document.strip() and document not in seen:
            seen.add(document)
            documents.append(document)
    return documents


def _append_document(storage_path: Path, document: str) -> None:
    """Append one MiniRAG document to local JSONL storage."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document": document,
    }
    # 原因：MiniRAG 需要重启后自动恢复，向量索引可以从文档内容确定性重建。
    # 作用：只持久化原始 document，启动时自动重建内部向量。
    with storage_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
