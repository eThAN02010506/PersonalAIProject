"""MiniRAG knowledge facade.

Only `insert(document)` and `search(query)` are exposed so the rest of the Agent never depends on
the retrieval implementation details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_MINIRAG_STORE_PATH = Path("storage/minirag/documents.jsonl")


@dataclass
class MiniRAG:
    """Minimal MiniRAG facade for local knowledge operations."""

    # Reason: Public API remains insert/search while storage can evolve behind the facade.
    _documents: list[str] = field(default_factory=list)

    storage_path: Path = DEFAULT_MINIRAG_STORE_PATH

    def __post_init__(self) -> None:
        """Load persisted Markdown documents on startup."""
        self.storage_path = Path(self.storage_path)
        self._documents.extend(_load_documents(self.storage_path))

    def insert(self, document: str) -> None:
        """Insert one Markdown-normalized document."""
        if not document.strip():
            raise ValueError("document must not be empty")
        if document in self._documents:
            return
        self._documents.append(document)
        _append_document(self.storage_path, document)

    def search(self, query: str) -> list[str]:
        """Search documents with a simple local fallback."""
        if not query.strip():
            raise ValueError("query must not be empty")

        query_terms = _query_terms(query)
        matches = [
            document
            for document in self._documents
            if any(term in document.lower() for term in query_terms)
        ]
        return matches


def _query_terms(query: str) -> list[str]:
    """Build simple searchable terms for English and Chinese queries."""
    lowered = query.lower().strip()
    terms = lowered.split()
    compact = "".join(terms)
    if compact and compact not in terms:
        terms.append(compact)
    if any("\u4e00" <= char <= "\u9fff" for char in compact):
        # 原因：中文问题通常没有空格，直接 split 会导致召回很弱。
        # 作用：加入中文字符级 term，让“分析收入”能命中包含“收入”的文档。
        terms.extend(char for char in compact if "\u4e00" <= char <= "\u9fff")
    return list(dict.fromkeys(term for term in terms if term))


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
    # 原因：MiniRAG 需要重启后自动恢复，但当前阶段不引入真实向量数据库。
    # 作用：把 insert(document) 追加写入 storage/minirag/documents.jsonl。
    with storage_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
