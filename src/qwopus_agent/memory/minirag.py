"""MiniRAG knowledge facade.

Only `insert(document)` and `search(query)` are exposed so the rest of the Agent never depends on
the retrieval implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MiniRAG:
    """Minimal MiniRAG facade for local knowledge operations."""

    # Reason: This first implementation keeps memory testable before a real vector backend is wired.
    _documents: list[str] = field(default_factory=list)

    def insert(self, document: str) -> None:
        """Insert one Markdown-normalized document."""
        if not document.strip():
            raise ValueError("document must not be empty")
        self._documents.append(document)

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
