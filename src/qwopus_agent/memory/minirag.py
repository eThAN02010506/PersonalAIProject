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

        query_terms = query.lower().split()
        matches = [
            document
            for document in self._documents
            if any(term in document.lower() for term in query_terms)
        ]
        return matches
