"""Application-facing knowledge contract independent from one RAG backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class KnowledgeStore(Protocol):
    """Minimal persistence and retrieval boundary used by Skills and Tool adapters."""

    def insert(self, document: str, *, document_id: str | None = None) -> str:
        """Persist one Markdown-normalized document and return its stable id."""

    def search(
        self,
        query: str,
        min_relevance: float = 0.25,
        *,
        document_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[str]:
        """Return source-labelled evidence without exposing index implementation."""
