"""Administrative knowledge-index lifecycle operations outside Agent-facing APIs."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.memory import MiniRAG


@dataclass(frozen=True)
class KnowledgeMaintenanceService:
    """Manage persisted sources while MiniRAG keeps its public insert/search facade."""

    memory: MiniRAG

    def list_sources(self) -> list[str]:
        """Return uploaded sources available for maintenance."""
        return self.memory._list_sources()

    def delete_source(self, source: str) -> int:
        """Delete one source's records, vectors, and unsupported graph facts."""
        return self.memory._delete_source(source)

    def rebuild_indexes(self) -> None:
        """Rebuild all derived indexes from persisted Markdown documents."""
        self.memory._rebuild_indexes()
