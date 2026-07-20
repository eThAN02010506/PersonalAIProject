"""Read-only knowledge-graph projections for UI inspection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_models import GraphSnapshot
from qwopus_agent.memory.knowledge_graph import DEFAULT_KNOWLEDGE_GRAPH_PATH

_NODE_COLORS = ("#DCEEFF", "#E4F5E9", "#FFF0CC", "#F6E2EF", "#E8E2FA", "#E5F4F2")


@dataclass(frozen=True)
class KnowledgeGraphService:
    """Load bounded graph snapshots without exposing mutation to the UI."""

    storage_path: Path = DEFAULT_KNOWLEDGE_GRAPH_PATH

    def entity_types(self) -> list[str]:
        """Return available entity types in deterministic order."""
        return sorted({entity.entity_type for entity in self._load().entities()})

    def snapshot(
        self,
        *,
        entity_type: str | None = None,
        max_nodes: int = 100,
    ) -> GraphSnapshot:
        """Return a bounded graph projection for rendering."""
        bounded_nodes = min(500, max(1, max_nodes))
        return self._load().snapshot(entity_type=entity_type, max_nodes=bounded_nodes)

    def evidence_rows(self, snapshot: GraphSnapshot) -> list[dict[str, object]]:
        """Return source-level rows only for edges visible in a snapshot."""
        graph = self._load()
        names = {node.id: node.label for node in snapshot.nodes}
        rows: list[dict[str, object]] = []
        for edge in snapshot.edges:
            relation = graph.get_relation(edge.id)
            if relation is None:
                continue
            for evidence in relation.evidence:
                rows.append(
                    {
                        "source_entity": names.get(relation.source_id, relation.source_id),
                        "relation": relation.relation,
                        "target_entity": names.get(relation.target_id, relation.target_id),
                        "source": evidence.source,
                        "page": evidence.page or "",
                        "evidence": evidence.text,
                    }
                )
        return rows

    def to_dot(self, snapshot: GraphSnapshot) -> str:
        """Serialize a snapshot as safe directed Graphviz DOT."""
        lines = [
            "digraph qwopus_knowledge_graph {",
            '  graph [rankdir="LR", bgcolor="transparent", pad="0.2"];',
            '  node [shape="box", style="rounded,filled", fontname="Helvetica"];',
            '  edge [color="#64748B", fontname="Helvetica", fontsize="10"];',
        ]
        for node in snapshot.nodes:
            label = node.label if len(node.label) <= 60 else f"{node.label[:57]}..."
            color = _entity_color(node.entity_type)
            lines.append(
                f"  {_dot_quote(node.id)} "
                f"[label={_dot_quote(label)}, fillcolor={_dot_quote(color)}];"
            )
        for edge in snapshot.edges:
            lines.append(
                f"  {_dot_quote(edge.source)} -> {_dot_quote(edge.target)} "
                f"[label={_dot_quote(edge.label)}];"
            )
        lines.append("}")
        return "\n".join(lines)

    def _load(self) -> PersistentKnowledgeGraph:
        return PersistentKnowledgeGraph(Path(self.storage_path))


def _entity_color(entity_type: str) -> str:
    digest = blake2b(entity_type.casefold().encode("utf-8"), digest_size=1).digest()[0]
    return _NODE_COLORS[digest % len(_NODE_COLORS)]


def _dot_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"'
