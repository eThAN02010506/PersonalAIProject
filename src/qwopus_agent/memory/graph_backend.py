"""Persistent directed knowledge graph for entity and relation retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from qwopus_agent.memory.graph_models import (
    EntityRecord,
    GraphEdgeView,
    GraphEvidence,
    GraphNodeView,
    GraphPath,
    GraphSnapshot,
    RelationRecord,
)

GRAPH_SCHEMA_VERSION = 1


@dataclass
class PersistentKnowledgeGraph:
    """Store canonical entities and directed relations with provenance."""

    storage_path: Path
    _entities: dict[str, EntityRecord] = field(default_factory=dict, init=False, repr=False)
    _relations: dict[str, RelationRecord] = field(default_factory=dict, init=False, repr=False)
    _graph: nx.MultiDiGraph[str] = field(
        default_factory=nx.MultiDiGraph,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.storage_path = Path(self.storage_path)
        self._load()

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def get_entity(self, entity_id: str) -> EntityRecord | None:
        return self._entities.get(entity_id)

    def get_relation(self, relation_id: str) -> RelationRecord | None:
        return self._relations.get(relation_id)

    def entities(self) -> tuple[EntityRecord, ...]:
        return tuple(
            sorted(self._entities.values(), key=lambda item: item.canonical_name.casefold())
        )

    def relations(self) -> tuple[RelationRecord, ...]:
        return tuple(sorted(self._relations.values(), key=lambda item: item.id))

    def upsert_entity(self, entity: EntityRecord) -> EntityRecord:
        """Insert an entity or merge new aliases, descriptions, and evidence."""
        current = self._entities.get(entity.id)
        if current is not None:
            entity = current.model_copy(
                update={
                    "aliases": _unique_strings((*current.aliases, *entity.aliases)),
                    "description": current.description or entity.description,
                    "entity_type": (
                        entity.entity_type
                        if current.entity_type.casefold() == "unknown"
                        else current.entity_type
                    ),
                    "evidence": _unique_evidence((*current.evidence, *entity.evidence)),
                }
            )
        self._entities[entity.id] = entity
        self._graph.add_node(entity.id)
        self._persist()
        return entity

    def upsert_relation(self, relation: RelationRecord) -> RelationRecord:
        """Insert a relation only when both canonical endpoint entities exist."""
        if relation.source_id not in self._entities or relation.target_id not in self._entities:
            raise ValueError("relation endpoints must exist before relation insertion")

        current = self._relations.get(relation.id)
        if current is not None:
            relation = current.model_copy(
                update={
                    "confidence": max(current.confidence, relation.confidence),
                    "evidence": _unique_evidence((*current.evidence, *relation.evidence)),
                }
            )
        self._relations[relation.id] = relation
        self._graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.id,
            relation_id=relation.id,
        )
        self._persist()
        return relation

    def paths_between(
        self,
        source_ids: tuple[str, ...],
        target_ids: tuple[str, ...],
        *,
        max_hops: int = 3,
        limit: int = 10,
    ) -> list[GraphPath]:
        """Return bounded paths while preserving each relation's original direction."""
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1")

        undirected = self._graph.to_undirected()
        paths: list[GraphPath] = []
        for source_id in source_ids:
            for target_id in target_ids:
                if (
                    source_id == target_id
                    or source_id not in undirected
                    or target_id not in undirected
                ):
                    continue
                for node_path in nx.all_simple_paths(
                    undirected,
                    source=source_id,
                    target=target_id,
                    cutoff=max_hops,
                ):
                    path = self._build_path(tuple(node_path))
                    if path is not None:
                        paths.append(path)
                    if len(paths) >= limit:
                        return paths
        return paths

    def neighborhood(
        self,
        entity_ids: tuple[str, ...],
        *,
        max_hops: int = 2,
        limit: int = 10,
    ) -> list[GraphPath]:
        """Return shortest paths from selected entities to nearby graph nodes."""
        undirected = self._graph.to_undirected()
        paths: list[GraphPath] = []
        for entity_id in entity_ids:
            if entity_id not in undirected:
                continue
            discovered = nx.single_source_shortest_path(undirected, entity_id, cutoff=max_hops)
            for target_id, node_path in discovered.items():
                if target_id == entity_id:
                    continue
                path = self._build_path(tuple(node_path))
                if path is not None:
                    paths.append(path)
                if len(paths) >= limit:
                    return paths
        return paths

    def snapshot(self, *, entity_type: str | None = None, max_nodes: int = 150) -> GraphSnapshot:
        """Return a bounded read-only projection for visualization."""
        entities = [
            entity
            for entity in self.entities()
            if entity_type is None or entity.entity_type == entity_type
        ][:max_nodes]
        entity_ids = {entity.id for entity in entities}
        relations = [
            relation
            for relation in self.relations()
            if relation.source_id in entity_ids and relation.target_id in entity_ids
        ]
        return GraphSnapshot(
            nodes=tuple(
                GraphNodeView(
                    id=entity.id,
                    label=entity.canonical_name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases,
                )
                for entity in entities
            ),
            edges=tuple(
                GraphEdgeView(
                    id=relation.id,
                    source=relation.source_id,
                    target=relation.target_id,
                    label=relation.relation,
                    confidence=relation.confidence,
                    evidence_count=len(relation.evidence),
                )
                for relation in relations
            ),
        )

    def remove_document(self, document_id: str) -> None:
        """Remove one document's evidence and delete unsupported graph facts."""
        updated_relations: dict[str, RelationRecord] = {}
        for relation in self._relations.values():
            evidence = tuple(
                item for item in relation.evidence if item.document_id != document_id
            )
            if evidence:
                updated_relations[relation.id] = relation.model_copy(update={"evidence": evidence})

        updated_entities: dict[str, EntityRecord] = {}
        supported_entity_ids = {
            endpoint
            for relation in updated_relations.values()
            for endpoint in (relation.source_id, relation.target_id)
        }
        for entity in self._entities.values():
            evidence = tuple(item for item in entity.evidence if item.document_id != document_id)
            if evidence or entity.id in supported_entity_ids:
                updated_entities[entity.id] = entity.model_copy(update={"evidence": evidence})

        self._entities = updated_entities
        self._relations = updated_relations
        self._rebuild_graph()
        self._persist()

    def _build_path(self, node_path: tuple[str, ...]) -> GraphPath | None:
        relations: list[RelationRecord] = []
        for left, right in zip(node_path, node_path[1:], strict=False):
            candidates = [
                relation
                for relation in self._relations.values()
                if {relation.source_id, relation.target_id} == {left, right}
            ]
            if not candidates:
                return None
            relations.append(max(candidates, key=lambda item: item.confidence))

        entities = [self._entities[entity_id] for entity_id in node_path]
        evidence = _unique_evidence(
            tuple(item for relation in relations for item in relation.evidence)
        )
        return GraphPath(
            entity_ids=node_path,
            entity_names=tuple(entity.canonical_name for entity in entities),
            relations=tuple(relations),
            evidence=evidence,
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if payload.get("schema_version") != GRAPH_SCHEMA_VERSION:
            return
        self._entities = {
            entity.id: entity
            for item in payload.get("entities", [])
            if isinstance(item, dict)
            for entity in [EntityRecord.model_validate(item)]
        }
        self._relations = {
            relation.id: relation
            for item in payload.get("relations", [])
            if isinstance(item, dict)
            for relation in [RelationRecord.model_validate(item)]
        }
        self._rebuild_graph()

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "entities": [entity.model_dump(mode="json") for entity in self.entities()],
            "relations": [relation.model_dump(mode="json") for relation in self.relations()],
        }
        temporary_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 原因：进程中断时不能留下半个图谱 JSON，否则下次启动会丢失全部关系。
        # 作用：先完整写入临时文件，再通过原子替换提交新图谱版本。
        temporary_path.replace(self.storage_path)

    def _rebuild_graph(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._graph.add_nodes_from(self._entities)
        for relation in self._relations.values():
            if relation.source_id in self._entities and relation.target_id in self._entities:
                self._graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    key=relation.id,
                    relation_id=relation.id,
                )


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value.strip()))


def _unique_evidence(values: tuple[GraphEvidence, ...]) -> tuple[GraphEvidence, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[GraphEvidence] = []
    for value in values:
        key = (value.chunk_id, value.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return tuple(unique)
