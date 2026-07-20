"""Knowledge-graph ingestion and bounded multi-hop query orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

from qwopus_agent.memory.entity_resolver import EntityResolver, normalize_entity_name
from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import GraphExtractor
from qwopus_agent.memory.graph_models import (
    EntityCandidate,
    EntityRecord,
    GraphChunk,
    GraphEvidence,
    GraphExtraction,
    GraphPath,
    RelationRecord,
)

DEFAULT_KNOWLEDGE_GRAPH_PATH = Path("storage/minirag/knowledge_graph.json")


@dataclass(frozen=True)
class GraphIngestionResult:
    """Counts returned after one bounded graph-ingestion operation."""

    extracted_entities: int
    extracted_relations: int
    graph_entities: int
    graph_relations: int


@dataclass
class KnowledgeGraphIndex:
    """Coordinate extraction, entity resolution, persistence, and graph queries."""

    graph: PersistentKnowledgeGraph
    extractor: GraphExtractor
    resolver: EntityResolver | None = None

    def __post_init__(self) -> None:
        self.resolver = self.resolver or EntityResolver(self.graph)

    def insert(self, chunks: tuple[GraphChunk, ...]) -> GraphIngestionResult:
        """Extract and persist evidence-backed facts from document chunks."""
        extraction = self.extractor.extract(chunks)
        self.insert_extraction(extraction)
        return GraphIngestionResult(
            extracted_entities=len(extraction.entities),
            extracted_relations=len(extraction.relations),
            graph_entities=self.graph.entity_count,
            graph_relations=self.graph.relation_count,
        )

    def insert_extraction(self, extraction: GraphExtraction) -> None:
        """Persist one already-validated extraction for deterministic testing and reuse."""
        candidates = {
            normalize_entity_name(candidate.name): candidate
            for candidate in extraction.entities
        }
        resolved: dict[str, EntityRecord] = {}
        for candidate in extraction.entities:
            resolved[normalize_entity_name(candidate.name)] = self._resolver.resolve(candidate)

        for relation in extraction.relations:
            source = self._resolve_endpoint(
                relation.source,
                relation.evidence,
                candidates,
                resolved,
            )
            target = self._resolve_endpoint(
                relation.target,
                relation.evidence,
                candidates,
                resolved,
            )
            relation_record = RelationRecord(
                id=_relation_id(source.id, relation.relation, target.id),
                source_id=source.id,
                relation=relation.relation.strip(),
                target_id=target.id,
                confidence=relation.confidence,
                evidence=(relation.evidence,),
            )
            self.graph.upsert_relation(relation_record)

    def search(self, query: str, *, max_hops: int = 4, limit: int = 10) -> list[GraphPath]:
        """Find paths between mentioned entities or the neighborhood of one entity."""
        mentioned = self.entities_mentioned_in(query)
        if len(mentioned) >= 2:
            return self.graph.paths_between(
                (mentioned[0].id,),
                tuple(entity.id for entity in mentioned[1:]),
                max_hops=max_hops,
                limit=limit,
            )
        if len(mentioned) == 1:
            return self.graph.neighborhood(
                (mentioned[0].id,),
                max_hops=max_hops,
                limit=limit,
            )
        return []

    def paths_between(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 4,
        limit: int = 10,
    ) -> list[GraphPath]:
        """Query graph paths using canonical names or aliases for both endpoints."""
        source_entity = self._resolver.find(source)
        target_entity = self._resolver.find(target)
        if source_entity is None or target_entity is None:
            return []
        return self.graph.paths_between(
            (source_entity.id,),
            (target_entity.id,),
            max_hops=max_hops,
            limit=limit,
        )

    def entities_mentioned_in(self, query: str) -> list[EntityRecord]:
        """Return graph entities in the same order their names appear in a query."""
        normalized_query = normalize_entity_name(query)
        matches: list[tuple[int, int, EntityRecord]] = []
        for entity in self.graph.entities():
            positions = [
                normalized_query.find(normalized_name)
                for name in (entity.canonical_name, *entity.aliases)
                if (normalized_name := normalize_entity_name(name))
                and normalized_name in normalized_query
            ]
            if positions:
                longest_name = max(
                    len(normalize_entity_name(name))
                    for name in (entity.canonical_name, *entity.aliases)
                )
                matches.append((min(positions), -longest_name, entity))

        # 原因：短别名可能包含在较长实体名内，直接遍历会让端点顺序不稳定。
        # 作用：按查询位置排序，同位置优先更长名称，并去除同一实体的重复 mention。
        matches.sort(key=lambda item: (item[0], item[1], item[2].id))
        seen: set[str] = set()
        result: list[EntityRecord] = []
        for _, _, entity in matches:
            if entity.id in seen:
                continue
            seen.add(entity.id)
            result.append(entity)
        return result

    @property
    def _resolver(self) -> EntityResolver:
        if self.resolver is None:
            raise RuntimeError("entity resolver was not initialized")
        return self.resolver

    def _resolve_endpoint(
        self,
        name: str,
        evidence: GraphEvidence,
        candidates: dict[str, EntityCandidate],
        resolved: dict[str, EntityRecord],
    ) -> EntityRecord:
        normalized_name = normalize_entity_name(name)
        if normalized_name in resolved:
            return resolved[normalized_name]
        candidate = candidates.get(normalized_name)
        if candidate is None:
            # 原因：部分模型会返回有效关系，却漏掉关系端点的 entity 数组项。
            # 作用：使用同一条已验证关系证据补齐 UNKNOWN 实体，避免丢失真实关系。
            candidate = EntityCandidate(
                name=name,
                entity_type="UNKNOWN",
                evidence=evidence,
            )
        entity = self._resolver.resolve(candidate)
        resolved[normalized_name] = entity
        return entity


def _relation_id(source_id: str, relation: str, target_id: str) -> str:
    identity = f"{source_id}\n{relation.strip().casefold()}\n{target_id}"
    digest = blake2b(identity.encode("utf-8"), digest_size=12).hexdigest()
    return f"relation-{digest}"
