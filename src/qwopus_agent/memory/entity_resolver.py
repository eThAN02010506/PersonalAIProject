"""Canonicalize and merge graph entities across uploaded documents."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_models import EntityCandidate, EntityRecord


class EntityEmbeddingBackend(Protocol):
    """Small injected contract used only for optional semantic entity matching."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode entity names into comparable vectors."""


@dataclass
class EntityResolver:
    """Resolve extracted mentions to stable entities in one persistent graph."""

    graph: PersistentKnowledgeGraph
    embedding_backend: EntityEmbeddingBackend | None = None
    semantic_threshold: float = 0.92

    def resolve(self, candidate: EntityCandidate) -> EntityRecord:
        """Find or create one canonical entity, then persist its new evidence."""
        current = self.find(candidate.name, candidate.entity_type)
        if current is None:
            entity = EntityRecord(
                id=_entity_id(candidate.name, candidate.entity_type),
                canonical_name=candidate.name.strip(),
                entity_type=candidate.entity_type.strip() or "UNKNOWN",
                aliases=_unique_aliases(candidate.name, candidate.aliases),
                description=candidate.description,
                evidence=(candidate.evidence,),
            )
        else:
            aliases = _unique_aliases(
                current.canonical_name,
                (*current.aliases, candidate.name, *candidate.aliases),
            )
            entity = current.model_copy(
                update={
                    "aliases": aliases,
                    "description": current.description or candidate.description,
                    "entity_type": (
                        candidate.entity_type
                        if current.entity_type.casefold() == "unknown"
                        else current.entity_type
                    ),
                    "evidence": (*current.evidence, candidate.evidence),
                }
            )
        return self.graph.upsert_entity(entity)

    def find(self, name: str, entity_type: str = "UNKNOWN") -> EntityRecord | None:
        """Match an entity exactly first, then semantically when configured."""
        normalized_name = normalize_entity_name(name)
        compatible = [
            entity
            for entity in self.graph.entities()
            if _types_are_compatible(entity.entity_type, entity_type)
        ]
        for entity in compatible:
            names = (entity.canonical_name, *entity.aliases)
            if normalized_name in {normalize_entity_name(value) for value in names}:
                return entity

        if self.embedding_backend is None or not compatible:
            return None
        return self._semantic_match(name, compatible, self.embedding_backend)

    def _semantic_match(
        self,
        name: str,
        entities: Sequence[EntityRecord],
        embedding_backend: EntityEmbeddingBackend,
    ) -> EntityRecord | None:
        candidate_names = [name, *(entity.canonical_name for entity in entities)]
        vectors = np.asarray(embedding_backend.encode(candidate_names), dtype=np.float32)
        if vectors.shape[0] != len(candidate_names):
            return None
        query = _normalized_vector(vectors[0])
        scores = [float(np.dot(query, _normalized_vector(vector))) for vector in vectors[1:]]
        if not scores:
            return None
        best_index = int(np.argmax(scores))
        # 原因：相似名称并不总是同一实体，低阈值会把独立公司或人物错误合并。
        # 作用：只接受高置信语义匹配；其余 mention 会创建为独立实体并保留证据。
        if scores[best_index] < self.semantic_threshold:
            return None
        return entities[best_index]


def normalize_entity_name(value: str) -> str:
    """Normalize Unicode, case, spaces, and punctuation for exact identity matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _types_are_compatible(left: str, right: str) -> bool:
    normalized_left = left.strip().casefold() or "unknown"
    normalized_right = right.strip().casefold() or "unknown"
    return (
        normalized_left == normalized_right
        or normalized_left == "unknown"
        or normalized_right == "unknown"
    )


def _entity_id(name: str, entity_type: str) -> str:
    identity = f"{normalize_entity_name(name)}\n{entity_type.strip().casefold()}"
    digest = blake2b(identity.encode("utf-8"), digest_size=12).hexdigest()
    return f"entity-{digest}"


def _unique_aliases(canonical_name: str, aliases: Sequence[str]) -> tuple[str, ...]:
    canonical = normalize_entity_name(canonical_name)
    seen = {canonical}
    result: list[str] = []
    for alias in aliases:
        clean_alias = alias.strip()
        normalized = normalize_entity_name(clean_alias)
        if not clean_alias or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(clean_alias)
    return tuple(result)


def _normalized_vector(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm
