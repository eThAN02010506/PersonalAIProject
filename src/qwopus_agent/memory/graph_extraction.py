"""Model-agnostic extraction of evidence-backed entities and relations."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qwopus_agent.llm.base import BaseLLM, ChatMessage
from qwopus_agent.memory.graph_models import (
    EntityCandidate,
    GraphChunk,
    GraphEvidence,
    GraphExtraction,
    RelationCandidate,
)

logger = logging.getLogger(__name__)


class GraphExtractionError(RuntimeError):
    """Raised when an LLM does not return a usable extraction payload."""


class GraphExtractor(Protocol):
    """Dependency-injection contract for graph extraction implementations."""

    def extract(self, chunks: Sequence[GraphChunk]) -> GraphExtraction:
        """Extract entities and directed relations from document chunks."""


@dataclass(frozen=True)
class CompositeGraphExtractor:
    """Merge independent extractors while isolating one provider failure."""

    extractors: tuple[GraphExtractor, ...]

    def extract(self, chunks: Sequence[GraphChunk]) -> GraphExtraction:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        for extractor in self.extractors:
            try:
                extraction = extractor.extract(chunks)
            except Exception:
                # 原因：LLM 服务可能暂时离线，但确定性抽取和向量入库仍应继续工作。
                # 作用：隔离单个 extractor 故障，并在日志中保留可诊断的完整异常。
                logger.exception(
                    "graph_extractor_failed extractor=%s",
                    type(extractor).__name__,
                )
                continue
            entities.extend(extraction.entities)
            relations.extend(extraction.relations)
        return _deduplicate_extraction(entities, relations)


@dataclass(frozen=True)
class RuleBasedGraphExtractor:
    """Extract deterministic relations written in an explicit portable syntax."""

    _pattern = re.compile(
        r"\[\[(?P<source>[^\]|]+)(?:\|(?P<source_type>[^\]]+))?\]\]"
        r"\s*-\[(?P<relation>[^\]]+)\]->\s*"
        r"\[\[(?P<target>[^\]|]+)(?:\|(?P<target_type>[^\]]+))?\]\]"
    )

    def extract(self, chunks: Sequence[GraphChunk]) -> GraphExtraction:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        for chunk in chunks:
            for match in self._pattern.finditer(chunk.content):
                evidence = _evidence(chunk, match.group(0))
                source = match.group("source").strip()
                target = match.group("target").strip()
                entities.extend(
                    (
                        EntityCandidate(
                            name=source,
                            entity_type=(match.group("source_type") or "UNKNOWN").strip(),
                            evidence=evidence,
                        ),
                        EntityCandidate(
                            name=target,
                            entity_type=(match.group("target_type") or "UNKNOWN").strip(),
                            evidence=evidence,
                        ),
                    )
                )
                relations.append(
                    RelationCandidate(
                        source=source,
                        relation=match.group("relation").strip(),
                        target=target,
                        evidence=evidence,
                    )
                )
        return _deduplicate_extraction(entities, relations)


@dataclass(frozen=True)
class LLMGraphExtractor:
    """Extract graph facts through any implementation of the BaseLLM contract."""

    llm_factory: Callable[[], BaseLLM]
    max_batch_characters: int = 12_000
    max_output_tokens: int = 3_000

    def extract(self, chunks: Sequence[GraphChunk]) -> GraphExtraction:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        for batch in _chunk_batches(chunks, self.max_batch_characters):
            response = self.llm_factory().generate(
                [
                    ChatMessage(role="system", content=_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=_render_chunks(batch)),
                ],
                temperature=0.0,
                max_tokens=self.max_output_tokens,
            )
            payload = _parse_json_object(response.content)
            batch_entities, batch_relations = _validated_candidates(payload, batch)
            entities.extend(batch_entities)
            relations.extend(batch_relations)
        return _deduplicate_extraction(entities, relations)


_SYSTEM_PROMPT = """Extract a knowledge graph from the supplied document chunks.
Return one JSON object only, with this exact shape:
{
  "entities": [
    {"name": "...", "entity_type": "...", "aliases": [], "description": "...",
     "chunk_id": "...", "evidence": "exact quote from that chunk"}
  ],
  "relations": [
    {"source": "...", "relation": "...", "target": "...", "confidence": 0.0,
     "chunk_id": "...", "evidence": "exact quote containing source and target"}
  ]
}
Use the document's original language for names and relation labels. Extract only facts directly
supported by an exact quote. Never infer a relation that is not stated in the supplied text.
"""


def _chunk_batches(
    chunks: Sequence[GraphChunk],
    max_characters: int,
) -> list[tuple[GraphChunk, ...]]:
    if max_characters < 1:
        raise ValueError("max_batch_characters must be positive")
    batches: list[tuple[GraphChunk, ...]] = []
    current: list[GraphChunk] = []
    current_size = 0
    for chunk in chunks:
        chunk_size = len(chunk.content)
        if current and current_size + chunk_size > max_characters:
            batches.append(tuple(current))
            current = []
            current_size = 0
        current.append(chunk)
        current_size += chunk_size
    if current:
        batches.append(tuple(current))
    return batches


def _render_chunks(chunks: Sequence[GraphChunk]) -> str:
    payload = [
        {
            "chunk_id": chunk.id,
            "source": chunk.source,
            "page": chunk.page,
            "content": chunk.content,
        }
        for chunk in chunks
    ]
    return json.dumps(payload, ensure_ascii=False)


def _parse_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    if start < 0:
        raise GraphExtractionError("graph extractor returned no JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError as exc:
        raise GraphExtractionError("graph extractor returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GraphExtractionError("graph extractor JSON must be an object")
    return payload


def _validated_candidates(
    payload: dict[str, Any],
    chunks: Sequence[GraphChunk],
) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    entities: list[EntityCandidate] = []
    relations: list[RelationCandidate] = []

    for item in _object_items(payload.get("entities")):
        chunk = chunks_by_id.get(str(item.get("chunk_id", "")))
        evidence_text = str(item.get("evidence", "")).strip()
        name = str(item.get("name", "")).strip()
        if chunk is None or not name or not _supports(chunk.content, evidence_text, (name,)):
            continue
        aliases = tuple(
            str(alias).strip()
            for alias in item.get("aliases", [])
            if str(alias).strip()
        )
        entities.append(
            EntityCandidate(
                name=name,
                entity_type=str(item.get("entity_type", "UNKNOWN")).strip() or "UNKNOWN",
                aliases=aliases,
                description=str(item.get("description", "")).strip(),
                evidence=_evidence(chunk, evidence_text),
            )
        )

    for item in _object_items(payload.get("relations")):
        chunk = chunks_by_id.get(str(item.get("chunk_id", "")))
        evidence_text = str(item.get("evidence", "")).strip()
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip()
        required_names = (source, target)
        # 原因：LLM 可能生成看似合理、但原文不存在的关系或伪造 chunk_id。
        # 作用：只有能回溯到输入 chunk 且引文同时包含两个端点的关系才进入图谱。
        if (
            chunk is None
            or not source
            or not target
            or not relation
            or not _supports(chunk.content, evidence_text, required_names)
        ):
            continue
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        relations.append(
            RelationCandidate(
                source=source,
                relation=relation,
                target=target,
                confidence=max(0.0, min(1.0, confidence)),
                evidence=_evidence(chunk, evidence_text),
            )
        )
    return entities, relations


def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _supports(content: str, evidence: str, names: tuple[str, ...]) -> bool:
    normalized_content = _normalize_text(content)
    normalized_evidence = _normalize_text(evidence)
    if not normalized_evidence or normalized_evidence not in normalized_content:
        return False
    return all(_normalize_text(name) in normalized_evidence for name in names)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _evidence(chunk: GraphChunk, text: str) -> GraphEvidence:
    return GraphEvidence(
        document_id=chunk.document_id,
        source=chunk.source,
        page=chunk.page,
        chunk_id=chunk.id,
        text=text,
    )


def _deduplicate_extraction(
    entities: Sequence[EntityCandidate],
    relations: Sequence[RelationCandidate],
) -> GraphExtraction:
    unique_entities = {
        (
            entity.name.casefold(),
            entity.entity_type.casefold(),
            entity.evidence.chunk_id,
        ): entity
        for entity in entities
    }
    unique_relations = {
        (
            relation.source.casefold(),
            relation.relation.casefold(),
            relation.target.casefold(),
            relation.evidence.chunk_id,
        ): relation
        for relation in relations
    }
    return GraphExtraction(
        entities=tuple(unique_entities.values()),
        relations=tuple(unique_relations.values()),
    )
