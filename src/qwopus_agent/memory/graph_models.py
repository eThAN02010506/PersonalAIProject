"""Typed records shared by knowledge-graph ingestion, retrieval, and UI layers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GraphChunk(BaseModel):
    """One bounded document passage supplied to a graph extractor."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    source: str
    page: str | None = None
    content: str


class GraphEvidence(BaseModel):
    """One source passage that proves an entity or relation."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source: str
    page: str | None = None
    chunk_id: str
    text: str


class EntityCandidate(BaseModel):
    """Entity extracted from one document chunk before canonicalization."""

    model_config = ConfigDict(frozen=True)

    name: str
    entity_type: str = "UNKNOWN"
    aliases: tuple[str, ...] = ()
    description: str = ""
    evidence: GraphEvidence


class RelationCandidate(BaseModel):
    """Directed relation extracted from one evidence passage."""

    model_config = ConfigDict(frozen=True)

    source: str
    relation: str
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: GraphEvidence


class GraphExtraction(BaseModel):
    """Validated extraction result produced for one or more chunks."""

    model_config = ConfigDict(frozen=True)

    entities: tuple[EntityCandidate, ...] = ()
    relations: tuple[RelationCandidate, ...] = ()


class EntityRecord(BaseModel):
    """Canonical entity persisted in the global knowledge graph."""

    model_config = ConfigDict(frozen=True)

    id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    evidence: tuple[GraphEvidence, ...] = ()


class RelationRecord(BaseModel):
    """Canonical directed edge with all supporting source passages."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_id: str
    relation: str
    target_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[GraphEvidence, ...] = ()


class GraphPath(BaseModel):
    """One bounded entity path returned to the Agent as structured evidence."""

    model_config = ConfigDict(frozen=True)

    entity_ids: tuple[str, ...]
    entity_names: tuple[str, ...]
    relations: tuple[RelationRecord, ...]
    evidence: tuple[GraphEvidence, ...]


class GraphNodeView(BaseModel):
    """Read-only node projection used by the graph visualization page."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    entity_type: str
    aliases: tuple[str, ...] = ()


class GraphEdgeView(BaseModel):
    """Read-only edge projection used by the graph visualization page."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    target: str
    label: str
    confidence: float
    evidence_count: int


class GraphSnapshot(BaseModel):
    """Bounded graph projection safe for UI rendering."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[GraphNodeView, ...] = ()
    edges: tuple[GraphEdgeView, ...] = ()
