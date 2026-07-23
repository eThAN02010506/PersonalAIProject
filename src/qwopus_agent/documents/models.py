"""Typed document structure shared by parsers, retrieval, tools, and APIs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocumentSection(BaseModel):
    """One heading-delimited section in a normalized document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    level: int = Field(ge=0, le=6)
    parent_id: str | None = None
    section_path: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    content: str = ""


class DocumentChunk(BaseModel):
    """One bounded evidence unit that never crosses a section boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    document_id: str
    source: str
    section_id: str
    section_path: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    content: str
    token_count: int = Field(ge=1)
    position: int = Field(ge=0)


class DocumentStructure(BaseModel):
    """Complete normalized hierarchy and retrieval chunks for one document version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    source: str
    sections: tuple[DocumentSection, ...]
    chunks: tuple[DocumentChunk, ...] = ()
