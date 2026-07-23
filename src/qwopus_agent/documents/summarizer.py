"""Hierarchical map-reduce summaries that retain source chunk references."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from qwopus_agent.documents.models import DocumentStructure
from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens

_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")


class SectionSummary(BaseModel):
    """One reduced section summary with the exact chunks that support it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section_id: str
    section_path: tuple[str, ...]
    summary: str
    chunk_ids: tuple[str, ...]


class HierarchicalDocumentSummary(BaseModel):
    """Map summaries reduced to sections and one bounded document overview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    source: str
    section_summaries: tuple[SectionSummary, ...]
    document_summary: str


def summarize_document(
    structure: DocumentStructure,
    *,
    map_tokens: int = 180,
    section_tokens: int = 600,
    document_tokens: int = 3000,
) -> HierarchicalDocumentSummary:
    """Summarize every chunk, then reduce by section and top-level chapter."""
    mapped = {
        chunk.id: _extractive_summary(chunk.content, max_tokens=map_tokens)
        for chunk in structure.chunks
    }
    section_summaries: list[SectionSummary] = []
    for section in structure.sections:
        section_chunks = [
            chunk
            for chunk in structure.chunks
            if chunk.section_path[: len(section.section_path)] == section.section_path
        ]
        if not section_chunks:
            continue
        reduced = _balanced_reduce(
            [mapped[chunk.id] for chunk in section_chunks],
            max_tokens=section_tokens,
        )
        section_summaries.append(
            SectionSummary(
                section_id=section.id,
                section_path=section.section_path,
                summary=reduced,
                chunk_ids=tuple(chunk.id for chunk in section_chunks),
            )
        )

    top_level = [
        summary
        for summary in section_summaries
        if len(summary.section_path) == 1
    ]
    if not top_level:
        top_level = section_summaries
    document_parts = [
        f"## {' / '.join(summary.section_path)}\n{summary.summary}"
        for summary in top_level
    ]
    # 原因：全文总结必须覆盖所有章节，直接截取拼接结果会再次偏向文件开头。
    # 作用：对每个章节平均分配 reduce 预算，使文档末尾章节也进入最终上下文。
    document_summary = _balanced_reduce(document_parts, max_tokens=document_tokens)
    return HierarchicalDocumentSummary(
        document_id=structure.document_id,
        source=structure.source,
        section_summaries=tuple(section_summaries),
        document_summary=document_summary,
    )


def _balanced_reduce(parts: list[str], *, max_tokens: int) -> str:
    non_empty = [part.strip() for part in parts if part.strip()]
    if not non_empty:
        return ""
    if sum(estimate_tokens(part) for part in non_empty) <= max_tokens:
        return "\n\n".join(non_empty)
    per_part = max(24, max_tokens // len(non_empty))
    return "\n\n".join(truncate_to_tokens(part, per_part) for part in non_empty)


def _extractive_summary(text: str, *, max_tokens: int) -> str:
    compact = text.strip()
    if estimate_tokens(compact) <= max_tokens:
        return compact
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BREAK.split(compact)
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return truncate_to_tokens(compact, max_tokens)
    first_budget = max_tokens // 2
    last_budget = max_tokens - first_budget
    return (
        truncate_to_tokens(sentences[0], first_budget)
        + "\n...\n"
        + truncate_to_tokens(sentences[-1], last_budget)
    )
