"""Section-aware token chunking for normalized documents."""

from __future__ import annotations

import re
from hashlib import blake2b

from qwopus_agent.documents.models import DocumentChunk, DocumentStructure
from qwopus_agent.utils.token_budget import estimate_tokens

DEFAULT_CHUNK_TOKENS = 700
DEFAULT_CHUNK_OVERLAP_TOKENS = 80
_TOKEN_SPAN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def chunk_document_structure(
    structure: DocumentStructure,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> DocumentStructure:
    """Attach bounded chunks without ever joining content from different sections."""
    if max_tokens < 32:
        raise ValueError("max_tokens must be at least 32")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens")

    chunks: list[DocumentChunk] = []
    for section in structure.sections:
        for content in _split_section(
            section.content,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ):
            chunks.append(
                DocumentChunk(
                    id=_stable_id(
                        "chunk",
                        f"{structure.document_id}\n{section.id}\n{len(chunks)}\n{content}",
                    ),
                    document_id=structure.document_id,
                    source=structure.source,
                    section_id=section.id,
                    section_path=section.section_path,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    content=content,
                    token_count=estimate_tokens(content),
                    position=len(chunks),
                )
            )
    if not chunks and structure.sections:
        section = structure.sections[-1]
        # 原因：标题型便签可能没有正文，旧知识库仍允许通过标题找到这种文档。
        # 作用：只在整份文档完全无正文时索引标题，不给正常父章节添加重复 Chunk。
        chunks.append(
            DocumentChunk(
                id=_stable_id("chunk", f"{structure.document_id}\n{section.id}\n{section.title}"),
                document_id=structure.document_id,
                source=structure.source,
                section_id=section.id,
                section_path=section.section_path,
                page_start=section.page_start,
                page_end=section.page_end,
                content=section.title,
                token_count=estimate_tokens(section.title),
                position=0,
            )
        )
    return structure.model_copy(update={"chunks": tuple(chunks)})


def _split_section(
    content: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    compact = content.strip()
    spans = list(_TOKEN_SPAN.finditer(compact))
    if not spans:
        return []
    if estimate_tokens(compact) <= max_tokens:
        return [compact]

    chunks: list[str] = []
    start_token = 0
    while start_token < len(spans):
        start_char = spans[start_token].start()
        end_token = _fitting_end_token(
            compact,
            spans,
            start_token=start_token,
            max_tokens=max_tokens,
        )
        end_char = spans[end_token - 1].end()
        if end_token < len(spans):
            paragraph_break = compact.rfind("\n\n", start_char, end_char)
            if paragraph_break > spans[min(start_token + max_tokens // 2, end_token - 1)].start():
                end_char = paragraph_break
                while end_token > start_token and spans[end_token - 1].start() >= end_char:
                    end_token -= 1
        chunk = compact[start_char:end_char].strip()
        if chunk:
            chunks.append(chunk)
        if end_token >= len(spans):
            break
        start_token = max(end_token - overlap_tokens, start_token + 1)
    return chunks


def _fitting_end_token(
    text: str,
    spans: list[re.Match[str]],
    *,
    start_token: int,
    max_tokens: int,
) -> int:
    low = start_token + 1
    high = min(start_token + max_tokens, len(spans))
    start_char = spans[start_token].start()
    while low < high:
        middle = (low + high + 1) // 2
        end_char = spans[middle - 1].end()
        if estimate_tokens(text[start_char:end_char]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _stable_id(kind: str, value: str) -> str:
    digest = blake2b(value.encode("utf-8"), digest_size=12).hexdigest()
    return f"{kind}-{digest}"
