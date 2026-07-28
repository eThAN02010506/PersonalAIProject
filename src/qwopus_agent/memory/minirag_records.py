"""Deterministic MiniRAG document records and JSONL persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path
from typing import Any

from qwopus_agent.documents import build_document_structure, chunk_document_structure
from qwopus_agent.memory.graph_models import GraphChunk

INDEX_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class KnowledgeChunk:
    """One persisted retrieval unit with traceable source metadata."""

    id: str
    document_id: str
    source: str
    page: str | None
    page_end: str | None
    section_id: str
    section_path: tuple[str, ...]
    content: str
    position: int


@dataclass(frozen=True)
class DocumentRecord:
    """One original Markdown document and its deterministic chunks."""

    id: str
    timestamp: str
    document: str
    chunks: tuple[KnowledgeChunk, ...]


def build_record(document_id: str, document: str) -> DocumentRecord:
    """Convert one Markdown document into stable section-aware chunks."""
    source, markdown = _source_and_markdown(document)
    structure = chunk_document_structure(
        build_document_structure(
            markdown,
            source=source,
            document_id=document_id,
        )
    )
    chunks = [
        KnowledgeChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            source=chunk.source,
            page=str(chunk.page_start) if chunk.page_start is not None else None,
            page_end=str(chunk.page_end) if chunk.page_end is not None else None,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            content=chunk.content,
            position=chunk.position,
        )
        for chunk in structure.chunks
    ]
    return DocumentRecord(
        id=document_id,
        timestamp=datetime.now(UTC).isoformat(),
        document=document,
        chunks=tuple(chunks),
    )


def load_records(storage_path: Path) -> list[DocumentRecord]:
    """Load the latest valid record for every persisted source."""
    if not storage_path.exists():
        return []

    records: dict[str, DocumentRecord] = {}
    source_ids: dict[str, str] = {}
    for line in storage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        document = payload.get("document")
        if not isinstance(document, str) or not document.strip():
            continue
        source_documents = _split_source_documents(document)
        for source_document in source_documents:
            document_id = (
                str(payload.get("document_id"))
                if len(source_documents) == 1 and payload.get("document_id")
                else stable_id("document", source_document)
            )
            # 原因：旧版 JSONL 可能把多个文件合在一条记录，也没有可靠的 chunk 元数据。
            # 作用：加载时按来源拆分并确定性重建，使单文件更新/删除不会误伤其他文件。
            record = build_record(document_id, source_document)
            source = single_record_source(record)
            if source is not None:
                source_key = source.casefold()
                previous_id = source_ids.get(source_key)
                if previous_id is not None:
                    records.pop(previous_id, None)
                source_ids[source_key] = record.id
            records[record.id] = record
    return list(records.values())


def append_record(storage_path: Path, record: DocumentRecord) -> None:
    """Append one immutable source record to the JSONL fact store."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    with storage_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_record_payload(record), ensure_ascii=False) + "\n")


def rewrite_records(storage_path: Path, records: Sequence[DocumentRecord]) -> None:
    """Atomically replace the JSONL fact store after update or deletion."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = storage_path.with_suffix(f"{storage_path.suffix}.tmp")
    content = "".join(
        json.dumps(_record_payload(record), ensure_ascii=False) + "\n"
        for record in records
    )
    temporary_path.write_text(content, encoding="utf-8")
    # 原因：更新期间进程中断不能留下半截 JSONL，documents 是索引事实来源。
    # 作用：完整写入临时文件后原子替换，向量与图谱可随时从它恢复。
    temporary_path.replace(storage_path)


def record_sources(record: DocumentRecord) -> set[str]:
    """Return all source names represented by one record."""
    return {chunk.source for chunk in record.chunks}


def single_record_source(record: DocumentRecord) -> str | None:
    """Return the sole known source when a record has exactly one."""
    sources = record_sources(record) - {"unknown"}
    return next(iter(sources)) if len(sources) == 1 else None


def stable_id(prefix: str, value: str) -> str:
    """Build a deterministic identifier without exposing document text."""
    digest = blake2b(value.encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}-{digest}"


def to_graph_chunks(chunks: Sequence[KnowledgeChunk]) -> tuple[GraphChunk, ...]:
    """Project retrieval chunks onto the knowledge-graph ingestion contract."""
    return tuple(
        GraphChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            source=chunk.source,
            page=chunk.page,
            section_id=chunk.section_id,
            content=chunk.content,
        )
        for chunk in chunks
    )


def _source_and_markdown(document: str) -> tuple[str, str]:
    match = re.match(r"\s*# File:\s*(.+?)\s*$", document, re.MULTILINE)
    if not match:
        return "unknown", document.strip()
    return match.group(1).strip(), document[match.end():].strip()


def _split_source_documents(document: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^# File:\s*(.+?)\s*$", document))
    if len(matches) <= 1:
        return [document]
    documents: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        source_document = document[match.start():end].strip()
        if source_document:
            documents.append(source_document)
    return documents


def _record_payload(record: DocumentRecord) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "document_id": record.id,
        "timestamp": record.timestamp,
        "document": record.document,
        "chunks": [asdict(chunk) for chunk in record.chunks],
    }
