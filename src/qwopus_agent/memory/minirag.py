"""Persistent semantic knowledge facade backed by MiniRAG's vector storage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from minirag.kg.nano_vector_db_impl import NanoVectorDBStorage
from minirag.utils import EmbeddingFunc

from qwopus_agent.memory.entity_resolver import EntityResolver
from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import GraphExtractor, RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphChunk, GraphPath
from qwopus_agent.memory.knowledge_graph import (
    DEFAULT_KNOWLEDGE_GRAPH_PATH,
    KnowledgeGraphIndex,
)

DEFAULT_MINIRAG_STORE_PATH = Path("storage/minirag/documents.jsonl")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_SCHEMA_VERSION = 2
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
SEARCH_TOP_K = 5
SEARCH_CANDIDATE_K = 30
COSINE_THRESHOLD = 0.25
GRAPH_SEARCH_LIMIT = 3

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingBackend(Protocol):
    """Internal contract that keeps document retrieval independent from the chat model."""

    model_name: str
    dimensions: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text into normalized semantic vectors."""


@dataclass
class SentenceTransformerEmbedding:
    """Local multilingual embedding model used only by the knowledge layer."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    _model: Any = field(default=None, init=False, repr=False)

    @property
    def dimensions(self) -> int:
        """Read the actual model dimension instead of assuming one vector shape."""
        return int(self._get_model().get_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts locally without using the configured chat LLM."""
        return np.asarray(
            self._get_model().encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_sentence_transformer(self.model_name)
        return self._model


@dataclass(frozen=True)
class _KnowledgeChunk:
    """One persisted retrieval unit with traceable source metadata."""

    id: str
    document_id: str
    source: str
    page: str | None
    content: str
    position: int


@dataclass(frozen=True)
class _DocumentRecord:
    """One original Markdown document and its deterministic chunks."""

    id: str
    timestamp: str
    document: str
    chunks: tuple[_KnowledgeChunk, ...]


@dataclass
class MiniRAG:
    """Expose only document insertion and semantic search to the Agent."""

    storage_path: Path = DEFAULT_MINIRAG_STORE_PATH
    embedding_backend: EmbeddingBackend | None = field(default=None, repr=False)
    graph_extractor: GraphExtractor | None = field(default=None, repr=False)
    graph_storage_path: Path | None = field(default=None, repr=False)
    _records: list[_DocumentRecord] = field(default_factory=list, init=False, repr=False)
    _chunks: dict[str, _KnowledgeChunk] = field(default_factory=dict, init=False, repr=False)
    _vector_store: NanoVectorDBStorage = field(init=False, repr=False)
    _graph: PersistentKnowledgeGraph = field(init=False, repr=False)
    _graph_index: KnowledgeGraphIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load documents and the persisted MiniRAG vector index on startup."""
        self.storage_path = Path(self.storage_path)
        self.embedding_backend = self.embedding_backend or SentenceTransformerEmbedding(
            model_name=os.getenv("QWOPUS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        self.graph_storage_path = self.graph_storage_path or _graph_path(self.storage_path)
        self.graph_extractor = self.graph_extractor or RuleBasedGraphExtractor()
        self._initialize_graph_index()
        self._records = _load_records(self.storage_path)
        self._chunks = {
            chunk.id: chunk
            for record in self._records
            for chunk in record.chunks
        }
        self._vector_store = self._create_vector_store()
        self._synchronize_vector_index()

    def insert(self, document: str) -> None:
        """Insert one Markdown-normalized document into persistent semantic memory."""
        if not document.strip():
            raise ValueError("document must not be empty")

        document_id = _stable_id("document", document)
        if any(record.id == document_id for record in self._records):
            return

        record = _build_record(document_id, document)
        source = _single_record_source(record)
        if source is not None:
            replaced = [
                current
                for current in self._records
                if source.casefold()
                in {item.casefold() for item in _record_sources(current)}
            ]
            if replaced:
                # 原因：同名文件再次上传代表新版本，旧向量和旧关系不能继续参与回答。
                # 作用：在写入新版本前清除该来源的旧派生数据和 JSONL 记录。
                self._remove_records(replaced)
        _append_record(self.storage_path, record)
        self._records.append(record)
        self._chunks.update((chunk.id, chunk) for chunk in record.chunks)
        self._upsert_chunks(record.chunks)
        try:
            # 原因：向量检索和关系检索需要共享同一批带文件/页码信息的 chunk。
            # 作用：每次 insert 同步增量构图，Agent 仍只依赖 MiniRAG.insert(document)。
            self._graph_index.insert(_to_graph_chunks(record.chunks))
        except Exception:
            # 原因：远程 LLM 抽取失败不应破坏已经完成的本地文档与向量入库。
            # 作用：记录完整异常并保留向量搜索能力，之后可通过重建操作补齐图谱。
            logger.exception("knowledge_graph_ingestion_failed document_id=%s", document_id)

    def search(self, query: str) -> list[str]:
        """Return graph paths first, then complementary semantic chunks."""
        if not query.strip():
            raise ValueError("query must not be empty")

        graph_paths = self._graph_index.search(query, limit=GRAPH_SEARCH_LIMIT)
        results = [_render_graph_search_result(path) for path in graph_paths]
        graph_chunk_ids = {
            evidence.chunk_id
            for path in graph_paths
            for evidence in path.evidence
        }
        if not self._chunks or len(results) >= SEARCH_TOP_K:
            return results[:SEARCH_TOP_K]

        matches = _run_coroutine(
            self._vector_store.query(query, top_k=SEARCH_CANDIDATE_K)
        )
        ranked_chunks: list[_KnowledgeChunk] = []
        for match in matches:
            chunk = self._chunks.get(str(match.get("id", "")))
            if chunk is not None and chunk.id not in graph_chunk_ids:
                ranked_chunks.append(chunk)
        vector_results = [
            _render_search_result(chunk)
            for chunk in _diverse_chunks(ranked_chunks)
        ]
        return (results + vector_results)[:SEARCH_TOP_K]

    def _list_sources(self) -> list[str]:
        """Return indexed file sources for the separate maintenance service."""
        return sorted(
            {
                source
                for record in self._records
                for source in _record_sources(record)
                if source != "unknown"
            },
            key=str.casefold,
        )

    def _delete_source(self, source: str) -> int:
        """Delete all records for one exact file source from every index."""
        normalized_source = source.strip().casefold()
        if not normalized_source:
            raise ValueError("source must not be empty")
        records = [
            record
            for record in self._records
            if normalized_source in {item.casefold() for item in _record_sources(record)}
        ]
        if not records:
            return 0
        self._remove_records(records)
        logger.info("knowledge_source_deleted source=%s records=%s", source, len(records))
        return len(records)

    def _rebuild_indexes(self) -> None:
        """Recreate vector and graph indexes from persisted Markdown records."""
        vector_path = _index_directory(self.storage_path) / "vdb_qwopus_chunks.json"
        vector_path.unlink(missing_ok=True)
        self._vector_store = self._create_vector_store()
        if self._chunks:
            self._upsert_chunks(tuple(self._chunks.values()))

        Path(self.graph_storage_path).unlink(missing_ok=True)
        self._initialize_graph_index()
        for record in self._records:
            try:
                self._graph_index.insert(_to_graph_chunks(record.chunks))
            except Exception:
                logger.exception(
                    "knowledge_graph_rebuild_failed document_id=%s",
                    record.id,
                )
        logger.info(
            "knowledge_indexes_rebuilt documents=%s chunks=%s",
            len(self._records),
            len(self._chunks),
        )

    def _initialize_graph_index(self) -> None:
        if self.graph_extractor is None or self.graph_storage_path is None:
            raise RuntimeError("graph dependencies were not initialized")
        self._graph = PersistentKnowledgeGraph(Path(self.graph_storage_path))
        self._graph_index = KnowledgeGraphIndex(
            graph=self._graph,
            extractor=self.graph_extractor,
            resolver=EntityResolver(
                graph=self._graph,
                embedding_backend=self.embedding_backend,
            ),
        )

    def _remove_records(self, records: Sequence[_DocumentRecord]) -> None:
        record_ids = {record.id for record in records}
        chunk_ids = [chunk.id for record in records for chunk in record.chunks]
        self._records = [record for record in self._records if record.id not in record_ids]
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id not in record_ids
        }
        if chunk_ids:
            _run_coroutine(self._vector_store.delete(chunk_ids))
            _run_coroutine(self._vector_store.index_done_callback())
        for document_id in record_ids:
            self._graph.remove_document(document_id)
        _rewrite_records(self.storage_path, self._records)

    def _create_vector_store(self) -> NanoVectorDBStorage:
        """Create MiniRAG's local NanoVectorDB adapter."""
        index_dir = _index_directory(self.storage_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        expected_config = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": self.embedding_backend.model_name,
            "embedding_dimensions": self.embedding_backend.dimensions,
        }
        config_path = index_dir / "index_config.json"
        vector_path = index_dir / "vdb_qwopus_chunks.json"
        if _load_json(config_path) != expected_config:
            # 原因：不同 embedding 模型的向量不能放在同一个索引中比较。
            # 作用：模型或 schema 改变时只重建派生索引，原始 documents.jsonl 不会丢失。
            vector_path.unlink(missing_ok=True)
            config_path.write_text(
                json.dumps(expected_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        async def embed(texts: list[str]) -> np.ndarray:
            return self.embedding_backend.encode(texts)

        embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding_backend.dimensions,
            max_token_size=512,
            func=embed,
        )
        return NanoVectorDBStorage(
            namespace="qwopus_chunks",
            global_config={
                "working_dir": str(index_dir),
                "embedding_batch_num": 16,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": COSINE_THRESHOLD,
                },
            },
            embedding_func=embedding_func,
            meta_fields={"document_id", "source", "page", "position"},
        )

    def _synchronize_vector_index(self) -> None:
        """Rebuild missing or stale vectors from persisted source documents."""
        stored_ids = {
            str(item.get("__id__", ""))
            for item in self._vector_store.client_storage.get("data", [])
        }
        expected_ids = set(self._chunks)
        if stored_ids - expected_ids:
            vector_path = _index_directory(self.storage_path) / "vdb_qwopus_chunks.json"
            vector_path.unlink(missing_ok=True)
            self._vector_store = self._create_vector_store()
            stored_ids = set()

        missing_chunks = tuple(
            chunk for chunk_id, chunk in self._chunks.items() if chunk_id not in stored_ids
        )
        if missing_chunks:
            # 原因：documents.jsonl 是事实来源，向量文件只是可以重新生成的派生数据。
            # 作用：应用重启或索引文件丢失后自动恢复，不要求用户重新上传文档。
            self._upsert_chunks(missing_chunks)

    def _upsert_chunks(self, chunks: Sequence[_KnowledgeChunk]) -> None:
        payload = {
            chunk.id: {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page or "",
                "position": chunk.position,
            }
            for chunk in chunks
        }
        _run_coroutine(self._vector_store.upsert(payload))
        _run_coroutine(self._vector_store.index_done_callback())


def _build_record(document_id: str, document: str) -> _DocumentRecord:
    chunks: list[_KnowledgeChunk] = []
    position = 0
    for source, page, section in _source_sections(document):
        for content in _chunk_text(section):
            chunks.append(
                _KnowledgeChunk(
                    id=_stable_id("chunk", f"{document_id}\n{source}\n{page}\n{content}"),
                    document_id=document_id,
                    source=source,
                    page=page,
                    content=content,
                    position=position,
                )
            )
            position += 1
    return _DocumentRecord(
        id=document_id,
        timestamp=datetime.now(UTC).isoformat(),
        document=document,
        chunks=tuple(chunks),
    )


def _source_sections(document: str) -> list[tuple[str, str | None, str]]:
    """Split combined Markdown by file and page while retaining citations."""
    file_pattern = re.compile(r"(?m)^# File:\s*(.+?)\s*$")
    file_matches = list(file_pattern.finditer(document))
    file_sections: list[tuple[str, str]] = []
    if not file_matches:
        file_sections.append(("unknown", document.strip()))
    else:
        for index, match in enumerate(file_matches):
            end = (
                file_matches[index + 1].start()
                if index + 1 < len(file_matches)
                else len(document)
            )
            file_sections.append((match.group(1).strip(), document[match.end():end].strip()))

    sections: list[tuple[str, str | None, str]] = []
    page_pattern = re.compile(r"(?im)^#{1,4}\s+(?:Page\s+(\d+)|第\s*(\d+)\s*页)\s*$")
    for source, file_content in file_sections:
        page_matches = list(page_pattern.finditer(file_content))
        if not page_matches:
            sections.append((source, None, file_content))
            continue
        for index, match in enumerate(page_matches):
            end = (
                page_matches[index + 1].start()
                if index + 1 < len(page_matches)
                else len(file_content)
            )
            page = match.group(1) or match.group(2)
            sections.append((source, page, file_content[match.end():end].strip()))
    return [(source, page, text) for source, page, text in sections if text]


def _chunk_text(text: str) -> list[str]:
    """Create bounded overlapping Markdown chunks for embedding."""
    compact = text.strip()
    if len(compact) <= CHUNK_SIZE:
        return [compact] if compact else []

    chunks: list[str] = []
    start = 0
    while start < len(compact):
        end = min(start + CHUNK_SIZE, len(compact))
        if end < len(compact):
            paragraph_break = compact.rfind("\n\n", start + CHUNK_SIZE // 2, end)
            if paragraph_break > start:
                end = paragraph_break
        chunk = compact[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(compact):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _render_search_result(chunk: _KnowledgeChunk) -> str:
    if chunk.source == "unknown" and chunk.page is None:
        return chunk.content
    citation = f"Source: {chunk.source}"
    if chunk.page is not None:
        citation += f" | Page: {chunk.page}"
    return f"[{citation}]\n{chunk.content}"


def _render_graph_search_result(path: GraphPath) -> str:
    names_by_id = dict(zip(path.entity_ids, path.entity_names, strict=False))
    relation_lines = [
        (
            f"- {names_by_id[relation.source_id]} -[{relation.relation}]-> "
            f"{names_by_id[relation.target_id]}"
        )
        for relation in path.relations
    ]
    evidence_lines: list[str] = []
    for evidence in path.evidence:
        citation = f"Source: {evidence.source}"
        if evidence.page is not None:
            citation += f" | Page: {evidence.page}"
        evidence_lines.append(f"- [{citation}] {evidence.text}")
    return (
        "[Knowledge Graph Path]\n"
        + "\n".join(relation_lines)
        + "\nEvidence:\n"
        + "\n".join(evidence_lines)
    )


def _diverse_chunks(ranked_chunks: Sequence[_KnowledgeChunk]) -> list[_KnowledgeChunk]:
    """Keep high vector relevance while preventing one large document from dominating."""
    primary: list[_KnowledgeChunk] = []
    overflow: list[_KnowledgeChunk] = []
    seen_documents: set[str] = set()
    for chunk in ranked_chunks:
        if chunk.document_id in seen_documents:
            overflow.append(chunk)
            continue
        seen_documents.add(chunk.document_id)
        primary.append(chunk)

    # 原因：Excel 等大型文件会产生很多相似 chunk，可能挤掉其他文档的候选片段。
    # 作用：优先保留每份文档的最佳命中，再按原向量排名补足上下文数量。
    return (primary + overflow)[:SEARCH_TOP_K]


def _load_records(storage_path: Path) -> list[_DocumentRecord]:
    if not storage_path.exists():
        return []

    records: dict[str, _DocumentRecord] = {}
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
                else _stable_id("document", source_document)
            )
            # 原因：旧版 JSONL 可能把多个文件合在一条记录，也没有可靠的 chunk 元数据。
            # 作用：加载时按来源拆分并确定性重建，使单文件更新/删除不会误伤其他文件。
            record = _build_record(document_id, source_document)
            source = _single_record_source(record)
            if source is not None:
                source_key = source.casefold()
                previous_id = source_ids.get(source_key)
                if previous_id is not None:
                    records.pop(previous_id, None)
                source_ids[source_key] = record.id
            records[record.id] = record
    return list(records.values())


def _append_record(storage_path: Path, record: _DocumentRecord) -> None:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    with storage_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_record_payload(record), ensure_ascii=False) + "\n")


def _rewrite_records(storage_path: Path, records: Sequence[_DocumentRecord]) -> None:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = storage_path.with_suffix(f"{storage_path.suffix}.tmp")
    content = "".join(
        json.dumps(_record_payload(record), ensure_ascii=False) + "\n"
        for record in records
    )
    temporary_path.write_text(content, encoding="utf-8")
    # 原因：更新或删除期间进程中断不能留下半截 JSONL，documents 是索引事实来源。
    # 作用：完整写入临时文件后原子替换，向量与图谱可随时从它恢复。
    temporary_path.replace(storage_path)


def _record_payload(record: _DocumentRecord) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "document_id": record.id,
        "timestamp": record.timestamp,
        "document": record.document,
        "chunks": [asdict(chunk) for chunk in record.chunks],
    }


def _record_sources(record: _DocumentRecord) -> set[str]:
    return {chunk.source for chunk in record.chunks}


def _single_record_source(record: _DocumentRecord) -> str | None:
    sources = _record_sources(record) - {"unknown"}
    return next(iter(sources)) if len(sources) == 1 else None


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


def _stable_id(prefix: str, value: str) -> str:
    digest = blake2b(value.encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}-{digest}"


def _index_directory(storage_path: Path) -> Path:
    return storage_path.parent / f"{storage_path.stem}_index"


def _graph_path(storage_path: Path) -> Path:
    if storage_path == DEFAULT_MINIRAG_STORE_PATH:
        return DEFAULT_KNOWLEDGE_GRAPH_PATH
    return storage_path.parent / f"{storage_path.stem}_graph.json"


def _to_graph_chunks(chunks: Sequence[_KnowledgeChunk]) -> tuple[GraphChunk, ...]:
    return tuple(
        GraphChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            source=chunk.source,
            page=chunk.page,
            content=chunk.content,
        )
        for chunk in chunks
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_coroutine(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run MiniRAG async storage from sync services and async Skills."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # 原因：RagSearchSkill 已在事件循环中，而公开 search() 必须保持同步接口。
    # 作用：在独立线程运行 vendor coroutine，避免嵌套 asyncio.run() 报错。
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    # 原因：知识库必须在离线环境稳定运行，且多个 facade 不应重复加载同一个模型。
    # 作用：缺少模型时立即报错；成功后复用本地模型，并读取其真实 embedding 维度。
    return SentenceTransformer(
        model_name,
        device="cpu",
        local_files_only=True,
    )
