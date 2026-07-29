"""Qwopus knowledge adapter backed by MiniRAG's NanoVectorDB component."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
from collections.abc import Coroutine, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, TypeVar

import numpy as np
from minirag.kg.nano_vector_db_impl import NanoVectorDBStorage
from minirag.utils import EmbeddingFunc
from numpy.typing import NDArray

import qwopus_agent.memory.minirag_records as minirag_records
import qwopus_agent.memory.minirag_retrieval as minirag_retrieval
from qwopus_agent.memory.entity_resolver import EntityResolver
from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import GraphExtractor, RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphEvidence, GraphPath
from qwopus_agent.memory.knowledge_graph import (
    DEFAULT_KNOWLEDGE_GRAPH_PATH,
    KnowledgeGraphIndex,
)

INDEX_SCHEMA_VERSION = minirag_records.INDEX_SCHEMA_VERSION
SEARCH_TOP_K = minirag_retrieval.SEARCH_TOP_K
_DocumentRecord = minirag_records.DocumentRecord
_KnowledgeChunk = minirag_records.KnowledgeChunk
_append_record = minirag_records.append_record
_build_record = minirag_records.build_record
_load_records = minirag_records.load_records
_record_sources = minirag_records.record_sources
_rewrite_records = minirag_records.rewrite_records
_single_record_source = minirag_records.single_record_source
_stable_id = minirag_records.stable_id
_to_graph_chunks = minirag_records.to_graph_chunks
_diverse_chunks = minirag_retrieval.diverse_chunks
_embedding_content = minirag_retrieval.embedding_content
_render_graph_search_result = minirag_retrieval.render_graph_search_result
_render_search_result = minirag_retrieval.render_search_result
_source_matches_query = minirag_retrieval.source_matches_query

DEFAULT_MINIRAG_STORE_PATH = Path("storage/minirag/documents.jsonl")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEARCH_CANDIDATE_K = 96
COSINE_THRESHOLD = 0.25
GRAPH_SEARCH_LIMIT = 3

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingBackend(Protocol):
    """Internal contract that keeps document retrieval independent from the chat model."""

    model_name: str

    @property
    def dimensions(self) -> int:
        """Return the stable vector width produced by this backend."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
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

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
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


@dataclass
class MiniRAG:
    """Implement KnowledgeStore with persistent vectors and Qwopus graph retrieval.

    This adapter uses MiniRAG's NanoVectorDB storage component. It is intentionally
    not the upstream project's full ``MiniRAG.query`` pipeline; Qwopus owns chunking,
    source scoping, graph extraction, and result rendering around that component.
    """

    storage_path: Path = DEFAULT_MINIRAG_STORE_PATH
    embedding_backend: EmbeddingBackend | None = field(default=None, repr=False)
    graph_extractor: GraphExtractor | None = field(default=None, repr=False)
    graph_storage_path: Path | None = field(default=None, repr=False)
    _records: list[_DocumentRecord] = field(default_factory=list, init=False, repr=False)
    _chunks: dict[str, _KnowledgeChunk] = field(default_factory=dict, init=False, repr=False)
    _vector_store: NanoVectorDBStorage = field(init=False, repr=False)
    _graph: PersistentKnowledgeGraph = field(init=False, repr=False)
    _graph_index: KnowledgeGraphIndex = field(init=False, repr=False)
    _thread_lock: Any = field(default_factory=RLock, init=False, repr=False)
    _storage_signature: tuple[int, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Load documents and the persisted MiniRAG vector index on startup."""
        self.storage_path = Path(self.storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        # 原因：聊天使用 spawn 子进程，可能与文档上传同时打开同一个会话索引。
        # 作用：初始化与索引恢复持有跨进程排他锁，避免读取一半写入的派生文件。
        with _exclusive_storage_lock(self.storage_path):
            self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize one storage snapshot while the caller owns its file lock."""
        embedding_backend: EmbeddingBackend
        if self.embedding_backend is None:
            embedding_backend = SentenceTransformerEmbedding(
                model_name=os.getenv("QWOPUS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            )
        else:
            embedding_backend = self.embedding_backend
        self.embedding_backend = embedding_backend
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
        self._storage_signature = _storage_signature(self.storage_path)

    def insert(self, document: str, *, document_id: str | None = None) -> str:
        """Insert one Markdown-normalized document into persistent semantic memory."""
        with self._thread_lock, _exclusive_storage_lock(self.storage_path):
            self._refresh_from_disk_if_changed()
            resolved_id = self._insert_unlocked(document, document_id=document_id)
            self._storage_signature = _storage_signature(self.storage_path)
            return resolved_id

    def _insert_unlocked(self, document: str, *, document_id: str | None = None) -> str:
        """Perform insertion while the public boundary owns both storage locks."""
        if not document.strip():
            raise ValueError("document must not be empty")

        resolved_document_id = document_id or _stable_id("document", document)
        if any(record.id == resolved_document_id for record in self._records):
            return resolved_document_id

        record = _build_record(resolved_document_id, document)
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
            logger.exception(
                "knowledge_graph_ingestion_failed document_id=%s",
                resolved_document_id,
            )
        return resolved_document_id

    def search(
        self,
        query: str,
        min_relevance: float = COSINE_THRESHOLD,
        *,
        document_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[str]:
        """Return graph paths first, then complementary semantic chunks."""
        with self._thread_lock, _exclusive_storage_lock(self.storage_path):
            self._refresh_from_disk_if_changed()
            return self._search_unlocked(
                query,
                min_relevance,
                document_ids=document_ids,
                section_ids=section_ids,
                sources=sources,
            )

    def _search_unlocked(
        self,
        query: str,
        min_relevance: float = COSINE_THRESHOLD,
        *,
        document_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[str]:
        """Search one internally consistent in-memory index snapshot."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")

        document_filter = set(document_ids or ())
        section_filter = set(section_ids or ())
        source_filter = {source.casefold() for source in (sources or ())}
        graph_paths = self._graph_index.search(query, limit=GRAPH_SEARCH_LIMIT)
        graph_paths = _filter_graph_paths(
            graph_paths,
            document_ids=document_filter,
            section_ids=section_filter,
            sources=source_filter,
        )
        results = [_render_graph_search_result(path) for path in graph_paths]
        graph_chunk_ids = {
            evidence.chunk_id
            for path in graph_paths
            for evidence in path.evidence
        }
        if not self._chunks or len(results) >= SEARCH_TOP_K:
            return results[:SEARCH_TOP_K]

        source_matched_chunks = [
            chunk
            for chunk in self._chunks.values()
            if (
                _source_matches_query(query, chunk.source)
                and chunk.id not in graph_chunk_ids
                and (not document_filter or chunk.document_id in document_filter)
                and (not section_filter or chunk.section_id in section_filter)
                and (not source_filter or chunk.source.casefold() in source_filter)
            )
        ]
        source_matched_ids = {chunk.id for chunk in source_matched_chunks}
        matches = _run_coroutine(
            self._vector_store.query(query, top_k=SEARCH_CANDIDATE_K)
        )
        ranked_chunks: list[_KnowledgeChunk] = []
        for match in matches:
            # 原因：向量库的固定底线只负责粗筛，用户需要为每次问答调整 Source 严格度。
            # 作用：按当前请求的余弦相似度阈值过滤，不修改共享索引或影响其他用户。
            if float(match.get("distance", 0.0)) < min_relevance:
                continue
            chunk = self._chunks.get(str(match.get("id", "")))
            if (
                chunk is not None
                and chunk.id not in graph_chunk_ids
                and chunk.id not in source_matched_ids
                and (not document_filter or chunk.document_id in document_filter)
                and (not section_filter or chunk.section_id in section_filter)
                and (not source_filter or chunk.source.casefold() in source_filter)
            ):
                ranked_chunks.append(chunk)
        vector_results = [
            _render_search_result(chunk)
            for chunk in _diverse_chunks(source_matched_chunks + ranked_chunks)
        ]
        return (results + vector_results)[:SEARCH_TOP_K]

    def list_sources(self) -> list[str]:
        """Return indexed file sources without exposing document contents."""
        with self._thread_lock, _exclusive_storage_lock(self.storage_path):
            self._refresh_from_disk_if_changed()
            return sorted(
                {
                    source
                    for record in self._records
                    for source in _record_sources(record)
                    if source != "unknown"
                },
                key=str.casefold,
            )

    def _list_sources(self) -> list[str]:
        """Compatibility alias for the existing maintenance service."""
        return self.list_sources()

    def _delete_source(self, source: str) -> int:
        """Delete all records for one exact file source from every index."""
        with self._thread_lock, _exclusive_storage_lock(self.storage_path):
            self._refresh_from_disk_if_changed()
            deleted = self._delete_source_unlocked(source)
            self._storage_signature = _storage_signature(self.storage_path)
            return deleted

    def _delete_source_unlocked(self, source: str) -> int:
        """Delete a source while the maintenance boundary owns storage locks."""
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
        with self._thread_lock, _exclusive_storage_lock(self.storage_path):
            self._refresh_from_disk_if_changed()
            self._rebuild_indexes_unlocked()

    def _rebuild_indexes_unlocked(self) -> None:
        """Rebuild derived indexes while the maintenance boundary owns storage locks."""
        vector_path = _index_directory(self.storage_path) / "vdb_qwopus_chunks.json"
        vector_path.unlink(missing_ok=True)
        self._vector_store = self._create_vector_store()
        if self._chunks:
            self._upsert_chunks(tuple(self._chunks.values()))

        graph_storage_path = self.graph_storage_path
        if graph_storage_path is None:
            raise RuntimeError("graph storage was not initialized")
        Path(graph_storage_path).unlink(missing_ok=True)
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

    @property
    def graph_index(self) -> KnowledgeGraphIndex:
        """Expose this MiniRAG instance's matching graph index to its Tool adapter."""
        # 原因：重新按全局常量创建图谱会绕过 conversation_id 的知识库边界。
        # 作用：rag_search 与 graph_search 始终查询同一个会话目录中的两种索引。
        return self._graph_index

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

    def _refresh_from_disk_if_changed(self) -> None:
        """Reload derived state when another process changed the fact store."""
        signature = _storage_signature(self.storage_path)
        if signature == self._storage_signature:
            return
        # 原因：文件锁只能阻止同时写盘，不能自动刷新其他进程已经缓存的记录和向量对象。
        # 作用：在操作锁内按需重载最新事实库，使 API worker 与 Agent 子进程立即看到彼此写入。
        self._initialize_storage()

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
        embedding_backend = self.embedding_backend
        if embedding_backend is None:
            raise RuntimeError("embedding backend was not initialized")
        index_dir = _index_directory(self.storage_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        expected_config = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": embedding_backend.model_name,
            "embedding_dimensions": embedding_backend.dimensions,
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

        async def embed(texts: list[str]) -> NDArray[np.float32]:
            return embedding_backend.encode(texts)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_backend.dimensions,
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
            meta_fields={
                "document_id",
                "source",
                "page",
                "page_end",
                "section_id",
                "section_path",
                "position",
            },
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
                "content": _embedding_content(chunk),
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page or "",
                "page_end": chunk.page_end or "",
                "section_id": chunk.section_id,
                "section_path": " / ".join(chunk.section_path),
                "position": chunk.position,
            }
            for chunk in chunks
        }
        _run_coroutine(self._vector_store.upsert(payload))
        _run_coroutine(self._vector_store.index_done_callback())


def _index_directory(storage_path: Path) -> Path:
    return storage_path.parent / f"{storage_path.stem}_index"


def _storage_signature(storage_path: Path) -> tuple[int, int] | None:
    """Return a cheap process-independent version for the JSONL fact store."""
    try:
        stat = storage_path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _graph_path(storage_path: Path) -> Path:
    if storage_path == DEFAULT_MINIRAG_STORE_PATH:
        return DEFAULT_KNOWLEDGE_GRAPH_PATH
    return storage_path.parent / f"{storage_path.stem}_graph.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _filter_graph_paths(
    paths: Sequence[GraphPath],
    *,
    document_ids: set[str],
    section_ids: set[str],
    sources: set[str],
) -> list[GraphPath]:
    """Return only paths whose every relation remains supported inside the scope."""
    if not document_ids and not section_ids and not sources:
        return list(paths)

    scoped_paths: list[GraphPath] = []
    for path in paths:
        scoped_relations = []
        for relation in path.relations:
            scoped_evidence = tuple(
                evidence
                for evidence in relation.evidence
                if _graph_evidence_in_scope(
                    evidence,
                    document_ids=document_ids,
                    section_ids=section_ids,
                    sources=sources,
                )
            )
            if not scoped_evidence:
                # 原因：多跳路径中的任一关系若只由未选章节支持，整条推理链就越界。
                # 作用：逐关系执行 fail-closed 过滤，避免保留一条证据不完整的路径。
                break
            scoped_relations.append(
                relation.model_copy(update={"evidence": scoped_evidence})
            )
        else:
            scoped_paths.append(
                path.model_copy(
                    update={
                        "relations": tuple(scoped_relations),
                        "evidence": _unique_graph_evidence(
                            tuple(
                                evidence
                                for relation in scoped_relations
                                for evidence in relation.evidence
                            )
                        ),
                    }
                )
            )
    return scoped_paths


def _graph_evidence_in_scope(
    evidence: GraphEvidence,
    *,
    document_ids: set[str],
    section_ids: set[str],
    sources: set[str],
) -> bool:
    return (
        (not document_ids or evidence.document_id in document_ids)
        and (not section_ids or evidence.section_id in section_ids)
        and (not sources or evidence.source.casefold() in sources)
    )


def _unique_graph_evidence(
    evidence_items: Sequence[GraphEvidence],
) -> tuple[GraphEvidence, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[GraphEvidence] = []
    for evidence in evidence_items:
        identity = (evidence.chunk_id, evidence.text)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(evidence)
    return tuple(unique)


@contextmanager
def _exclusive_storage_lock(storage_path: Path) -> Iterator[None]:
    """Prevent two local processes from mutating one MiniRAG store concurrently."""
    lock_path = storage_path.with_suffix(f"{storage_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
