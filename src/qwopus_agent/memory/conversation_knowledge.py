"""Conversation-scoped lifecycle for persistent MiniRAG stores."""

from __future__ import annotations

import re
import shutil
from _thread import LockType
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex
    from qwopus_agent.memory.minirag import MiniRAG

DEFAULT_CONVERSATION_KNOWLEDGE_ROOT = Path("storage/minirag/conversations")
_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_FILE_HEADER_PATTERN = re.compile(r"(?m)^(# File:\s*)(.+?)\s*$")


def conversation_knowledge_path(
    conversation_id: str,
    *,
    root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
) -> Path:
    """Return the fact-store path owned by one conversation."""
    normalized_id = conversation_id.strip()
    if not _CONVERSATION_ID_PATTERN.fullmatch(normalized_id):
        # 原因：conversation_id 最终会参与本地路径构造，不能接受路径分隔符或父目录跳转。
        # 作用：知识库路径始终被限制在配置的 conversations 根目录内。
        raise ValueError("conversation_id contains unsupported characters")
    return Path(root) / normalized_id / "documents.jsonl"


@dataclass
class _KnowledgeEntry:
    """One cached memory instance and its per-conversation operation lock."""

    memory: MiniRAG
    lock: LockType = field(default_factory=Lock)
    active: bool = True


@dataclass(frozen=True)
class _MirroredConversationMemory:
    """Keep private reads isolated while mirroring uploads into the global aggregate."""

    conversation_id: str
    private: MiniRAG
    global_entry: _KnowledgeEntry

    def insert(self, document: str, *, document_id: str | None = None) -> str:
        """Insert privately first, then update the explicitly authorized global view."""
        resolved_id = self.private.insert(document, document_id=document_id)
        scoped_document = _scope_document_source(document, self.conversation_id)
        with self.global_entry.lock:
            # 原因：不同会话可能同时上传同名文件，原始 source 会在全局索引中互相替换。
            # 作用：source 和 document_id 都加入会话命名空间，全局库可保留每个聊天的副本。
            self.global_entry.memory.insert(
                scoped_document,
                document_id=f"conversation-{self.conversation_id}-{resolved_id}",
            )
        return resolved_id

    def search(
        self,
        query: str,
        min_relevance: float = 0.25,
        *,
        document_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[str]:
        """Search only the current conversation during document analysis."""
        return self.private.search(
            query,
            min_relevance,
            document_ids=document_ids,
            section_ids=section_ids,
            sources=sources,
        )

    @property
    def graph_index(self) -> KnowledgeGraphIndex:
        """Keep analysis graph tools bound to the current conversation."""
        return self.private.graph_index


@dataclass
class ConversationKnowledgeManager:
    """Create and serialize MiniRAG instances without mixing conversation data."""

    root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
    global_storage_path: Path | None = None
    factory: Callable[[Path], MiniRAG] | None = field(default=None, repr=False)
    _entries: dict[str, _KnowledgeEntry] = field(default_factory=dict, init=False, repr=False)
    _global_entries: dict[str, _KnowledgeEntry] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _deleted: set[str] = field(default_factory=set, init=False, repr=False)
    _entries_lock: LockType = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.global_storage_path = (
            Path(self.global_storage_path)
            if self.global_storage_path is not None
            else self.root.parent / "documents.jsonl"
        )

    def storage_path(self, conversation_id: str) -> Path:
        """Expose the validated path rule to API and worker-process adapters."""
        return conversation_knowledge_path(conversation_id, root=self.root)

    def global_storage_path_for(self, account_id: str) -> Path:
        """Return one account's aggregate MiniRAG path."""
        normalized_id = account_id.strip()
        if not _CONVERSATION_ID_PATTERN.fullmatch(normalized_id):
            raise ValueError("account_id contains unsupported characters")
        return self.root.parent / "users" / normalized_id / "documents.jsonl"

    def get(self, conversation_id: str) -> MiniRAG:
        """Return the process-local MiniRAG instance for one conversation."""
        return self._entry_for(conversation_id).memory

    def list_sources(self, conversation_id: str) -> list[str]:
        """Return source names without creating an empty knowledge store."""
        storage_path = self.storage_path(conversation_id)
        with self._entries_lock:
            entry = self._entries.get(conversation_id)
        if entry is None:
            if not storage_path.is_file():
                return []
            entry = self._entry_for(conversation_id)
        with entry.lock:
            if not entry.active:
                return []
            # 原因：意图解析只需要可引用文件名，不能读取或注入文档正文。
            # 作用：为“这个文档/第二份文档”提供有界清单，并复用 MiniRAG 的持久化来源索引。
            return entry.memory.list_sources()

    def _entry_for(self, conversation_id: str) -> _KnowledgeEntry:
        """Resolve one entry atomically against concurrent deletion."""
        storage_path = self.storage_path(conversation_id)
        with self._entries_lock:
            if conversation_id in self._deleted:
                # 原因：上传请求可能在验证 conversation 后与删除请求并发进入知识层。
                # 作用：删除后的会话在当前进程中成为 tombstone，旧请求不能重建孤立索引。
                raise RuntimeError("conversation knowledge was deleted")
            entry = self._entries.get(conversation_id)
            if entry is None:
                factory = self.factory or _create_minirag
                # 原因：embedding、向量索引和图谱初始化成本较高，不能在同一会话的每次上传重复执行。
                # 作用：API 进程按 conversation_id 延迟创建并复用一个 MiniRAG 实例。
                entry = _KnowledgeEntry(memory=factory(storage_path))
                self._entries[conversation_id] = entry
            return entry

    @contextmanager
    def lease(
        self,
        conversation_id: str,
        *,
        global_scope: str | None = None,
    ) -> Iterator[MiniRAG]:
        """Serialize mutations for one conversation while other conversations remain parallel."""
        # 原因：分两次读取缓存时，删除可能夹在 get() 与字典访问之间并泄漏 KeyError。
        # 作用：一次取得稳定 entry；删除会等待其锁，迟到 lease 则收到受控 RuntimeError。
        entry = self._entry_for(conversation_id)
        global_entry = self._global_entry_for(global_scope)
        with entry.lock:
            if not entry.active:
                raise RuntimeError("conversation knowledge was deleted")
            # 原因：分析流程只能接收一个 MiniRAG 入口，但上传还需要进入可选的全局聚合库。
            # 作用：读取和 Tool 始终落在私库，只有 insert 同步写入全局库。
            yield cast(
                "MiniRAG",
                _MirroredConversationMemory(
                    conversation_id=conversation_id,
                    private=entry.memory,
                    global_entry=global_entry,
                ),
            )

    def delete(
        self,
        conversation_id: str,
        *,
        global_scope: str | None = None,
    ) -> None:
        """Remove cached and persisted knowledge after explicit conversation deletion."""
        # 原因：账号级 Global 库按 global_scope 分文件存储，删除 chat 时也必须清理同一 scope。
        # 作用：私库删除和账号全局镜像删除保持一致，避免被删除聊天仍可被 Global 搜到。
        global_entry = self._global_entry_for(global_scope)
        directory = self.storage_path(conversation_id).parent
        with self._entries_lock:
            self._deleted.add(conversation_id)
            entry = self._entries.pop(conversation_id, None)
            if entry is None:
                if global_entry is not None:
                    with global_entry.lock:
                        _delete_global_scope(global_entry.memory, conversation_id)
                shutil.rmtree(directory, ignore_errors=True)
                return
            entry.active = False
            with entry.lock:
                if global_entry is not None:
                    with global_entry.lock:
                        _delete_global_scope(global_entry.memory, conversation_id)
                # 原因：删除聊天后保留其向量和图谱会形成无法管理的孤立本地数据。
                # 作用：显式删除对话时一并删除仅属于该会话的派生知识库目录。
                shutil.rmtree(directory, ignore_errors=True)

    def claim_legacy_global(self, account_id: str) -> None:
        """Move the pre-account aggregate into the first administrator namespace."""
        destination = self.global_storage_path_for(account_id)
        source = self.global_storage_path
        if source is None or not source.exists() or destination.exists():
            return
        with self._entries_lock:
            if "legacy" in self._global_entries:
                raise RuntimeError("Legacy global knowledge is currently in use.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 原因：MiniRAG 的 JSONL、向量目录和知识图谱是一个一致的派生集合。
            # 作用：首次初始化时整体迁移到管理员命名空间，其他账号不会读取旧全局证据。
            for legacy_path in (
                source,
                Path(f"{source}.lock"),
                source.parent / f"{source.stem}_index",
                source.parent / "knowledge_graph.json",
            ):
                if legacy_path.exists():
                    shutil.move(
                        str(legacy_path),
                        str(destination.parent / legacy_path.name),
                    )

    def _global_entry_for(self, scope: str | None = None) -> _KnowledgeEntry:
        """Return the process-local aggregate used only by explicit Global tools."""
        key = scope or "legacy"
        with self._entries_lock:
            entry = self._global_entries.get(key)
            if entry is None:
                factory = self.factory or _create_minirag
                storage_path = (
                    self.global_storage_path_for(scope)
                    if scope is not None
                    else self.global_storage_path
                )
                if storage_path is None:
                    raise RuntimeError("global knowledge path was not initialized")
                entry = _KnowledgeEntry(
                    memory=factory(storage_path)
                )
                self._global_entries[key] = entry
            return entry


def _create_minirag(storage_path: Path) -> MiniRAG:
    """Import the expensive knowledge stack only when a conversation first needs it."""
    from qwopus_agent.memory.minirag import MiniRAG

    return MiniRAG(storage_path=storage_path)


def _scope_document_source(document: str, conversation_id: str) -> str:
    """Namespace every persisted source before it enters the global aggregate."""
    prefix = f"conversation:{conversation_id}/"
    if _FILE_HEADER_PATTERN.search(document):
        return _FILE_HEADER_PATTERN.sub(
            lambda match: f"{match.group(1)}{prefix}{match.group(2).strip()}",
            document,
        )
    return f"# File: {prefix}unknown\n\n{document}"


def _delete_global_scope(memory: MiniRAG, conversation_id: str) -> None:
    """Remove only one conversation's mirrored records from the global aggregate."""
    source_prefix = f"conversation:{conversation_id}/".casefold()
    for source in memory._list_sources():
        if source.casefold().startswith(source_prefix):
            memory._delete_source(source)
