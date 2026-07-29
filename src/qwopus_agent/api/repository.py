"""SQLite persistence for conversations used by the primary web frontend."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from qwopus_agent.services.orchestration_models import ConversationTaskState
from qwopus_agent.utils.conversation_log import list_conversations, load_chat_messages
from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens

DEFAULT_DATABASE_PATH = Path("storage/qwopus.db")


@dataclass(frozen=True)
class ConversationRecord:
    """Database row for one conversation."""

    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MessageRecord:
    """Database row for one chat message."""

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class ConversationMemoryRecord:
    """Compressed model context while full messages remain untouched."""

    conversation_id: str
    summary: str
    summary_until_message_id: str | None
    pinned_facts: tuple[str, ...]
    open_tasks: tuple[str, ...]
    task_state: ConversationTaskState
    updated_at: str


class ConversationRepository:
    """Small repository that keeps SQL out of API routes and UI code."""

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE_PATH,
        *,
        import_legacy: bool = True,
    ) -> None:
        self.database_path = database_path
        self.import_legacy = import_legacy

    def initialize(self) -> None:
        """Create the local schema and import legacy JSONL once."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    conversation_id TEXT PRIMARY KEY
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_until_message_id TEXT,
                    pinned_facts TEXT NOT NULL DEFAULT '[]',
                    open_tasks TEXT NOT NULL DEFAULT '[]',
                    task_state TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                """
            )
            # 原因：已有用户数据库早于结构化任务状态，CREATE TABLE IF NOT EXISTS 不会补列。
            # 作用：只追加带默认值的兼容字段，不删除或重写任何历史会话和消息。
            _ensure_task_state_column(connection)
            count = connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        # 原因：正式前端首次启动要继承 Streamlit 历史，但隔离测试不能读取用户真实日志。
        # 作用：生产环境默认迁移一次，测试或全新部署可显式关闭旧日志导入。
        if count == 0 and self.import_legacy:
            self._import_legacy_log()

    def list_conversations(self) -> list[ConversationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [ConversationRecord(**dict(row)) for row in rows]

    def create_conversation(self, title: str = "New chat") -> ConversationRecord:
        now = _now()
        record = ConversationRecord(uuid4().hex, title.strip() or "New chat", now, now)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (record.id, record.title, record.created_at, record.updated_at),
            )
        return record

    def rename_conversation(self, conversation_id: str, title: str) -> ConversationRecord | None:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip(), now, conversation_id),
            )
        return self.get_conversation(conversation_id) if cursor.rowcount else None

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return bool(cursor.rowcount)

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return ConversationRecord(**dict(row)) if row else None

    def list_messages(self, conversation_id: str) -> list[MessageRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, conversation_id, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        return [MessageRecord(**dict(row)) for row in rows]

    def add_message(self, conversation_id: str, role: str, content: str) -> MessageRecord:
        now = _now()
        record = MessageRecord(uuid4().hex, conversation_id, role, content, now)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (record.id, conversation_id, role, content, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return record

    def get_memory(self, conversation_id: str) -> ConversationMemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_id, summary, summary_until_message_id, "
                "pinned_facts, open_tasks, task_state, updated_at FROM conversation_memory "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationMemoryRecord(
            conversation_id=row["conversation_id"],
            summary=row["summary"],
            summary_until_message_id=row["summary_until_message_id"],
            pinned_facts=tuple(json.loads(row["pinned_facts"])),
            open_tasks=tuple(json.loads(row["open_tasks"])),
            task_state=ConversationTaskState.model_validate_json(
                row["task_state"] or "{}"
            ),
            updated_at=row["updated_at"],
        )

    def set_memory_context(
        self,
        conversation_id: str,
        *,
        pinned_facts: tuple[str, ...] = (),
        open_tasks: tuple[str, ...] = (),
    ) -> None:
        current = self.get_memory(conversation_id)
        self._write_memory(
            ConversationMemoryRecord(
                conversation_id=conversation_id,
                summary=current.summary if current else "",
                summary_until_message_id=(
                    current.summary_until_message_id if current else None
                ),
                pinned_facts=pinned_facts,
                open_tasks=open_tasks,
                task_state=current.task_state if current else ConversationTaskState(),
                updated_at=_now(),
            )
        )

    def set_task_state(
        self,
        conversation_id: str,
        task_state: ConversationTaskState,
    ) -> None:
        """Persist one successful task without replacing compressed history."""
        current = self.get_memory(conversation_id)
        self._write_memory(
            ConversationMemoryRecord(
                conversation_id=conversation_id,
                summary=current.summary if current else "",
                summary_until_message_id=(
                    current.summary_until_message_id if current else None
                ),
                pinned_facts=current.pinned_facts if current else (),
                open_tasks=task_state.open_tasks,
                task_state=task_state,
                updated_at=_now(),
            )
        )

    def build_model_history(
        self,
        conversation_id: str,
        *,
        keep_recent: int = 6,
        max_summary_tokens: int = 1200,
    ) -> list[dict[str, str]]:
        """Compact older turns and return summary plus recent original messages."""
        messages = self.list_messages(conversation_id)
        memory = self.get_memory(conversation_id)
        summarized_index = _message_index(
            messages,
            memory.summary_until_message_id if memory else None,
        )
        unsummarized = messages[summarized_index + 1 :]
        candidates = unsummarized[:-keep_recent] if len(unsummarized) > keep_recent else []
        if candidates:
            parts = [memory.summary] if memory and memory.summary else []
            parts.extend(f"{message.role}: {message.content}" for message in candidates)
            # 原因：完整消息必须永久保留，但重复发送所有旧轮次会挤掉当前文档证据。
            # 作用：只压缩模型上下文并记录压缩边界，SQLite messages 表不做删除或改写。
            summary = _balanced_context_summary(parts, max_tokens=max_summary_tokens)
            memory = ConversationMemoryRecord(
                conversation_id=conversation_id,
                summary=summary,
                summary_until_message_id=candidates[-1].id,
                pinned_facts=memory.pinned_facts if memory else (),
                open_tasks=memory.open_tasks if memory else (),
                task_state=memory.task_state if memory else ConversationTaskState(),
                updated_at=_now(),
            )
            self._write_memory(memory)
            summarized_index = _message_index(messages, memory.summary_until_message_id)

        history: list[dict[str, str]] = []
        if memory and (memory.summary or memory.pinned_facts or memory.open_tasks):
            context = ["[Conversation summary]", memory.summary]
            if memory.pinned_facts:
                context.append("Pinned facts:\n- " + "\n- ".join(memory.pinned_facts))
            if memory.open_tasks:
                context.append("Open tasks:\n- " + "\n- ".join(memory.open_tasks))
            history.append({"role": "assistant", "content": "\n\n".join(filter(None, context))})
        history.extend(
            {"role": message.role, "content": message.content}
            for message in messages[summarized_index + 1 :]
        )
        return history

    def _write_memory(self, memory: ConversationMemoryRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_memory("
                "conversation_id, summary, summary_until_message_id, pinned_facts, "
                "open_tasks, task_state, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "summary=excluded.summary, "
                "summary_until_message_id=excluded.summary_until_message_id, "
                "pinned_facts=excluded.pinned_facts, open_tasks=excluded.open_tasks, "
                "task_state=excluded.task_state, "
                "updated_at=excluded.updated_at",
                (
                    memory.conversation_id,
                    memory.summary,
                    memory.summary_until_message_id,
                    json.dumps(memory.pinned_facts, ensure_ascii=False),
                    json.dumps(memory.open_tasks, ensure_ascii=False),
                    memory.task_state.model_dump_json(),
                    memory.updated_at,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _import_legacy_log(self) -> None:
        # 原因：Streamlit 调试台已经保存了用户历史，切换主前端不能让记录突然消失。
        # 作用：仅在空数据库首次启动时复制 JSONL 会话，原调试日志保持不变。
        for summary in list_conversations():
            now = summary.updated_at or _now()
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO conversations(id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (summary.conversation_id, summary.title, now, now),
                )
            for message in load_chat_messages(conversation_id=summary.conversation_id):
                self.add_message(summary.conversation_id, message["role"], message["content"])


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_task_state_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(conversation_memory)").fetchall()
    }
    if "task_state" not in columns:
        connection.execute(
            "ALTER TABLE conversation_memory "
            "ADD COLUMN task_state TEXT NOT NULL DEFAULT '{}'"
        )


def _message_index(messages: list[MessageRecord], message_id: str | None) -> int:
    if message_id is None:
        return -1
    return next(
        (index for index, message in enumerate(messages) if message.id == message_id),
        -1,
    )


def _balanced_context_summary(parts: list[str], *, max_tokens: int) -> str:
    non_empty = [part.strip() for part in parts if part.strip()]
    if not non_empty:
        return ""
    if sum(estimate_tokens(part) for part in non_empty) <= max_tokens:
        return "\n".join(non_empty)
    per_part = max(1, max_tokens // len(non_empty))
    return "\n".join(truncate_to_tokens(part, per_part) for part in non_empty)
