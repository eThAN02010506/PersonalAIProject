"""SQLite persistence for conversations used by the primary web frontend."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from qwopus_agent.utils.conversation_log import list_conversations, load_chat_messages

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
                """
            )
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
