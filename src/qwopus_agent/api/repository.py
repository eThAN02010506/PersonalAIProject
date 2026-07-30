"""SQLite persistence for conversations used by the primary web frontend."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
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
    owner_user_id: str | None = None
    owner_username: str | None = None
    is_owner: bool = False
    shared_count: int = 0


@dataclass(frozen=True)
class UserRecord:
    """Safe account fields; password hashes never leave the repository boundary."""

    id: str
    username: str
    display_name: str
    role: Literal["admin", "member"]
    active: bool
    created_at: str


@dataclass(frozen=True)
class ConversationMemberRecord:
    """One account with explicit access to a conversation."""

    user_id: str
    username: str
    display_name: str
    access: Literal["owner", "member"]


@dataclass(frozen=True)
class MessageRecord:
    """Database row for one chat message."""

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class ConversationRunRecord:
    """Persistent, sanitized provenance for one Agent run."""

    run_id: str
    conversation_id: str
    user_message_id: str | None
    assistant_message_id: str | None
    requested_by_user_id: str | None
    objective: str
    operational_objective: str
    status: Literal["completed", "failed", "cancelled"]
    model_id: str
    reusable_skills: tuple[str, ...]
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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'member')),
                    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner_user_id TEXT REFERENCES users(id) ON DELETE RESTRICT
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
                CREATE TABLE IF NOT EXISTS conversation_runs (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    user_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    requested_by_user_id TEXT,
                    objective TEXT NOT NULL,
                    operational_objective TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('completed', 'failed', 'cancelled')),
                    model_id TEXT NOT NULL,
                    reusable_skills TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_runs_conversation
                    ON conversation_runs(conversation_id, created_at);
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
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS conversation_members (
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(conversation_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_members_user
                    ON conversation_members(user_id, conversation_id);
                CREATE TABLE IF NOT EXISTS document_owners (
                    document_id TEXT NOT NULL,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(document_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_document_owners_user
                    ON document_owners(user_id, document_id);
                CREATE TABLE IF NOT EXISTS conversation_documents (
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL,
                    attached_by_user_id TEXT NOT NULL
                        REFERENCES users(id) ON DELETE RESTRICT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(conversation_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS report_access (
                    filename TEXT PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    created_by_user_id TEXT NOT NULL
                        REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL
                );
                """
            )
            # 原因：已有用户数据库早于结构化任务状态，CREATE TABLE IF NOT EXISTS 不会补列。
            # 作用：只追加带默认值的兼容字段，不删除或重写任何历史会话和消息。
            _ensure_task_state_column(connection)
            # 原因：旧版 conversations 表没有账号所有者，重建表会冒险改写完整聊天历史。
            # 作用：使用 SQLite 支持的增量列迁移；首次管理员创建时再原子认领空所有者数据。
            _ensure_conversation_owner_column(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_owner "
                "ON conversations(owner_user_id, updated_at)"
            )
            connection.execute(
                "DELETE FROM sessions WHERE expires_at <= ?",
                (_now(),),
            )
            count = connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        # 原因：正式前端首次启动要继承 Streamlit 历史，但隔离测试不能读取用户真实日志。
        # 作用：生产环境默认迁移一次，测试或全新部署可显式关闭旧日志导入。
        if count == 0 and self.import_legacy:
            self._import_legacy_log()

    def list_conversations(self) -> list[ConversationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                _conversation_select()
                + " "
                "FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [_conversation_record(row) for row in rows]

    def list_conversations_with_reusable_runs(self) -> list[ConversationRecord]:
        """List conversations that contain at least one promotable run source."""
        with self._connect() as connection:
            rows = connection.execute(
                _conversation_select()
                + " FROM conversations WHERE EXISTS ("
                "SELECT 1 FROM conversation_runs run "
                "WHERE run.conversation_id = conversations.id "
                "AND run.status = 'completed' AND run.reusable_skills <> '[]'"
                ") ORDER BY conversations.updated_at DESC"
            ).fetchall()
        return [_conversation_record(row) for row in rows]

    def list_conversations_for_user(self, user_id: str) -> list[ConversationRecord]:
        """List only conversations owned by or explicitly shared with one account."""
        with self._connect() as connection:
            rows = connection.execute(
                _conversation_select(viewer=True)
                + " FROM conversations c "
                "LEFT JOIN users owner ON owner.id = c.owner_user_id "
                "WHERE c.owner_user_id = ? OR EXISTS ("
                "SELECT 1 FROM conversation_members member "
                "WHERE member.conversation_id = c.id AND member.user_id = ?"
                ") ORDER BY c.updated_at DESC",
                (user_id, user_id, user_id),
            ).fetchall()
        return [_conversation_record(row) for row in rows]

    def create_conversation(
        self,
        title: str = "New chat",
        *,
        owner_user_id: str | None = None,
    ) -> ConversationRecord:
        now = _now()
        conversation_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations("
                "id, title, created_at, updated_at, owner_user_id"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    title.strip() or "New chat",
                    now,
                    now,
                    owner_user_id,
                ),
            )
        record = self.get_conversation(conversation_id)
        if record is None:
            raise RuntimeError("Created conversation could not be read.")
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
                _conversation_select()
                + " FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return _conversation_record(row) if row else None

    def get_conversation_for_user(
        self,
        conversation_id: str,
        user_id: str,
    ) -> ConversationRecord | None:
        """Return a conversation only when the account has a relationship to it."""
        with self._connect() as connection:
            row = connection.execute(
                _conversation_select(viewer=True)
                + " FROM conversations c "
                "LEFT JOIN users owner ON owner.id = c.owner_user_id "
                "WHERE c.id = ? AND (c.owner_user_id = ? OR EXISTS ("
                "SELECT 1 FROM conversation_members member "
                "WHERE member.conversation_id = c.id AND member.user_id = ?"
                "))",
                (user_id, conversation_id, user_id, user_id),
            ).fetchone()
        return _conversation_record(row) if row else None

    def is_conversation_owner(self, conversation_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND owner_user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        return row is not None

    def list_conversation_members(
        self,
        conversation_id: str,
    ) -> list[ConversationMemberRecord]:
        """Return the owner followed by explicitly shared active members."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT owner.id AS user_id, owner.username, owner.display_name,
                       'owner' AS access, 0 AS ordering
                FROM conversations c
                JOIN users owner ON owner.id = c.owner_user_id
                WHERE c.id = ?
                UNION ALL
                SELECT member_user.id AS user_id, member_user.username,
                       member_user.display_name, 'member' AS access, 1 AS ordering
                FROM conversation_members member
                JOIN users member_user ON member_user.id = member.user_id
                WHERE member.conversation_id = ? AND member_user.active = 1
                ORDER BY ordering, username COLLATE NOCASE
                """,
                (conversation_id, conversation_id),
            ).fetchall()
        return [
            ConversationMemberRecord(
                user_id=str(row["user_id"]),
                username=str(row["username"]),
                display_name=str(row["display_name"]),
                access=row["access"],
            )
            for row in rows
        ]

    def add_conversation_member(
        self,
        conversation_id: str,
        username: str,
    ) -> ConversationMemberRecord | None:
        """Share a conversation with one existing active account."""
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name FROM users "
                "WHERE username = ? COLLATE NOCASE AND active = 1",
                (username,),
            ).fetchone()
            if row is None:
                return None
            owner = connection.execute(
                "SELECT owner_user_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if owner is None or owner["owner_user_id"] == row["id"]:
                return None
            connection.execute(
                "INSERT OR IGNORE INTO conversation_members("
                "conversation_id, user_id, created_at"
                ") VALUES (?, ?, ?)",
                (conversation_id, row["id"], now),
            )
        return ConversationMemberRecord(
            user_id=str(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            access="member",
        )

    def remove_conversation_member(
        self,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_members "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        return bool(cursor.rowcount)

    def is_conversation_member(self, conversation_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversation_members "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        return row is not None

    def has_users(self) -> bool:
        with self._connect() as connection:
            return bool(connection.execute("SELECT 1 FROM users LIMIT 1").fetchone())

    def create_initial_admin(
        self,
        *,
        username: str,
        display_name: str,
        password_hash: str,
    ) -> UserRecord | None:
        """Create exactly one first administrator and claim legacy conversations."""
        now = _now()
        user_id = uuid4().hex
        with self._connect() as connection:
            # 原因：两个首次打开的浏览器可能同时提交初始化表单。
            # 作用：IMMEDIATE 事务让“没有用户”检查和管理员插入成为一个不可分割操作。
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                connection.rollback()
                return None
            connection.execute(
                "INSERT INTO users("
                "id, username, display_name, password_hash, role, active, created_at"
                ") VALUES (?, ?, ?, ?, 'admin', 1, ?)",
                (user_id, username, display_name, password_hash, now),
            )
            connection.execute(
                "UPDATE conversations SET owner_user_id = ? WHERE owner_user_id IS NULL",
                (user_id,),
            )
            connection.commit()
        return self.get_user(user_id)

    def create_user(
        self,
        *,
        username: str,
        display_name: str,
        password_hash: str,
        role: Literal["admin", "member"] = "member",
    ) -> UserRecord:
        now = _now()
        user_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users("
                "id, username, display_name, password_hash, role, active, created_at"
                ") VALUES (?, ?, ?, ?, ?, 1, ?)",
                (user_id, username, display_name, password_hash, role, now),
            )
        user = self.get_user(user_id)
        if user is None:
            raise RuntimeError("Created user could not be read.")
        return user

    def list_users(self) -> list[UserRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, username, display_name, role, active, created_at "
                "FROM users ORDER BY created_at, username COLLATE NOCASE"
            ).fetchall()
        return [_user_record(row) for row in rows]

    def get_user(self, user_id: str) -> UserRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, active, created_at "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return _user_record(row) if row else None

    def get_user_with_password(self, username: str) -> tuple[UserRecord, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, active, created_at, password_hash "
                "FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return _user_record(row), str(row["password_hash"])

    def set_user_active(self, user_id: str, active: bool) -> UserRecord | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET active = ? WHERE id = ?",
                (int(active), user_id),
            )
            if cursor.rowcount and not active:
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return self.get_user(user_id) if cursor.rowcount else None

    def set_user_password(self, user_id: str, password_hash: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ? AND active = 1",
                (password_hash, user_id),
            )
            if cursor.rowcount:
                # 原因：密码变更后旧浏览器会话继续有效会削弱账号恢复能力。
                # 作用：撤销所有旧令牌；调用方随后为当前浏览器签发一个新会话。
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return bool(cursor.rowcount)

    def active_admin_count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1"
                ).fetchone()[0]
            )

    def create_session(
        self,
        *,
        token_hash: str,
        user_id: str,
        expires_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (token_hash, user_id, _now(), expires_at),
            )

    def user_for_session(self, token_hash: str) -> UserRecord | None:
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user.id, user.username, user.display_name, user.role, "
                "user.active, user.created_at "
                "FROM sessions session "
                "JOIN users user ON user.id = session.user_id "
                "WHERE session.token_hash = ? AND session.expires_at > ? "
                "AND user.active = 1",
                (token_hash, now),
            ).fetchone()
        return _user_record(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (token_hash,),
            )

    def register_document(
        self,
        document_id: str,
        *,
        owner_user_id: str,
        conversation_id: str | None = None,
    ) -> None:
        """Record who uploaded a saved document and which chat can use it."""
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO document_owners("
                "document_id, user_id, created_at"
                ") VALUES (?, ?, ?)",
                (document_id, owner_user_id, now),
            )
            if conversation_id is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO conversation_documents("
                    "conversation_id, document_id, attached_by_user_id, created_at"
                    ") VALUES (?, ?, ?, ?)",
                    (conversation_id, document_id, owner_user_id, now),
                )

    def link_document_to_conversation(
        self,
        document_id: str,
        *,
        conversation_id: str,
        attached_by_user_id: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversation_documents("
                "conversation_id, document_id, attached_by_user_id, created_at"
                ") VALUES (?, ?, ?, ?)",
                (conversation_id, document_id, attached_by_user_id, _now()),
            )

    def accessible_document_ids(self, user_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id FROM document_owners WHERE user_id = ?
                UNION
                SELECT attached.document_id
                FROM conversation_documents attached
                JOIN conversations conversation
                    ON conversation.id = attached.conversation_id
                WHERE conversation.owner_user_id = ?
                   OR EXISTS (
                       SELECT 1 FROM conversation_members member
                       WHERE member.conversation_id = conversation.id
                         AND member.user_id = ?
                   )
                """,
                (user_id, user_id, user_id),
            ).fetchall()
        return {str(row["document_id"]) for row in rows}

    def can_access_document(self, document_id: str, user_id: str) -> bool:
        return document_id in self.accessible_document_ids(user_id)

    def register_report(
        self,
        filename: str,
        *,
        created_by_user_id: str,
        conversation_id: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO report_access("
                "filename, conversation_id, created_by_user_id, created_at"
                ") VALUES (?, ?, ?, ?)",
                (filename, conversation_id, created_by_user_id, _now()),
            )

    def can_access_report(self, filename: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM report_access report
                LEFT JOIN conversations conversation
                    ON conversation.id = report.conversation_id
                WHERE report.filename = ?
                  AND (
                      report.created_by_user_id = ?
                      OR conversation.owner_user_id = ?
                      OR EXISTS (
                          SELECT 1 FROM conversation_members member
                          WHERE member.conversation_id = report.conversation_id
                            AND member.user_id = ?
                      )
                  )
                """,
                (filename, user_id, user_id, user_id),
            ).fetchone()
        return row is not None

    def claim_legacy_files(
        self,
        user_id: str,
        *,
        document_ids: list[str],
        report_filenames: list[str],
    ) -> None:
        """Assign pre-account local artifacts to the first administrator."""
        now = _now()
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO document_owners("
                "document_id, user_id, created_at"
                ") VALUES (?, ?, ?)",
                [(document_id, user_id, now) for document_id in document_ids],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO report_access("
                "filename, conversation_id, created_by_user_id, created_at"
                ") VALUES (?, NULL, ?, ?)",
                [(filename, user_id, now) for filename in report_filenames],
            )

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

    def save_conversation_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_message_id: str | None,
        assistant_message_id: str | None,
        requested_by_user_id: str | None,
        objective: str,
        operational_objective: str,
        status: Literal["completed", "failed", "cancelled"],
        model_id: str,
        reusable_skills: tuple[str, ...] = (),
    ) -> ConversationRunRecord:
        """Persist only reusable Skill names and message references for one run."""
        now = _now()
        skills_json = json.dumps(list(reusable_skills), ensure_ascii=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_runs("
                "run_id, conversation_id, user_message_id, assistant_message_id, "
                "requested_by_user_id, objective, operational_objective, status, "
                "model_id, reusable_skills, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    conversation_id,
                    user_message_id,
                    assistant_message_id,
                    requested_by_user_id,
                    objective,
                    operational_objective,
                    status,
                    model_id,
                    skills_json,
                    now,
                ),
            )
        return ConversationRunRecord(
            run_id=run_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            requested_by_user_id=requested_by_user_id,
            objective=objective,
            operational_objective=operational_objective,
            status=status,
            model_id=model_id,
            reusable_skills=reusable_skills,
            created_at=now,
        )

    def list_conversation_runs(
        self,
        conversation_id: str,
    ) -> list[ConversationRunRecord]:
        """List durable run provenance without loading raw Agent diagnostics."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, conversation_id, user_message_id, "
                "assistant_message_id, requested_by_user_id, objective, "
                "operational_objective, status, model_id, reusable_skills, "
                "created_at FROM conversation_runs WHERE conversation_id = ? "
                "ORDER BY created_at DESC, rowid DESC",
                (conversation_id,),
            ).fetchall()
        return [
            ConversationRunRecord(
                run_id=str(row["run_id"]),
                conversation_id=str(row["conversation_id"]),
                user_message_id=(
                    str(row["user_message_id"]) if row["user_message_id"] else None
                ),
                assistant_message_id=(
                    str(row["assistant_message_id"])
                    if row["assistant_message_id"]
                    else None
                ),
                requested_by_user_id=(
                    str(row["requested_by_user_id"])
                    if row["requested_by_user_id"]
                    else None
                ),
                objective=str(row["objective"]),
                operational_objective=str(row["operational_objective"]),
                status=row["status"],
                model_id=str(row["model_id"]),
                reusable_skills=tuple(json.loads(row["reusable_skills"])),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

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


def _ensure_conversation_owner_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
    }
    if "owner_user_id" not in columns:
        connection.execute(
            "ALTER TABLE conversations ADD COLUMN owner_user_id TEXT "
            "REFERENCES users(id) ON DELETE RESTRICT"
        )


def _conversation_select(*, viewer: bool = False) -> str:
    """Return one consistent projection for raw and access-scoped conversation reads."""
    if viewer:
        return (
            "SELECT c.id, c.title, c.created_at, c.updated_at, c.owner_user_id, "
            "owner.username AS owner_username, "
            "CASE WHEN c.owner_user_id = ? THEN 1 ELSE 0 END AS is_owner, "
            "(SELECT COUNT(*) FROM conversation_members sharing "
            "WHERE sharing.conversation_id = c.id) AS shared_count"
        )
    return (
        "SELECT conversations.id, conversations.title, conversations.created_at, "
        "conversations.updated_at, conversations.owner_user_id, "
        "(SELECT username FROM users "
        "WHERE users.id = conversations.owner_user_id) AS owner_username, "
        "0 AS is_owner, "
        "(SELECT COUNT(*) FROM conversation_members sharing "
        "WHERE sharing.conversation_id = conversations.id) AS shared_count"
    )


def _conversation_record(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        id=str(row["id"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        owner_user_id=(
            str(row["owner_user_id"])
            if row["owner_user_id"] is not None
            else None
        ),
        owner_username=(
            str(row["owner_username"])
            if row["owner_username"] is not None
            else None
        ),
        is_owner=bool(row["is_owner"]),
        shared_count=int(row["shared_count"]),
    )


def _user_record(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=str(row["id"]),
        username=str(row["username"]),
        display_name=str(row["display_name"]),
        role=row["role"],
        active=bool(row["active"]),
        created_at=str(row["created_at"]),
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
