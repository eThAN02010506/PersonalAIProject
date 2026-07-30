import sqlite3
import tempfile
import unittest
from pathlib import Path

from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.services.orchestration_models import (
    AnswerContract,
    ConversationTaskState,
)


class ConversationRepositoryTests(unittest.TestCase):
    def test_model_history_compacts_old_turns_without_deleting_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = ConversationRepository(
                Path(tmpdir) / "qwopus.db",
                import_legacy=False,
            )
            repository.initialize()
            conversation = repository.create_conversation()
            for index in range(12):
                role = "user" if index % 2 == 0 else "assistant"
                repository.add_message(
                    conversation.id,
                    role,
                    f"message {index} project detail",
                )

            history = repository.build_model_history(conversation.id, keep_recent=6)
            memory = repository.get_memory(conversation.id)

            # 原因：压缩只服务模型上下文，不能改变用户看到的完整聊天记录。
            # 作用：同时锁定 SQLite 原消息数量、摘要边界和最近原始消息。
            self.assertEqual(len(repository.list_messages(conversation.id)), 12)
            self.assertIsNotNone(memory)
            self.assertEqual(len(history), 7)
            self.assertIn("Conversation summary", history[0]["content"])
            self.assertEqual(history[-1]["content"], "message 11 project detail")

    def test_pinned_facts_and_open_tasks_are_included_in_model_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = ConversationRepository(
                Path(tmpdir) / "qwopus.db",
                import_legacy=False,
            )
            repository.initialize()
            conversation = repository.create_conversation()
            repository.set_memory_context(
                conversation.id,
                pinned_facts=("The project name is Qwopus-Agent.",),
                open_tasks=("Finish the document pipeline.",),
            )

            history = repository.build_model_history(conversation.id)

        self.assertIn("Qwopus-Agent", history[0]["content"])
        self.assertIn("document pipeline", history[0]["content"])

    def test_task_state_round_trips_without_replacing_memory_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = ConversationRepository(
                Path(tmpdir) / "qwopus.db",
                import_legacy=False,
            )
            repository.initialize()
            conversation = repository.create_conversation()
            repository.set_memory_context(
                conversation.id,
                pinned_facts=("Keep this fact.",),
                open_tasks=("Old task",),
            )
            repository.set_task_state(
                conversation.id,
                ConversationTaskState(
                    last_successful_objective="Compare the selected documents.",
                    last_task_type="compare",
                    last_answer_contract=AnswerContract(
                        task_type="compare",
                        required_facets=("differences", "conclusion"),
                    ),
                    active_document_sources=("alpha.pdf", "beta.pdf"),
                    open_tasks=("Review the conclusion.",),
                    updated_at="2026-07-29T00:00:00+00:00",
                ),
            )

            memory = repository.get_memory(conversation.id)

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertEqual(memory.pinned_facts, ("Keep this fact.",))
        self.assertEqual(memory.open_tasks, ("Review the conclusion.",))
        self.assertEqual(
            memory.task_state.active_document_sources,
            ("alpha.pdf", "beta.pdf"),
        )
        self.assertEqual(memory.task_state.last_task_type, "compare")

    def test_attached_document_ids_are_scoped_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repository = ConversationRepository(
                Path(tmpdir) / "qwopus.db",
                import_legacy=False,
            )
            repository.initialize()
            user = repository.create_user(
                username="document-owner",
                display_name="Document Owner",
                password_hash="test-hash",
            )
            first = repository.create_conversation(owner_user_id=user.id)
            second = repository.create_conversation(owner_user_id=user.id)
            # 原因：会话附件表受用户、文档所有权外键约束，测试必须走正式注册边界。
            # 作用：同时证明仓储查询只返回当前会话实际登记的文档。
            repository.register_document(
                "document-b",
                conversation_id=first.id,
                owner_user_id=user.id,
            )
            repository.register_document(
                "document-a",
                conversation_id=first.id,
                owner_user_id=user.id,
            )
            repository.register_document(
                "document-other",
                conversation_id=second.id,
                owner_user_id=user.id,
            )

            attached = repository.document_ids_for_conversation(first.id)

        self.assertEqual(set(attached), {"document-a", "document-b"})
        self.assertNotIn("document-other", attached)

    def test_initialize_migrates_legacy_memory_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "qwopus.db"
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE conversation_memory (
                        conversation_id TEXT PRIMARY KEY,
                        summary TEXT NOT NULL DEFAULT '',
                        summary_until_message_id TEXT,
                        pinned_facts TEXT NOT NULL DEFAULT '[]',
                        open_tasks TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    );
                    """
                )

            repository = ConversationRepository(database_path, import_legacy=False)
            repository.initialize()
            with sqlite3.connect(database_path) as connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(conversation_memory)"
                    ).fetchall()
                }

        self.assertIn("task_state", columns)


if __name__ == "__main__":
    unittest.main()
