import tempfile
import unittest
from pathlib import Path

from qwopus_agent.api.repository import ConversationRepository


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


if __name__ == "__main__":
    unittest.main()
