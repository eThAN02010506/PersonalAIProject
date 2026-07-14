import tempfile
import unittest
from pathlib import Path

from qwopus_agent.utils.conversation_log import append_conversation_event, load_chat_messages


class ConversationLogTests(unittest.TestCase):
    def test_append_and_load_recent_chat_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "conversations.jsonl"

            append_conversation_event(
                "chat_message",
                {"role": "user", "content": "你好"},
                log_path=log_path,
            )
            append_conversation_event(
                "chat_message",
                {"role": "assistant", "content": "你好，我在。"},
                log_path=log_path,
            )
            append_conversation_event(
                "analysis",
                {"answer": "not chat"},
                log_path=log_path,
            )

            self.assertEqual(
                load_chat_messages(log_path=log_path),
                [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，我在。"},
                ],
            )

    def test_load_chat_messages_ignores_missing_log(self) -> None:
        self.assertEqual(load_chat_messages(Path("missing.jsonl")), [])
