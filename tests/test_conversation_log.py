import tempfile
import unittest
from pathlib import Path

from qwopus_agent.utils.conversation_log import (
    LEGACY_CONVERSATION_ID,
    append_conversation_event,
    conversation_title,
    create_conversation,
    delete_conversation,
    list_conversations,
    load_chat_messages,
    rename_conversation,
)


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

    def test_conversations_keep_messages_separate_and_sort_by_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "conversations.jsonl"
            first_id = create_conversation(log_path=log_path)
            second_id = create_conversation("Second", log_path=log_path)
            append_conversation_event(
                "chat_message",
                {"role": "user", "content": "First question"},
                log_path=log_path,
                conversation_id=first_id,
            )
            append_conversation_event(
                "chat_message",
                {"role": "assistant", "content": "First answer"},
                log_path=log_path,
                conversation_id=first_id,
            )

            self.assertEqual(
                load_chat_messages(log_path, conversation_id=second_id),
                [],
            )
            self.assertEqual(len(load_chat_messages(log_path, conversation_id=first_id)), 2)
            summaries = list_conversations(log_path)
            self.assertEqual(summaries[0].conversation_id, first_id)
            self.assertEqual(summaries[0].title, "First question")

    def test_rename_and_delete_conversation_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "conversations.jsonl"
            conversation_id = create_conversation(log_path=log_path)
            rename_conversation(conversation_id, "Project notes", log_path=log_path)
            self.assertEqual(list_conversations(log_path)[0].title, "Project notes")

            delete_conversation(conversation_id, log_path=log_path)
            self.assertEqual(list_conversations(log_path), [])

    def test_legacy_messages_are_grouped_without_rewriting_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "conversations.jsonl"
            append_conversation_event(
                "chat_message",
                {"role": "user", "content": "Legacy question"},
                log_path=log_path,
            )

            summary = list_conversations(log_path)[0]
            self.assertEqual(summary.conversation_id, LEGACY_CONVERSATION_ID)
            self.assertEqual(summary.title, "历史对话")
            self.assertEqual(
                load_chat_messages(log_path, conversation_id=LEGACY_CONVERSATION_ID)[0]["content"],
                "Legacy question",
            )

    def test_conversation_title_is_compact(self) -> None:
        self.assertEqual(conversation_title("  hello\n world  "), "hello world")
        self.assertEqual(len(conversation_title("x" * 100)), 32)
