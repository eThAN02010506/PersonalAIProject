import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.api.repository import ConversationRepository


class ConversationRunRepositoryTests(unittest.TestCase):
    def test_run_provenance_is_persistent_sanitized_and_cascades(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repository = ConversationRepository(
                Path(tmpdir) / "qwopus.db",
                import_legacy=False,
            )
            repository.initialize()
            conversation = repository.create_conversation("Research")
            user_message = repository.add_message(
                conversation.id,
                "user",
                "Research current rice prices",
            )
            assistant_message = repository.add_message(
                conversation.id,
                "assistant",
                "A source-grounded answer.",
            )

            repository.save_conversation_run(
                run_id="run-1",
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                requested_by_user_id=None,
                objective="Research current rice prices",
                operational_objective="Find and summarize current rice prices",
                status="completed",
                model_id="runtime-model",
                reusable_skills=("web_search", "rag_search"),
            )

            runs = repository.list_conversation_runs(conversation.id)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].reusable_skills, ("web_search", "rag_search"))
            self.assertEqual(runs[0].assistant_message_id, assistant_message.id)
            self.assertFalse(hasattr(runs[0], "tool_observations"))
            self.assertEqual(
                [item.id for item in repository.list_conversations_with_reusable_runs()],
                [conversation.id],
            )

            repository.delete_conversation(conversation.id)
            self.assertEqual(repository.list_conversation_runs(conversation.id), [])


if __name__ == "__main__":
    unittest.main()
