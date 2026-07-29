import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.api.app import create_app
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.documents import (
    DocumentStore,
    InvalidDocumentIdError,
    build_document_structure,
    chunk_document_structure,
)
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.services.orchestration_models import OrchestrationResult
from tests.minirag_fakes import make_test_minirag


class SavedDocumentStoreTests(unittest.TestCase):
    def test_load_document_confines_every_saved_artifact_to_validated_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DocumentStore(root / "documents")
            original = root / "lesson.md"
            markdown = "# Lesson\nspecific-source-fact-731"
            original.write_text(markdown, encoding="utf-8")
            structure = chunk_document_structure(
                build_document_structure(markdown, source=original.name)
            )
            store.persist(
                original_path=original,
                markdown=markdown,
                structure=structure,
                metadata={"parser": "markdown"},
            )

            saved = store.load_document(structure.document_id)

            self.assertEqual(saved.document.source, "lesson.md")
            self.assertEqual(saved.normalized_markdown, markdown)
            self.assertEqual(saved.normalized_path.name, "normalized.md")
            self.assertEqual(
                saved.normalized_path.read_text(encoding="utf-8"),
                markdown,
            )
            self.assertEqual(saved.metadata["parser"], "markdown")
            self.assertEqual(
                saved.original_path.parent.name,
                structure.document_id,
            )
            self.assertEqual(store.load_structure(structure.document_id), structure)
            with self.assertRaisesRegex(
                InvalidDocumentIdError,
                "unsupported characters",
            ):
                store.load_document(f"../{structure.document_id}")


class SavedDocumentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.repository = ConversationRepository(
            root / "qwopus.db",
            import_legacy=False,
        )
        self.document_store = DocumentStore(root / "documents")
        self.knowledge = ConversationKnowledgeManager(
            root=root / "knowledge",
            factory=make_test_minirag,
        )
        self.settings = SmolagentsModelSettings(
            model_id="test-model",
            base_url="http://127.0.0.1:9999/v1",
        )
        self.runtime = MagicMock()
        self.runtime.current_settings.return_value = self.settings
        self.client_context = TestClient(
            create_app(
                self.repository,
                knowledge_manager=self.knowledge,
                model_runtime=self.runtime,
                debug_directory=root / "debug",
                runtime_log_path=root / "runtime.log",
                document_store=self.document_store,
            )
        )
        self.client = self.client_context.__enter__()
        initialized = self.client.post(
            "/api/auth/bootstrap",
            json={
                "username": "saved-admin",
                "display_name": "Saved Admin",
                "password": "saved-password-123",
            },
        )
        self.assertEqual(initialized.status_code, 201)
        self.user_id = initialized.json()["user"]["id"]

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temp_directory.cleanup()

    def _save(self, source: str, body: str):
        original = Path(self.temp_directory.name) / source
        markdown = f"# {source}\n{body}"
        original.write_text(markdown, encoding="utf-8")
        structure = chunk_document_structure(
            build_document_structure(markdown, source=source)
        )
        self.document_store.persist(
            original_path=original,
            markdown=markdown,
            structure=structure,
            metadata={"parser": "markdown"},
        )
        self.repository.register_document(
            structure.document_id,
            owner_user_id=self.user_id,
        )
        return structure

    def _conversation(self) -> str:
        response = self.client.post(
            "/api/conversations",
            json={"title": "Saved documents"},
        )
        self.assertEqual(response.status_code, 201)
        return str(response.json()["id"])

    def test_attach_indexes_every_selected_source_in_current_conversation(self) -> None:
        first = self._save("lesson-21.md", "unique-lesson-21-fact")
        second = self._save("lesson-22.md", "unique-lesson-22-fact")
        conversation_id = self._conversation()

        response = self.client.post(
            f"/api/conversations/{conversation_id}/documents/attach",
            json={"document_ids": [first.document_id, second.document_id]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["attached_count"], 2)
        self.assertEqual(
            {item["source"] for item in response.json()["documents"]},
            {"lesson-21.md", "lesson-22.md"},
        )
        private = self.knowledge.get(conversation_id)
        self.assertEqual(
            private._list_sources(),
            ["lesson-21.md", "lesson-22.md"],
        )
        evidence = "\n".join(
            private.search("unique lesson 21 fact", min_relevance=0.25)
        )
        self.assertIn("unique-lesson-21-fact", evidence)

        repeated = self.client.post(
            f"/api/conversations/{conversation_id}/documents/attach",
            json={"document_ids": [first.document_id, second.document_id]},
        )
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(private._list_sources(), ["lesson-21.md", "lesson-22.md"])

    def test_attach_rejects_duplicate_invalid_and_missing_ids_before_mutation(self) -> None:
        saved = self._save("lesson-23.md", "unique-lesson-23-fact")
        conversation_id = self._conversation()
        endpoint = f"/api/conversations/{conversation_id}/documents/attach"

        duplicate = self.client.post(
            endpoint,
            json={"document_ids": [saved.document_id, saved.document_id]},
        )
        invalid = self.client.post(
            endpoint,
            json={"document_ids": [f"../{saved.document_id}"]},
        )
        missing = self.client.post(
            endpoint,
            json={
                "document_ids": [
                    saved.document_id,
                    "document-000000000000000000000000",
                ]
            },
        )

        self.assertEqual(duplicate.status_code, 422)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(self.knowledge.get(conversation_id)._list_sources(), [])

    def test_saved_document_analysis_reuses_normalized_files_and_original_names(self) -> None:
        first = self._save("lesson-24.pdf", "selected first evidence")
        second = self._save("lesson-25.docx", "selected second evidence")
        conversation_id = self._conversation()
        analysis_result = AnalysisResult(
            markdown_summary="# Summary",
            markdown_document="# Combined",
            document_structures=(first, second),
        )
        orchestrator = MagicMock()
        orchestrator.run_sync.return_value = OrchestrationResult(
            success=True,
            final_answer="Both saved documents analyzed.",
            route="single_agent",
            analysis_result=analysis_result,
        )

        with patch(
            "qwopus_agent.api.routes.documents.AgentOrchestrator",
            return_value=orchestrator,
        ):
            response = self.client.post(
                "/api/documents/analyze",
                json={
                    "conversation_id": conversation_id,
                    "document_ids": [first.document_id, second.document_id],
                    "question": "Compare every selected lesson.",
                    "analysis_mode": "full",
                },
            )

        self.assertEqual(response.status_code, 200)
        request = orchestrator.run_sync.call_args.args[0]
        self.assertEqual(
            [file.name for file in request.uploaded_files],
            ["lesson-24.pdf", "lesson-25.docx"],
        )
        self.assertTrue(
            all(
                file.local_path is not None
                and file.local_path.name == "normalized.md"
                and file.local_path.parent.name
                in {first.document_id, second.document_id}
                for file in request.uploaded_files
            )
        )
        self.assertEqual(
            [
                file.local_path.read_text(encoding="utf-8")
                for file in request.uploaded_files
                if file.local_path is not None
            ],
            [
                "# lesson-24.pdf\nselected first evidence",
                "# lesson-25.docx\nselected second evidence",
            ],
        )
        self.assertTrue(all(file.content is None for file in request.uploaded_files))


if __name__ == "__main__":
    unittest.main()
