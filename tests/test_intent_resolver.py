import unittest

from qwopus_agent.services.intent_resolver import (
    IntentResolver,
    build_context_snapshot,
)
from qwopus_agent.services.orchestration_models import ConversationTaskState


class IntentResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = IntentResolver()

    def test_build_snapshot_merges_persisted_and_current_sources_once(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            task_state=ConversationTaskState(
                last_successful_objective="Compare the architecture options.",
                active_document_sources=("alpha.pdf",),
                open_tasks=("Verify the recommendation.",),
            ),
            document_sources=("alpha.pdf", "beta.docx"),
            active_skill_names=("document_parser", "document_parser"),
        )

        self.assertEqual(snapshot.previous_objective, "Compare the architecture options.")
        self.assertEqual(snapshot.document_sources, ("alpha.pdf", "beta.docx"))
        self.assertEqual(snapshot.active_skill_names, ("document_parser",))

    def test_contextual_continuation_inherits_previous_objective(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            task_state=ConversationTaskState(
                last_successful_objective="分析 Qwopus-Agent 的上下文管理方案。",
            ),
        )

        result = self.resolver.resolve("再详细一点", snapshot=snapshot)

        self.assertFalse(result.requires_clarification)
        self.assertEqual(result.task_type, "analyze")
        self.assertIn("上下文管理方案", result.operational_objective)
        self.assertIn("再详细一点", result.operational_objective)
        self.assertEqual(result.context_references[0].kind, "task")

    def test_continuation_without_previous_task_requires_clarification(self) -> None:
        result = self.resolver.resolve("继续")

        self.assertTrue(result.requires_clarification)
        self.assertIn("具体任务", result.clarification_question or "")

    def test_precise_mode_does_not_inherit_previous_objective(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            task_state=ConversationTaskState(
                last_successful_objective="Analyze the old design.",
            ),
        )

        result = self.resolver.resolve(
            "Continue with a new independent answer.",
            snapshot=snapshot,
            interpretation_mode="precise",
        )

        self.assertNotIn("old design", result.operational_objective)
        self.assertFalse(result.context_references)
        self.assertTrue(result.requires_clarification)

    def test_precise_mode_clarifies_a_bare_chinese_continuation(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            task_state=ConversationTaskState(
                last_successful_objective="分析 Qwopus-Agent 的上下文管理方案。",
            ),
        )

        result = self.resolver.resolve(
            "再详细一点",
            snapshot=snapshot,
            interpretation_mode="precise",
        )

        self.assertTrue(result.requires_clarification)
        self.assertIn("具体任务", result.clarification_question or "")
        self.assertNotIn("上下文管理方案", result.operational_objective)

    def test_single_document_pronoun_is_bound_to_the_only_source(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            document_sources=("requirements.pdf",),
        )

        result = self.resolver.resolve("总结这个文档", snapshot=snapshot)

        self.assertFalse(result.requires_clarification)
        self.assertEqual(result.task_type, "summarize")
        self.assertEqual(result.context_references[0].label, "requirements.pdf")
        self.assertIn("source-grounded evidence", result.answer_contract.required_facets)

    def test_singular_pronoun_with_multiple_documents_requires_selection(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            document_sources=("alpha.pdf", "beta.docx"),
        )

        result = self.resolver.resolve("分析这个文档", snapshot=snapshot)

        self.assertTrue(result.requires_clarification)
        self.assertIn("alpha.pdf", result.clarification_question or "")
        self.assertIn("beta.docx", result.clarification_question or "")

    def test_plural_reference_selects_all_documents(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            document_sources=("alpha.pdf", "beta.docx"),
        )

        result = self.resolver.resolve("比较这些文档", snapshot=snapshot)

        self.assertFalse(result.requires_clarification)
        self.assertEqual(result.task_type, "compare")
        self.assertEqual(
            [reference.label for reference in result.context_references],
            ["alpha.pdf", "beta.docx"],
        )

    def test_ordinal_reference_selects_the_requested_document(self) -> None:
        snapshot = build_context_snapshot(
            conversation_id="conversation-1",
            document_sources=("alpha.pdf", "beta.docx"),
        )

        result = self.resolver.resolve("Summarize the second document.", snapshot=snapshot)

        self.assertEqual(
            [reference.label for reference in result.context_references],
            ["beta.docx"],
        )

    def test_exploratory_detailed_mode_expands_answer_contract(self) -> None:
        result = self.resolver.resolve(
            "How should this service be deployed?",
            interpretation_mode="exploratory",
            response_detail="detailed",
        )

        self.assertEqual(result.task_type, "how_to")
        self.assertIn("ordered steps", result.answer_contract.required_facets)
        self.assertIn("alternatives", result.answer_contract.required_facets)
        self.assertIn("second-order effects", result.answer_contract.required_facets)


if __name__ == "__main__":
    unittest.main()
