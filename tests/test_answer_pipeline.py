import unittest

from qwopus_agent.services.answer_pipeline import (
    build_answer_plan,
    build_evidence_ledger,
    parse_evidence_packet,
    parse_evidence_review,
)
from qwopus_agent.services.orchestration_models import AnswerContract, SourceCitation


class AnswerPipelineTests(unittest.TestCase):
    def test_detailed_plan_adds_task_specific_depth_without_word_target(self) -> None:
        plan = build_answer_plan(
            "Analyze the architecture.",
            AnswerContract(
                task_type="analyze",
                complexity="complex",
                response_detail="detailed",
                required_facets=("findings", "evidence", "risks"),
            ),
        )

        self.assertEqual(plan.required_sections[0], "direct answer")
        self.assertIn("What evidence supports each major finding?", plan.depth_questions)
        self.assertTrue(any("specificity" in rule for rule in plan.style_rules))
        self.assertNotIn("800", plan.model_dump_json())

    def test_evidence_packet_parses_json_and_attaches_runtime_citations(self) -> None:
        packet = parse_evidence_packet(
            """
            {
              "facts": [{
                "claim": "The service is online.",
                "support": "The health endpoint returned status ok.",
                "sources": ["health response"],
                "confidence": "high"
              }],
              "limitations": ["One point-in-time check."]
            }
            """,
            task_id="research",
            agent_name="research_agent",
            citations=(
                SourceCitation(
                    kind="web",
                    source="status",
                    url="https://example.com/status",
                ),
            ),
        )

        self.assertEqual(packet.facts[0].confidence, "high")
        self.assertEqual(
            packet.facts[0].sources,
            ("health response", "https://example.com/status"),
        )

    def test_weak_model_text_falls_back_to_bounded_evidence(self) -> None:
        packet = parse_evidence_packet(
            "The first finding is supported. Additional explanation follows.",
            task_id="knowledge",
            agent_name="knowledge_agent",
            fallback_confidence=0.72,
        )

        self.assertEqual(packet.facts[0].claim, "The first finding is supported.")
        self.assertEqual(packet.facts[0].confidence, "medium")
        self.assertIn("deterministic evidence fallback", packet.limitations[0])

    def test_ledger_deduplicates_facts_and_merges_sources(self) -> None:
        first = parse_evidence_packet(
            '{"facts":[{"claim":"Shared fact","support":"Same support",'
            '"sources":["a.md"],"confidence":"medium"}]}',
            task_id="document",
            agent_name="document_agent",
        )
        second = parse_evidence_packet(
            '{"facts":[{"claim":"Shared fact","support":"Same support",'
            '"sources":["b.md"],"confidence":"high"}]}',
            task_id="knowledge",
            agent_name="knowledge_agent",
        )

        ledger = build_evidence_ledger((first, second))

        self.assertEqual(len(ledger.facts), 1)
        self.assertEqual(ledger.facts[0].sources, ("a.md", "b.md"))
        self.assertEqual(ledger.facts[0].confidence, "high")

    def test_unstructured_review_never_triggers_automatic_gap_fill(self) -> None:
        review = parse_evidence_review("The answer probably needs more work.")

        self.assertEqual(review.gaps, ())
        self.assertIn("not structured", review.unsupported_claims[0])

    def test_untrusted_model_sources_are_removed_without_tool_evidence(self) -> None:
        packet = parse_evidence_packet(
            (
                '{"facts":[{"claim":"Architecture claim","support":"Model reasoning",'
                '"sources":["Invented Study, 2025, p. 1"],"confidence":"high"}],'
                '"limitations":[]}'
            ),
            task_id="chat",
            agent_name="chat_agent",
            trust_declared_sources=False,
        )

        # 原因：任意模型都可能用看似正规的书名和页码填充 sources。
        # 作用：没有 Tool Observation 时不让模型自报来源进入最终 Ledger。
        self.assertEqual(packet.facts[0].sources, ())
        self.assertEqual(packet.facts[0].confidence, "medium")
        self.assertIn("No tool-grounded source", packet.limitations[0])


if __name__ == "__main__":
    unittest.main()
