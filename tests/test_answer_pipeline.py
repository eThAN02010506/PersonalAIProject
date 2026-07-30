import unittest

from qwopus_agent.services.answer_pipeline import (
    build_answer_plan,
    build_evidence_ledger,
    is_internal_pipeline_payload,
    parse_evidence_packet,
    parse_evidence_review,
)
from qwopus_agent.services.orchestration_models import AnswerContract, SourceCitation


class AnswerPipelineTests(unittest.TestCase):
    def test_answer_plan_rejects_blank_objective_before_model_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Answer planning objective must not be blank",
        ):
            build_answer_plan("   ", AnswerContract())

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
        self.assertEqual(plan.plan_items[0].item_id, "P1")
        self.assertIn("Analyze the architecture", plan.plan_items[0].question)
        self.assertEqual(
            [item.section for item in plan.plan_items],
            ["direct answer", "findings", "evidence", "risks"],
        )

    def test_evidence_packet_parses_json_and_attaches_runtime_citations(self) -> None:
        packet = parse_evidence_packet(
            """
            {
              "facts": [{
                "claim": "The service is online.",
                "support": "The health endpoint returned status ok.",
                "sources": ["health response"],
                "confidence": "high",
                "plan_item_ids": ["P1", "invalid"]
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
        self.assertEqual(packet.facts[0].plan_item_ids, ("P1",))

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
            '"sources":["a.md"],"confidence":"medium","plan_item_ids":["P1"]}]}',
            task_id="document",
            agent_name="document_agent",
        )
        second = parse_evidence_packet(
            '{"facts":[{"claim":"Shared fact","support":"Same support",'
            '"sources":["b.md"],"confidence":"high","plan_item_ids":["P2"]}]}',
            task_id="knowledge",
            agent_name="knowledge_agent",
        )

        ledger = build_evidence_ledger((first, second))

        self.assertEqual(len(ledger.facts), 1)
        self.assertEqual(ledger.facts[0].sources, ("a.md", "b.md"))
        self.assertEqual(ledger.facts[0].confidence, "high")
        self.assertEqual(ledger.facts[0].plan_item_ids, ("P1", "P2"))

    def test_evidence_sources_are_bounded_after_parsing_and_merging(self) -> None:
        declared_sources = [f"document-{index}.md" for index in range(19)]
        first = parse_evidence_packet(
            (
                '{"facts":[{"claim":"Shared fact","support":"Same support",'
                f'"sources":{declared_sources!r},"confidence":"medium"}}]'
                ',"limitations":[]}'
            ).replace("'", '"'),
            task_id="knowledge",
            agent_name="knowledge_agent",
            citations=tuple(
                SourceCitation(kind="local", source=f"citation-{index}.md")
                for index in range(19)
            ),
            max_sources=20,
        )
        second = parse_evidence_packet(
            '{"facts":[{"claim":"Shared fact","support":"Same support",'
            '"sources":["another.md"],"confidence":"high"}],"limitations":[]}',
            task_id="research",
            agent_name="research_agent",
        )

        ledger = build_evidence_ledger((first, second), max_sources=20)

        # 原因：真实知识检索可能同时命中十几份文档，不能让来源数量使整轮 Agent 失败。
        # 作用：锁定解析和二次合并都遵守 EvidenceFact 的固定上限。
        self.assertEqual(len(first.facts[0].sources), 20)
        self.assertEqual(len(ledger.facts[0].sources), 20)

    def test_unstructured_review_never_triggers_automatic_gap_fill(self) -> None:
        review = parse_evidence_review("The answer probably needs more work.")

        self.assertEqual(review.gaps, ())
        self.assertIn("not structured", review.unsupported_claims[0])

    def test_structured_review_promotes_missing_coverage_to_one_gap(self) -> None:
        review = parse_evidence_review(
            """
            {
              "agreements": ["P1 is established."],
              "conflicts": [],
              "unsupported_claims": [],
              "gaps": [],
              "resolution": "Use P1 and retrieve P2.",
              "coverage": [
                {"plan_item_id": "P1", "status": "supported",
                 "finding": "The conclusion has direct support."},
                {"plan_item_id": "P2", "status": "missing",
                 "finding": "No deployment condition was supplied."}
              ]
            }
            """
        )

        self.assertEqual(review.coverage[0].status, "supported")
        self.assertEqual(
            review.gaps,
            ("P2: No deployment condition was supplied.",),
        )

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

    def test_internal_payload_detection_parses_structure_not_string_prefix(self) -> None:
        evidence = (
            'Draft follows:\n```json\n{"facts":[],"limitations":["missing source"]}\n```'
        )
        ordinary_json = '{"status":"ok","items":[1,2]}'

        self.assertTrue(is_internal_pipeline_payload(evidence))
        self.assertFalse(is_internal_pipeline_payload(ordinary_json))


if __name__ == "__main__":
    unittest.main()
