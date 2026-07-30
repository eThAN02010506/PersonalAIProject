import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.evaluation import RetrievalBenchmarkCase, evaluate_retrieval
from tests.minirag_fakes import make_test_minirag


class RetrievalEvaluationTests(unittest.TestCase):
    def test_real_minirag_results_meet_source_reference_thresholds(self) -> None:
        with TemporaryDirectory() as tmpdir:
            memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            memory.insert(
                "# File: finance.md\n\nAnnual revenue reached 12 million in 2026."
            )
            memory.insert(
                "# File: staffing.md\n\nThe engineering team added four positions."
            )

            results = memory.search("annual revenue 12 million", min_relevance=0.25)
            report = evaluate_retrieval(
                RetrievalBenchmarkCase(
                    name="annual-revenue",
                    query="annual revenue 12 million",
                    expected_sources=("finance.md",),
                    forbidden_sources=("staffing.md",),
                ),
                results,
            )

        self.assertTrue(report.passed, report)
        self.assertEqual(report.source_recall, 1.0)
        self.assertEqual(report.reciprocal_rank, 1.0)

    def test_missing_and_noisy_sources_fail_independent_dimensions(self) -> None:
        report = evaluate_retrieval(
            RetrievalBenchmarkCase(
                name="cross-document",
                query="compare plan alpha and plan beta",
                expected_sources=("alpha.md", "beta.md"),
                forbidden_sources=("unrelated.md",),
                min_source_precision=0.75,
            ),
            [
                "[Source: alpha.md]\nAlpha evidence.",
                "[Source: unrelated.md]\nNoise.",
            ],
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.source_recall, 0.5)
        self.assertEqual(report.source_precision, 0.5)
        self.assertEqual(report.missing_sources, ("beta.md",))
        self.assertEqual(report.forbidden_hits, ("unrelated.md",))

    def test_graph_result_counts_each_evidence_source_once(self) -> None:
        report = evaluate_retrieval(
            RetrievalBenchmarkCase(
                name="graph-path",
                query="How is A related to C?",
                expected_sources=("a.md", "b.md"),
            ),
            [
                "[Knowledge Graph Path]\n"
                "A -[owns]-> B\nB -[funds]-> C\n"
                "Evidence:\n"
                "- [Source: a.md] A owns B\n"
                "- [Source: b.md] B funds C"
            ],
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.retrieved_sources, ("a.md", "b.md"))


if __name__ == "__main__":
    unittest.main()
