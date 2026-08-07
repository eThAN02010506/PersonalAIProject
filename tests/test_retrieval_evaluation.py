import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.evaluation import RetrievalBenchmarkCase, evaluate_retrieval
from tests.minirag_fakes import make_test_minirag


class RetrievalRegressionSet(unittest.TestCase):
    """Fixed queries over a deterministic corpus with reference thresholds.

    原因：检索质量不能只断言“结果非空”，必须用期望来源、禁止来源和排名阈值
    检测漏召回、噪声与跨作用域泄漏。
    作用：作为 MiniRAG 检索的回归基线，覆盖跨文档隔离、噪声降级和中文查询。
    """

    CORPUS = {
        "finance.md": (
            "# File: finance.md\n\n"
            "Annual revenue reached 12 million in 2026. "
            "The firm's net income improved across the year. "
            "Revenue growth came from the core product line."
        ),
        "staffing.md": (
            "# File: staffing.md\n\n"
            "The engineering team added four positions. "
            "Hiring focused on backend roles and platform reliability. "
            "Headcount planning targets the next fiscal year."
        ),
        "marketing.md": (
            "# File: marketing.md\n\n"
            "The spring campaign lifted brand awareness. "
            "Budget allocation favored digital channels. "
            "Annual revenue reporting is owned by finance, not marketing."
        ),
        "china_report.md": (
            "# File: china_report.md\n\n"
            "年度收入在一季度创下新高，电商渠道增长最快。 "
            "本年度净利润改善来自核心产品线，与营销无关。"
        ),
    }

    def _memory(self, tmpdir: str) -> object:
        memory = make_test_minirag(Path(tmpdir) / "documents.jsonl")
        for body in self.CORPUS.values():
            memory.insert(body)
        return memory

    def test_revenue_query_recalls_finance_but_not_staffing(self) -> None:
        # 语义最匹配 finance.md；staffing 是无关来源，不应被召回。
        # marketing 提到 annual revenue 但主语是报告归属，允许作为低分次要来源。
        case = RetrievalBenchmarkCase(
            name="annual-revenue-isolation",
            query="annual revenue 12 million",
            expected_sources=("finance.md",),
            forbidden_sources=("staffing.md",),
            min_source_precision=0.5,
            min_reciprocal_rank=1.0,
        )
        with TemporaryDirectory() as tmpdir:
            results = self._memory(tmpdir).search(
                case.query, min_relevance=0.25
            )
            report = evaluate_retrieval(case, results)
        self.assertTrue(report.passed, report)
        self.assertEqual(report.source_recall, 1.0)
        self.assertEqual(report.reciprocal_rank, 1.0)
        self.assertEqual(report.forbidden_hits, ())

    def test_english_query_keeps_source_precision_with_noise_source(self) -> None:
        # marketing.md 提到 annual revenue，但不回答“收入达 1200 万”的事实。
        # 允许 precision 门槛吸收一个次要来源，但 finance 必须 rank 第一。
        case = RetrievalBenchmarkCase(
            name="revenue-with-noise",
            query="revenue reached twelve million",
            expected_sources=("finance.md",),
            forbidden_sources=("staffing.md",),
            min_source_precision=0.5,
            min_reciprocal_rank=1.0,
        )
        with TemporaryDirectory() as tmpdir:
            results = self._memory(tmpdir).search(
                case.query, min_relevance=0.25
            )
            report = evaluate_retrieval(case, results)
        self.assertTrue(report.passed, report)
        self.assertEqual(report.reciprocal_rank, 1.0)

    def test_chinese_revenue_query_recalls_expected_source(self) -> None:
        # 中文查询通过 TestEmbeddingBackend 的 alias（收入→revenue、年度→annual）命中
        # 含中文关键词的 china_report.md，且不泄漏到英文 finance.md。
        case = RetrievalBenchmarkCase(
            name="chinese-revenue",
            query="年度收入创新高",
            expected_sources=("china_report.md",),
            forbidden_sources=("staffing.md",),
            min_source_recall=1.0,
        )
        with TemporaryDirectory() as tmpdir:
            results = self._memory(tmpdir).search(
                case.query, min_relevance=0.2
            )
            report = evaluate_retrieval(case, results)
        self.assertTrue(report.passed, report)
        self.assertEqual(report.source_recall, 1.0)

    def test_headcount_query_does_not_leak_revenue_source(self) -> None:
        # staffing 是关于招聘的，finance 不应作为头寸查询的来源泄漏。
        case = RetrievalBenchmarkCase(
            name="headcount-isolation",
            query="engineering team headcount hiring",
            expected_sources=("staffing.md",),
            forbidden_sources=("finance.md",),
            min_source_recall=1.0,
            min_reciprocal_rank=1.0,
        )
        with TemporaryDirectory() as tmpdir:
            results = self._memory(tmpdir).search(
                case.query, min_relevance=0.25
            )
            report = evaluate_retrieval(case, results)
        self.assertTrue(report.passed, report)
        self.assertEqual(report.forbidden_hits, ())


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
