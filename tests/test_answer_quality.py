import unittest

from qwopus_agent.services.answer_quality import AnswerQualityEvaluator
from qwopus_agent.services.intent_resolver import IntentResolver


class AnswerQualityEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = AnswerQualityEvaluator()
        self.resolver = IntentResolver()

    def test_short_complex_analysis_requires_refinement(self) -> None:
        intent = self.resolver.resolve(
            "请详细分析这个架构的优点、风险和限制",
            response_detail="detailed",
        )

        report = self.evaluator.evaluate("这个架构总体可行。", intent)

        self.assertFalse(report.passed)
        self.assertIn("insufficient_depth", report.issues)
        self.assertIn("missing_analysis_structure", report.issues)

    def test_simple_answer_does_not_require_artificial_expansion(self) -> None:
        intent = self.resolver.resolve(
            "项目名称是什么？",
            response_detail="detailed",
        )

        report = self.evaluator.evaluate("项目名称是 Qwopus-Agent。", intent)

        self.assertTrue(report.passed)

    def test_explicit_brevity_disables_depth_requirement(self) -> None:
        intent = self.resolver.resolve(
            "请简短总结，只给结论",
            response_detail="detailed",
        )

        report = self.evaluator.evaluate("结论：方案可行，但需要补充恢复机制。", intent)

        self.assertEqual(intent.answer_contract.response_detail, "concise")
        self.assertTrue(report.passed)

    def test_detailed_comparison_with_structure_passes(self) -> None:
        intent = self.resolver.resolve(
            "详细比较方案 A 和方案 B",
            response_detail="detailed",
        )
        answer = (
            "## 共同点\n\n两种方案都提供模块边界、依赖注入和独立测试能力，"
            "也都能把模型实现隔离在统一接口之后。"
            "它们的共同目标是降低组件耦合并允许按需替换实现。\n\n"
            "## 主要差异\n\n方案 A 更强调同步调用和较低实现成本，适合任务量小、"
            "步骤固定的场景；方案 B 使用异步编排和持久状态，能够处理中断恢复、"
            "并发任务和更复杂的依赖图，但部署与诊断成本更高。\n\n"
            "## 权衡与结论\n\n若当前阶段优先验证业务闭环，方案 A 的风险更低；"
            "若已确认需要长任务恢复和并行执行，则方案 B 的额外复杂度是合理的。"
            "最终应以真实并发量、失败恢复目标和团队维护能力作为选择标准。"
        )

        report = self.evaluator.evaluate(answer, intent)

        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
