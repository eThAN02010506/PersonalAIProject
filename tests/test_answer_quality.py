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

    def test_detailed_analysis_rejects_disconnected_bullet_stack(self) -> None:
        intent = self.resolver.resolve(
            "请详细分析该系统的机制、证据、风险、替代方案和建议",
            response_detail="detailed",
        )
        answer = "\n".join(
            (
                "- 机制：系统通过规划器选择任务，但这里没有解释组件之间的因果联系与影响。",
                "- 证据：测试覆盖了主要入口，但这里没有说明证据如何支持最终判断。",
                "- 示例：可以处理文档查询，但这里没有展开真实场景中的执行过程。",
                "- 边界：模型断线时会失败，但这里没有解释恢复条件和剩余风险。",
                "- 替代：可以改用单代理，但这里没有比较何时应当采用这一方案。",
                "- 风险：并发任务可能竞争资源，但这里没有分析发生条件和后果。",
                "- 建议：继续增加测试，但这里没有给出优先级、步骤和验收标准。",
            )
        )

        report = self.evaluator.evaluate(answer, intent)

        self.assertFalse(report.passed)
        self.assertIn("fragmented_answer", report.issues)

    def test_detailed_how_to_allows_an_ordered_step_list(self) -> None:
        intent = self.resolver.resolve(
            "请详细说明如何部署、验证并排查这个服务",
            response_detail="detailed",
        )
        answer = "\n".join(
            (
                "部署前先确认 Python 环境、模型地址、写入目录和端口均可用。",
                "",
                "1. 创建隔离环境并安装锁定依赖，记录安装输出用于诊断。",
                "2. 配置模型兼容接口，通过健康检查确认模型名称和连接状态。",
                "3. 启动 FastAPI 服务，观察启动日志并确认数据库迁移成功完成。",
                "4. 从正式前端发送简单问题，验证直接路径只产生一次模型调用。",
                "5. 上传测试文档并开启知识检索，确认回答引用正确文件和页面。",
                "6. 断开模型连接执行失败案例，确认用户看到安全错误而非内部异常。",
                "7. 恢复服务后重复请求，检查会话、文件和 Debug 记录仍保持一致。",
                "",
                "每一步都应保存明确的通过条件；若某一步失败，只回退并修复该层，"
                "不要同时改动模型、知识库和前端，以免无法定位根因。",
            )
        )

        report = self.evaluator.evaluate(answer, intent)

        self.assertNotIn("fragmented_answer", report.issues)


if __name__ == "__main__":
    unittest.main()
