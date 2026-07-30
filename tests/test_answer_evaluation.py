import unittest

from qwopus_agent.evaluation import AnswerBenchmarkCase, evaluate_answer
from qwopus_agent.services.orchestration_models import AnswerContract


class AnswerEvaluationTests(unittest.TestCase):
    def test_detailed_analysis_requires_quality_and_reference_concepts(self) -> None:
        answer = (
            "## 结论\n\n该方案适合先验证单机流程，因为职责边界清晰，但不能把进程隔离"
            "误认为完整安全沙箱。\n\n"
            "## 机制与证据\n\nPlanner 只生成计划，Executor 只执行任务，因此两部分可以"
            "分别测试；例如，路由测试可以固定计划，再验证 Skill 调用顺序。"
            "持久化知识按会话目录隔离，所以默认查询不会读取其他聊天。\n\n"
            "## 风险与验证\n\n模型断线会中断最终综合，因此需要设置请求超时、有限重试和"
            "整轮截止时间。上线前应注入超时和 503 错误，确认失败原因进入 Debug 记录，"
            "而用户界面只显示安全错误。两个任务并发写入同一会话时，还可能出现旧答案"
            "覆盖新答案的问题，因此写入应携带运行版本或条件更新，并用故意打乱完成顺序"
            "的测试验证最终消息归属。\n\n"
            "## 边界与实施顺序\n\n会话目录隔离只控制知识检索范围，不能阻止 Python Tool"
            "访问未授权路径；后者仍需要独立进程、路径白名单和资源限制。应先验证状态转换"
            "与失败恢复，再验证文件和知识权限，最后才增加并行 Agent。这样每一层失败都能"
            "定位到明确边界，也避免用更多模型调用掩盖底层一致性问题。"
        )
        report = evaluate_answer(
            AnswerBenchmarkCase(
                name="detailed-architecture",
                answer=answer,
                contract=AnswerContract(
                    task_type="analyze",
                    complexity="complex",
                    response_detail="detailed",
                ),
                required_concepts=(
                    ("Planner", "规划器"),
                    ("Executor", "执行器"),
                    ("超时", "timeout"),
                    ("会话", "conversation"),
                ),
                forbidden_phrases=("Observation:", "Thought:"),
            )
        )

        self.assertTrue(report.passed, report)
        self.assertEqual(report.concept_recall, 1.0)

    def test_generic_detailed_answer_fails_even_when_one_keyword_matches(self) -> None:
        report = evaluate_answer(
            AnswerBenchmarkCase(
                name="generic-analysis",
                answer="这个方案总体不错，因为需要持续完善。Planner 很重要。",
                contract=AnswerContract(
                    task_type="analyze",
                    complexity="complex",
                    response_detail="detailed",
                ),
                required_concepts=(
                    ("Planner",),
                    ("Executor",),
                    ("验证",),
                ),
            )
        )

        self.assertFalse(report.passed)
        self.assertIn("insufficient_depth", report.quality.issues)
        self.assertLess(report.concept_recall, 1.0)

    def test_concise_factual_answer_is_not_artificially_expanded(self) -> None:
        report = evaluate_answer(
            AnswerBenchmarkCase(
                name="concise-name",
                answer="项目名称是 Qwopus-Agent。",
                contract=AnswerContract(
                    task_type="answer",
                    complexity="simple",
                    response_detail="concise",
                ),
                required_concepts=(("Qwopus-Agent",),),
            )
        )

        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
