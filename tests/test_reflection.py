import unittest

from qwopus_agent.reflection import TaskReflectionEvaluator


class TaskReflectionEvaluatorTests(unittest.TestCase):
    def test_reflection_accepts_complete_successful_answer(self) -> None:
        evaluator = TaskReflectionEvaluator(min_answer_chars=20)

        result = evaluator.evaluate(
            objective="Summarize the uploaded document",
            answer="This answer is complete enough for the requested task.",
            success=True,
            step_names=["document_parser"],
        )

        self.assertFalse(result.needs_retry)
        self.assertEqual(result.observations, [])

    def test_reflection_flags_missing_trace_and_short_answer(self) -> None:
        evaluator = TaskReflectionEvaluator(min_answer_chars=20)

        # 原因：Reflection 要能发现“有输出但质量/轨迹不足”的情况。
        # 作用：为后续自动重试、报告审计和调试提示提供结构化信号。
        result = evaluator.evaluate(
            objective="Analyze",
            answer="Too short.",
            success=True,
            step_names=[],
        )

        self.assertTrue(result.needs_retry)
        self.assertIn("Final answer is very short.", result.observations)
        self.assertIn("No execution steps were recorded.", result.observations)


if __name__ == "__main__":
    unittest.main()
