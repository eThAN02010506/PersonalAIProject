import unittest

from qwopus_agent.utils.token_budget import (
    TokenBudgetManager,
    estimate_tokens,
    truncate_to_tokens,
)


class TokenBudgetTests(unittest.TestCase):
    def test_multilingual_estimate_and_truncation_share_one_limit(self) -> None:
        text = "中文内容 mixed with English words. " * 50

        truncated = truncate_to_tokens(text, 80)

        self.assertGreater(estimate_tokens(text), 80)
        self.assertLessEqual(estimate_tokens(truncated), 80)

    def test_evidence_budget_reserves_system_history_output_and_safety(self) -> None:
        budget = TokenBudgetManager(
            context_window=16000,
            output_reserve=2000,
            system_reserve=3000,
            history_reserve=2000,
            safety_reserve=1000,
        )

        self.assertEqual(budget.evidence_budget, 8000)

    def test_small_model_window_adapts_every_prompt_partition(self) -> None:
        budget = TokenBudgetManager(
            context_window=2048,
            output_reserve=1024,
            system_reserve=4096,
            history_reserve=4096,
            safety_reserve=2048,
        )

        # 原因：固定 reserve 在小模型上可能相加超过整个 context window。
        # 作用：锁定所有输入分区会自适应缩小，且 Tool/综合预算来自同一个剩余空间。
        self.assertLessEqual(
            budget.system_budget + budget.history_budget + budget.evidence_budget,
            budget.input_budget,
        )
        self.assertLessEqual(budget.observation_budget, budget.evidence_budget)
        self.assertLessEqual(budget.synthesis_budget, budget.evidence_budget)

    def test_invalid_context_and_negative_reserves_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2048"):
            TokenBudgetManager(context_window=1024)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            TokenBudgetManager(history_reserve=-1)


if __name__ == "__main__":
    unittest.main()
