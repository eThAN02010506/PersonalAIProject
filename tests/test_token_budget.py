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


if __name__ == "__main__":
    unittest.main()
