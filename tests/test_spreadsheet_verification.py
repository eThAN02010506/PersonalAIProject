"""Tests for model self-computed spreadsheet value verification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qwopus_agent.integrations.spreadsheet_verification import (
    _close_enough,
    _extract_claimed_value,
    _values_match,
    local_verify_missing_spreadsheet_methods,
    verify_one_method,
)


class ClaimedValueExtractionTests(unittest.TestCase):
    def test_extracts_chinese_mean(self) -> None:
        self.assertEqual(
            _extract_claimed_value("describe", "Sepal.Length 的平均值是 5.8433。"),
            (5.8433,),
        )

    def test_extracts_english_mean(self) -> None:
        self.assertEqual(
            _extract_claimed_value("describe", "The mean of Sepal.Length is 5.8433."),
            (5.8433,),
        )

    def test_extracts_median(self) -> None:
        self.assertEqual(
            _extract_claimed_value("quantiles", "中位数是 5.8。"),
            (5.8,),
        )

    def test_extracts_outlier_count_zero(self) -> None:
        self.assertEqual(
            _extract_claimed_value("iqr_outliers", "outlier_count: 0"),
            (0,),
        )

    def test_extracts_outlier_count_positive(self) -> None:
        self.assertEqual(
            _extract_claimed_value("zscore_outliers", "outlier_count is 3"),
            (3,),
        )

    def test_extracts_p_value_scientific_notation(self) -> None:
        self.assertEqual(
            _extract_claimed_value(
                "normality_test", "p-value = 0.056824, 不拒绝正态。"
            ),
            (0.056824,),
        )

    def test_extracts_r_squared(self) -> None:
        self.assertEqual(
            _extract_claimed_value("linear_regression", "R² = 0.013823"),
            (0.013823,),
        )

    def test_returns_none_when_no_number(self) -> None:
        self.assertIsNone(
            _extract_claimed_value("describe", "数据看起来比较集中。")
        )


class ValueMatchTests(unittest.TestCase):
    def test_accepts_rounded_local_value(self) -> None:
        # 模型报 4 位小数，本地 6 位，容差内应接受。
        self.assertTrue(_values_match((5.843333,), (5.8433,)))

    def test_rejects_wrong_value(self) -> None:
        self.assertFalse(_values_match((5.843333,), (6.2,)))

    def test_accepts_exact_integer_count(self) -> None:
        self.assertTrue(_values_match((3,), (3,)))
        self.assertFalse(_values_match((3,), (4,)))

    def test_close_enough_abs_and_rel_tolerance(self) -> None:
        self.assertTrue(_close_enough(5.843333, 5.8433))
        self.assertTrue(_close_enough(1000.0, 1000.5))
        self.assertFalse(_close_enough(5.84, 6.2))


class VerificationEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "iris.xlsx"
        data = [5.1, 4.9, 5.8, 6.4, 5.843333, 5.2, 5.0, 6.1, 6.3, 5.7]
        pd.DataFrame(
            {
                "Sepal.Length": data,
                "Sepal.Width": [3.5, 3.0, 3.2, 3.1, 2.8, 3.6, 3.9, 3.4, 3.0, 3.7],
            }
        ).to_excel(self.path, index=False)
        self.true_mean = round(sum(data) / len(data), 4)
        self.paths = {"iris.xlsx": self.path}

    def test_verify_one_method_accepts_correct_self_computed_mean(self) -> None:
        narrative = f"Sepal.Length 的平均值是 {self.true_mean}。"
        result = verify_one_method(
            ("excel_statistics", "describe"),
            spreadsheet_names=["iris.xlsx"],
            spreadsheet_paths=self.paths,
            user_question="计算 Sepal.Length 的平均值并返回表格",
            narrative=narrative,
            debug_steps=[],
        )
        self.assertIsNotNone(result)
        synthetic_step, prose = result
        self.assertEqual(
            synthetic_step["tool_calls"][0]["function"]["name"],
            "excel_statistics",
        )
        self.assertIn("复核 describe", prose)

    def test_verify_one_method_degrades_on_wrong_self_computed_mean(self) -> None:
        # B：本地复算成功但值不匹配时降级为展示本地核验表，不再引用模型自算值。
        narrative = "Sepal.Length 的平均值是 6.2。"
        result = verify_one_method(
            ("excel_statistics", "describe"),
            spreadsheet_names=["iris.xlsx"],
            spreadsheet_paths=self.paths,
            user_question="计算 Sepal.Length 的平均值并返回表格",
            narrative=narrative,
            debug_steps=[],
        )
        self.assertIsNotNone(result)
        synthetic_step, prose = result
        self.assertIn("重新计算", prose)
        self.assertNotIn("6.2", prose)

    def test_verify_one_method_degrades_without_number(self) -> None:
        # B：模型没写数字时同样降级为本地核验表，而不是 fail-closed。
        narrative = "数据分布看起来比较集中。"
        result = verify_one_method(
            ("excel_statistics", "describe"),
            spreadsheet_names=["iris.xlsx"],
            spreadsheet_paths=self.paths,
            user_question="计算 Sepal.Length 的平均值并返回表格",
            narrative=narrative,
            debug_steps=[],
        )
        self.assertIsNotNone(result)
        synthetic_step, prose = result
        self.assertIn("重新计算", prose)

    def test_local_verify_appends_step_and_discards_missing_tool(self) -> None:
        steps = [
            {
                "step_number": 1,
                "observations": f"Sepal.Length 的平均值是 {self.true_mean}。",
                "tool_calls": [
                    {"function": {"name": "excel_schema", "arguments": {}}}
                ],
            }
        ]
        missing_tools = {"excel_statistics"}
        result = local_verify_missing_spreadsheet_methods(
            steps,
            spreadsheet_paths=self.paths,
            spreadsheet_names=["iris.xlsx"],
            user_question="计算 Sepal.Length 的平均值并返回表格",
            missing_tools=missing_tools,
            required_spreadsheet_methods=(("excel_statistics", "describe"),),
            debug_steps=[],
        )
        self.assertTrue(result)
        self.assertNotIn("excel_statistics", missing_tools)
        self.assertEqual(len(steps), 2)
        self.assertEqual(
            steps[-1]["tool_calls"][0]["function"]["name"],
            "excel_statistics",
        )

    def test_local_verify_degrades_and_discards_missing_tool_on_mismatch(self) -> None:
        # B：自算值错误时本地复算成功即降级接受，missing_tool 被清掉。
        steps = [
            {
                "step_number": 1,
                "observations": "Sepal.Length 的平均值是 9.9。",
                "tool_calls": [],
            }
        ]
        missing_tools = {"excel_statistics"}
        result = local_verify_missing_spreadsheet_methods(
            steps,
            spreadsheet_paths=self.paths,
            spreadsheet_names=["iris.xlsx"],
            user_question="计算 Sepal.Length 的平均值并返回表格",
            missing_tools=missing_tools,
            required_spreadsheet_methods=(("excel_statistics", "describe"),),
            debug_steps=[],
        )
        self.assertTrue(result)
        self.assertNotIn("excel_statistics", missing_tools)

    def test_local_comparison_values_reads_rows_and_tables(self) -> None:
        # 回归测试：describe 的 mean 在 rows，regression 的 r_squared 在 tables，
        # ANOVA 的 p_value 在 tables 且必须过滤 NaN 行。
        import qwopus_agent.integrations.spreadsheet_verification as module

        # describe → mean 在 rows
        describe = module._run_local_method(
            "excel_statistics",
            "describe",
            self.path,
            "计算 Sepal.Length 的均值",
        )
        mean_values = module._local_comparison_values("describe", describe.data)
        self.assertTrue(mean_values)
        self.assertAlmostEqual(mean_values[0], self.true_mean, places=3)

        # iqr_outliers → outlier_count 在 details
        iqr = module._run_local_method(
            "excel_statistics",
            "iqr_outliers",
            self.path,
            "Sepal.Length IQR 离群点",
        )
        self.assertEqual(module._local_comparison_values("iqr_outliers", iqr.data), (0.0,))

        # linear_regression → r_squared 在 named tables
        with tempfile.TemporaryDirectory() as tmpdir:
            regression_path = Path(tmpdir) / "regression.xlsx"
            pd.DataFrame(
                {
                    "y": [2.0, 3.0, 4.0, 5.0, 6.0],
                    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            ).to_excel(regression_path, index=False)
            regression = module._run_local_method(
                "excel_modeling",
                "linear_regression",
                regression_path,
                "回归 y 对 x",
            )
            r_squared = module._local_comparison_values(
                "linear_regression", regression.data
            )
            self.assertTrue(r_squared)
            self.assertGreater(r_squared[0], 0.9)  # 完全线性数据 R² 应接近 1

        # one_way_anova → p_value 在 ANOVA 表，且不含 NaN 行
        with tempfile.TemporaryDirectory() as tmpdir:
            anova_path = Path(tmpdir) / "anova.xlsx"
            pd.DataFrame(
                {
                    "Sepal.Length": [5.1, 4.9, 5.8, 6.4, 5.5, 6.0, 5.2, 4.8],
                    "Species": ["a", "a", "a", "b", "b", "b", "c", "c"],
                }
            ).to_excel(anova_path, index=False)
            anova = module._run_local_method(
                "excel_modeling",
                "one_way_anova",
                anova_path,
                "按 Species 对 Sepal.Length 做方差分析",
            )
            anova_values = module._local_comparison_values("one_way_anova", anova.data)
            self.assertTrue(anova_values)
            self.assertTrue(all(value == value for value in anova_values))  # 无 NaN

    def test_schema_targets_parses_table_and_columns(self) -> None:
        import qwopus_agent.integrations.spreadsheet_verification as module

        steps = [
            {
                "step_number": 1,
                "observations": (
                    "# Spreadsheet Analysis: iris.xlsx\n"
                    "## Sheet: Sheet1\n"
                    "- Rows: 150\n"
                    "- Column names: Sepal.Length, Sepal.Width, Species\n"
                ),
                "tool_calls": [
                    {"function": {"name": "excel_schema", "arguments": {}}}
                ],
            }
        ]
        targets = module._schema_targets(steps)
        self.assertIn("Sheet1", targets)
        self.assertIn("Sepal.Length", targets["Sheet1"])
        self.assertIn("Species", targets["Sheet1"])

    def test_choose_table_prefers_schema_table_name(self) -> None:
        import pandas as pd

        import qwopus_agent.integrations.spreadsheet_verification as module

        frames = {
            "Data::table_1": pd.DataFrame({"value": [1, 2]}),
            "Sheet1": pd.DataFrame({"Sepal.Length": [5.1], "Species": ["a"]}),
        }
        chosen = module._choose_table(
            frames,
            "分析 Sheet1 的数据",
            {"Sheet1": ["Sepal.Length", "Species"]},
        )
        self.assertEqual(chosen, "Sheet1")

    def test_verify_prefers_schema_columns_for_value_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.xlsx"
            pd.DataFrame(
                {
                    "Sepal.Length": [5.1, 4.9, 5.8, 6.4, 5.5],
                    "Sepal.Width": [3.5, 3.0, 3.2, 3.1, 2.8],
                }
            ).to_excel(path, index=False)
            schema_steps = [
                {
                    "step_number": 1,
                    "observations": (
                        "# Spreadsheet Analysis: multi.xlsx\n"
                        "## Sheet: Sheet1\n"
                        "- Column names: Sepal.Length, Sepal.Width\n"
                    ),
                    "tool_calls": [
                        {"function": {"name": "excel_schema", "arguments": {}}}
                    ],
                }
            ]
            narrative = "Sepal.Length 的平均值是 5.54。"
            result = verify_one_method(
                ("excel_statistics", "describe"),
                spreadsheet_names=["multi.xlsx"],
                spreadsheet_paths={"multi.xlsx": path},
                user_question="计算 Sepal.Length 的平均值",
                narrative=narrative,
                debug_steps=[],
                steps=schema_steps,
            )
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
