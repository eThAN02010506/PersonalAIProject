import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis.pandas_sandbox import (
    _sandbox_command,
    _sandbox_environment,
    execute_pandas_code,
)


class PandasSandboxTests(unittest.TestCase):
    def test_executes_groupby_analysis_and_returns_markdown(self) -> None:
        dataframes = {
            "Sheet1": pd.DataFrame(
                {
                    "region": ["East", "West", "East"],
                    "revenue": [10, 20, 30],
                }
            )
        }

        # 原因：Excel 分析需要让 LLM 生成的 pandas 代码在本地计算真实结果。
        # 作用：验证沙箱能执行常见 groupby 聚合，并只返回计算后的结果。
        result = execute_pandas_code(
            'df = dfs["Sheet1"]\nresult = df.groupby("region")["revenue"].sum().reset_index()',
            dataframes,
        )

        self.assertIn("East", result.markdown)
        self.assertIn("40", result.markdown)
        self.assertIn("West", result.markdown)

    def test_strips_markdown_fence(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1, 2]})}

        result = execute_pandas_code(
            '```python\ndf = dfs["Sheet1"]\nresult = df["value"].sum()\n```',
            dataframes,
        )

        self.assertEqual(result.value, 3)

    def test_coerces_mixed_report_columns_before_calculating_mean(self) -> None:
        dataframes = {
            "Report": pd.DataFrame(
                {
                    "label": ["Heading", "Item"],
                    "first": ["Year one", "1.0"],
                    "second": ["Year two", "3.0"],
                }
            )
        }

        result = execute_pandas_code(
            (
                'df = dfs["Report"]\n'
                'values = df[["first", "second"]].apply('
                'pd.to_numeric, errors="coerce")\n'
                'result = values.mean(axis=1).dropna().reset_index(name="mean")'
            ),
            dataframes,
        )

        # 原因：复杂 Excel 的标题文字会让数值列成为 object，直接 mean 会遗漏真实数据。
        # 作用：锁定 Agent 可在受限沙箱内安全转换列，并以表格返回计算后的平均值。
        self.assertEqual(result.value[0]["mean"], 2.0)
        self.assertIn("| mean |", result.markdown)

    def test_allows_descriptive_local_names_and_safe_missing_value_methods(self) -> None:
        dataframes = {
            "Report": pd.DataFrame(
                {
                    "label": ["First", "Second"],
                    "value": [10.0, None],
                    "other": [20.0, 30.0],
                }
            )
        }

        result = execute_pandas_code(
            (
                'source_frame = dfs["Report"]\n'
                'numeric_columns = source_frame.select_dtypes(include="number")\n'
                "missing_counts = numeric_columns.isna().sum()\n"
                "result = pd.DataFrame({"
                '"mean": numeric_columns.mean(), "missing": missing_counts'
                "}).reset_index()"
            ),
            dataframes,
        )

        # 原因：模型会为复杂统计生成有语义的中间变量名，名称本身不是逃逸能力。
        # 作用：确认沙箱允许正常 pandas 表达，同时结果仍只能来自已加载 DataFrame。
        self.assertEqual([row["mean"] for row in result.value], [10.0, 25.0])
        self.assertEqual([row["missing"] for row in result.value], [1, 0])

    def test_rejects_overwriting_sandbox_capabilities(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1]})}

        with self.assertRaisesRegex(ValueError, "safe local variable"):
            execute_pandas_code("pd = 1\nresult = pd", dataframes)
        with self.assertRaisesRegex(ValueError, "safe local variable"):
            execute_pandas_code("open = 1\nresult = open", dataframes)

    def test_allows_local_columns_but_rejects_mutating_workbook_mapping(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1, 2]})}

        result = execute_pandas_code(
            (
                'analysis_frame = dfs["Sheet1"].copy()\n'
                'analysis_frame["doubled"] = analysis_frame["value"] * 2\n'
                'result = analysis_frame[["value", "doubled"]]'
            ),
            dataframes,
        )

        # 原因：模型常用列赋值表达派生指标，而 worker 中的工作簿已深拷贝。
        # 作用：允许实用的数据变换，同时锁定共享 dfs 映射仍不可被替换。
        self.assertEqual(result.value[1]["doubled"], 4)
        with self.assertRaisesRegex(ValueError, "local variables or their columns"):
            execute_pandas_code(
                'dfs["Sheet1"] = pd.DataFrame()\nresult = 1',
                dataframes,
            )

    def test_allows_vectorized_stack_and_local_axis_labels(self) -> None:
        dataframes = {
            "Sheet1": pd.DataFrame(
                {"first": [1, 100], "second": [2, 200]},
                index=["A", "B"],
            )
        }

        result = execute_pandas_code(
            (
                'df = dfs["Sheet1"]\n'
                "stacked = df.stack().reset_index()\n"
                'stacked.columns = ["item", "metric", "value"]\n'
                "result = stacked.sort_values(\"value\", ascending=False).head(2)"
            ),
            dataframes,
        )

        # 原因：跨多列异常值分析需要把宽表向量化成长表，避免开放 Python 循环。
        # 作用：允许局部表的轴标签整理，但不开放任意对象属性修改。
        self.assertEqual([row["value"] for row in result.value], [200, 100])
        with self.assertRaisesRegex(ValueError, "local variables or their columns"):
            execute_pandas_code(
                'df = dfs["Sheet1"]\ndf.encoding = "utf-8"\nresult = df',
                dataframes,
            )

    def test_executes_vectorized_row_level_iqr_outlier_pattern(self) -> None:
        dataframes = {
            "Sheet1": pd.DataFrame(
                {
                    "label": ["A", "B", "C", "D", "Outlier"],
                    "first": [1, 2, 2, 3, 100],
                    "second": [1, 2, 2, 3, 100],
                }
            )
        }

        result = execute_pandas_code(
            (
                'df = dfs["Sheet1"]\n'
                'values = df[["first", "second"]]\n'
                "series = values.mean(axis=1)\n"
                "summary = series.quantile([0.25, 0.75])\n"
                "mask = (series < summary.iloc[0] - 1.5 * "
                "(summary.iloc[1] - summary.iloc[0])) | "
                "(series > summary.iloc[1] + 1.5 * "
                "(summary.iloc[1] - summary.iloc[0]))\n"
                'result = pd.DataFrame({"label": df["label"], "value": series})[mask]'
            ),
            dataframes,
        )

        # 原因：抽象异常值问题应由向量化 pandas 完成，不能诱导模型请求 Python 循环。
        # 作用：锁定 Prompt 中提供的 IQR 范式与实际沙箱能力完全一致。
        self.assertEqual(result.value, [{"label": "Outlier", "value": 100.0}])

    def test_rejects_imports(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported sandbox syntax"):
            execute_pandas_code("import os\nresult = 1", {"Sheet1": pd.DataFrame()})

    def test_rejects_file_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "Blocked sandbox call"):
            execute_pandas_code('result = open("secret.txt").read()', {"Sheet1": pd.DataFrame()})

    def test_rejects_dataframe_export(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1]})}

        with self.assertRaisesRegex(ValueError, "Blocked sandbox attribute"):
            execute_pandas_code('df = dfs["Sheet1"]\nresult = df.to_csv("out.csv")', dataframes)

    def test_rejects_unlisted_pandas_io_and_plot_methods(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1]})}

        with self.assertRaisesRegex(ValueError, "Blocked sandbox attribute"):
            execute_pandas_code('result = pd.read_pickle("secret.pkl")', dataframes)
        with self.assertRaisesRegex(ValueError, "Blocked sandbox attribute"):
            execute_pandas_code('df = dfs["Sheet1"]\nresult = df.plot()', dataframes)

    def test_rejects_loops_and_exponentiation_before_execution(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1]})}

        with self.assertRaisesRegex(ValueError, "Unsupported sandbox syntax"):
            execute_pandas_code("while True:\n    result = 1", dataframes)
        with self.assertRaisesRegex(ValueError, "Exponentiation"):
            execute_pandas_code("result = 10 ** 5", dataframes)

    def test_worker_errors_return_through_the_sandbox_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "ZeroDivisionError"):
            execute_pandas_code("result = 1 / 0", {"Sheet1": pd.DataFrame()})

    def test_requires_result_assignment(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1]})}

        with self.assertRaisesRegex(ValueError, "must assign"):
            execute_pandas_code('df = dfs["Sheet1"]', dataframes)

    def test_rejects_unknown_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown sandbox name"):
            execute_pandas_code("result = secret_value", {"Sheet1": pd.DataFrame()})

    def test_rejects_exposing_low_level_memory_buffer(self) -> None:
        # 原因：df.values.data 会暴露 numpy 底层内存视图，虽有方法白名单兜底但无分析用途。
        # 作用：把这类可绕过 pandas 高层 API 的属性访问加入黑名单。
        with self.assertRaisesRegex(ValueError, "Blocked sandbox attribute"):
            execute_pandas_code(
                'df = dfs["Sheet1"]\nresult = df.values.data',
                {"Sheet1": pd.DataFrame({"value": [1, 2]})},
            )

    @unittest.skipUnless(sys.platform == "darwin", "Seatbelt is specific to macOS.")
    def test_macos_worker_command_enforces_seatbelt_policy(self) -> None:
        command = _sandbox_command()
        profile = command[command.index("-p") + 1]

        # 原因：子进程隔离测试通过，并不能证明 macOS 命令没有退回普通 Python。
        # 作用：锁定真实 worker 必须经过 Seatbelt，且策略拒绝网络和文件写入。
        self.assertEqual(command[0], "/usr/bin/sandbox-exec")
        self.assertIn("(deny network*)", profile)
        self.assertIn("(deny file-write*)", profile)

    def test_worker_environment_does_not_inherit_application_secrets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TAVILY_API_KEY": "secret-search-key",
                "QWOPUS_LAN_PASSWORD": "secret-lan-password",
            },
        ):
            environment = _sandbox_environment()

        # 原因：即使生成代码无法读取 os.environ，worker 也不应持有它不需要的密钥。
        # 作用：将模型、搜索和 LAN 凭据留在父进程，缩小潜在逃逸后的影响范围。
        self.assertNotIn("TAVILY_API_KEY", environment)
        self.assertNotIn("QWOPUS_LAN_PASSWORD", environment)


if __name__ == "__main__":
    unittest.main()
