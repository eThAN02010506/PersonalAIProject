import unittest

import pandas as pd

from qwopus_agent.analysis.pandas_sandbox import execute_pandas_code


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

    def test_requires_result_assignment(self) -> None:
        dataframes = {"Sheet1": pd.DataFrame({"value": [1]})}

        with self.assertRaisesRegex(ValueError, "must assign"):
            execute_pandas_code('df = dfs["Sheet1"]', dataframes)

    def test_rejects_unknown_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown sandbox name"):
            execute_pandas_code("result = secret_value", {"Sheet1": pd.DataFrame()})


if __name__ == "__main__":
    unittest.main()
