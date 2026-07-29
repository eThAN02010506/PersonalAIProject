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
