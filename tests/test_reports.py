import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qwopus_agent.reports import ReportGenerator


class ReportGeneratorTests(unittest.TestCase):
    def test_report_generator_creates_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=Path(tmpdir))
            tables = {"summary": pd.DataFrame([{"metric": "rows", "value": 3}])}

            # 原因：报告生成必须通过统一模块完成，不能散落在 UI 或 Skill 中。
            # 作用：验证 Markdown、Excel、Charts manifest、PDF 都由同一入口产出。
            report = generator.generate(
                title="Analysis Report",
                markdown_body="Final answer body.",
                tables=tables,
                basename="analysis report",
            )

            artifact_kinds = {artifact.kind for artifact in report.artifacts}
            self.assertEqual(
                artifact_kinds,
                {"markdown", "excel", "charts", "pdf"},
            )
            self.assertTrue(report.markdown.exists())
            self.assertIn("Final answer body.", report.markdown.read_text(encoding="utf-8"))
            for artifact in report.artifacts:
                self.assertTrue(artifact.path.exists(), artifact.path)


if __name__ == "__main__":
    unittest.main()
