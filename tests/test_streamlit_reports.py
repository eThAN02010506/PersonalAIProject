import unittest
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.ui.streamlit_chat import _generate_analysis_report, _report_mime_type


class StreamlitReportDownloadTests(unittest.TestCase):
    def test_generate_analysis_report_creates_download_artifacts(self) -> None:
        result = AnalysisResult(
            markdown_summary="# Local summary",
            tables={"summary": pd.DataFrame([{"metric": "rows", "value": 3}])},
            metadata={},
            markdown_document="document",
            llm_analysis="Final answer body.",
        )

        with patch("qwopus_agent.ui.streamlit_chat.Path") as path_cls:
            path_cls.return_value = self._temporary_output_dir()
            # 原因：Streamlit 页面只负责触发下载，实际报告生成必须走 ReportGenerator。
            # 作用：验证 UI 辅助函数能从 AnalysisResult 产出真实图表下载 artifact。
            report = _generate_analysis_report(result)

        self.assertEqual(
            {artifact.kind for artifact in report.artifacts},
            {"markdown", "excel", "chart_png", "chart_svg", "pdf"},
        )
        for artifact in report.artifacts:
            self.assertTrue(artifact.path.exists(), artifact.path)

    def test_report_mime_types_are_download_friendly(self) -> None:
        self.assertEqual(_report_mime_type("markdown"), "text/markdown")
        self.assertEqual(
            _report_mime_type("excel"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.assertEqual(_report_mime_type("pdf"), "application/pdf")
        self.assertEqual(_report_mime_type("chart_png"), "image/png")
        self.assertEqual(_report_mime_type("chart_svg"), "image/svg+xml")

    def _temporary_output_dir(self):
        from pathlib import Path
        from tempfile import mkdtemp

        return Path(mkdtemp())


if __name__ == "__main__":
    unittest.main()
