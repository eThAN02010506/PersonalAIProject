import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from qwopus_agent.reports import ReportGenerator


class ReportGeneratorTests(unittest.TestCase):
    def test_report_generator_creates_real_png_and_svg_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=Path(tmpdir))
            tables = {
                "Quarterly Sales": pd.DataFrame(
                    {
                        "quarter": ["Q1", "Q2", "Q3"],
                        "revenue": [120, 160, 190],
                        "cost": [80, 95, 110],
                    }
                )
            }

            # 原因：检查文件存在无法区分真实图像与旧 chart manifest。
            # 作用：同时验证统一入口、PNG 签名和 SVG 根元素。
            report = generator.generate(
                title="Analysis Report",
                markdown_body="Final answer body.",
                tables=tables,
                basename="analysis report",
            )

            artifacts = {artifact.kind: artifact.path for artifact in report.artifacts}
            self.assertEqual(
                set(artifacts),
                {"markdown", "excel", "chart_png", "chart_svg", "pdf"},
            )
            self.assertIn("Final answer body.", report.markdown.read_text(encoding="utf-8"))
            self.assertTrue(artifacts["chart_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(artifacts["chart_png"].stat().st_size, 5_000)
            svg = artifacts["chart_svg"].read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertGreater(len(svg), 1_000)

    def test_non_numeric_tables_do_not_create_empty_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportGenerator(output_dir=Path(tmpdir)).generate(
                title="Text Report",
                markdown_body="Body",
                tables={"notes": pd.DataFrame({"label": ["a", "b"]})},
            )

            kinds = {artifact.kind for artifact in report.artifacts}
            self.assertEqual(kinds, {"markdown", "excel", "pdf"})

    def test_multiple_tables_receive_distinct_chart_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportGenerator(output_dir=Path(tmpdir)).generate(
                title="Multiple",
                markdown_body="Body",
                tables={
                    "same/name": pd.DataFrame({"label": ["a"], "value": [1]}),
                    "same:name": pd.DataFrame({"label": ["b"], "value": [2]}),
                },
                basename="multi",
            )

            chart_paths = [
                artifact.path
                for artifact in report.artifacts
                if artifact.kind in {"chart_png", "chart_svg"}
            ]
            self.assertEqual(len(chart_paths), 4)
            self.assertEqual(len(set(chart_paths)), 4)

    def test_pdf_preserves_unicode_and_complete_multi_page_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_marker = "中文报告开头：完整内容必须可搜索。"
            final_marker = "中文报告结尾：旧的 3500 字符截断不得再次出现。"
            paragraphs = ["这是用于分页验证的长段落。" * 30] * 30
            body = "\n\n".join([first_marker, *paragraphs, final_marker])

            report = ReportGenerator(output_dir=Path(tmpdir)).generate(
                title="多语言分析报告",
                markdown_body=body,
                basename="unicode",
            )
            pdf_path = next(
                artifact.path for artifact in report.artifacts if artifact.kind == "pdf"
            )
            reader = PdfReader(pdf_path)
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

            # 原因：文件签名测试无法发现中文变成问号或正文被静默截断。
            # 作用：同时锁定 Unicode 文本层、自动分页和末尾内容完整性。
            self.assertGreater(len(reader.pages), 1)
            self.assertIn(first_marker, extracted)
            self.assertIn(final_marker, extracted)


if __name__ == "__main__":
    unittest.main()
