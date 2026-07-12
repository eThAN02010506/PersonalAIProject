import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qwopus_agent.analysis import analyze_uploaded_file
from qwopus_agent.documents import parse_document, save_uploaded_bytes


class DocumentAnalysisTests(unittest.TestCase):
    def test_parse_text_document_to_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("Hello\nWorld", encoding="utf-8")

            parsed = parse_document(path)

            self.assertEqual(parsed.markdown, "Hello\nWorld")
            self.assertEqual(parsed.metadata["source_type"], "text")

    def test_analyze_csv_returns_schema_and_sample_without_full_llm_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.csv"
            pd.DataFrame(
                {
                    "region": ["East", "West", "East"],
                    "revenue": [10, 20, 30],
                }
            ).to_csv(path, index=False)

            result = analyze_uploaded_file(path)

            self.assertIn("Spreadsheet Analysis", result.markdown_summary)
            self.assertIn("csv_schema", result.tables)
            self.assertIn("csv_sample", result.tables)
            self.assertEqual(result.metadata["sheets"]["csv"]["rows"], 3)
            self.assertIn("csv_schema", result.markdown_document)

    def test_analysis_result_can_carry_llm_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("Project Alpha increased revenue.", encoding="utf-8")

            result = analyze_uploaded_file(path, user_question="总结")

            enriched = type(result)(
                markdown_summary=result.markdown_summary,
                tables=result.tables,
                metadata=result.metadata,
                markdown_document=result.markdown_document,
                llm_analysis="这是一份关于 Project Alpha 收入增长的文档。",
            )

            self.assertEqual(enriched.llm_analysis, "这是一份关于 Project Alpha 收入增长的文档。")

    def test_save_uploaded_bytes_uses_upload_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stored = save_uploaded_bytes(
                filename="../unsafe.txt",
                content=b"data",
                upload_dir=Path(tmpdir),
            )

            self.assertEqual(stored.original_name, "unsafe.txt")
            self.assertTrue(stored.path.exists())
            self.assertEqual(stored.path.read_bytes(), b"data")
