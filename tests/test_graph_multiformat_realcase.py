import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from docx import Document
from reportlab.pdfgen.canvas import Canvas

from qwopus_agent.analysis import analyze_uploaded_file
from qwopus_agent.documents.mineru import MinerUUnavailableError
from qwopus_agent.services.knowledge_graph_service import KnowledgeGraphService
from tests.minirag_fakes import make_test_minirag


class MultiFormatGraphRealCaseTests(unittest.TestCase):
    def test_pdf_docx_txt_xlsx_build_four_hop_path_and_survive_restart(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            files = self._create_real_files(root)
            storage_path = root / "documents.jsonl"
            memory = make_test_minirag(storage_path)

            # 原因：此用例验证格式回退解析和知识链路；MinerU 模型另有独立真实测试。
            # 作用：避免 OCR 推理影响确定性，同时 PDF/DOCX 仍是实际二进制文件。
            with patch(
                "qwopus_agent.documents.parser.parse_document_with_mineru",
                side_effect=MinerUUnavailableError("exercise deterministic fallback"),
            ):
                analyses = {
                    path.name: analyze_uploaded_file(path)
                    for path in files
                }

            self.assertEqual(analyses["ownership.pdf"].metadata["source_type"], "pdf")
            self.assertEqual(analyses["project.docx"].metadata["source_type"], "docx")
            self.assertEqual(analyses["manager.txt"].metadata["source_type"], "text")
            self.assertEqual(analyses["budget.xlsx"].metadata["source_type"], "spreadsheet")

            for file_name, analysis in analyses.items():
                self.assertIsNotNone(analysis.markdown_document)
                memory.insert(f"# File: {file_name}\n\n{analysis.markdown_document}")

            query = "How is Aurora Holdings related to USD 6 million?"
            result = memory.search(query)[0]

            self.assertIn("Aurora Holdings -[owns]-> Blue Harbor Ltd", result)
            self.assertIn("Blue Harbor Ltd -[participates_in]-> Project Lantern", result)
            self.assertIn("Project Lantern -[managed_by]-> Ethan Jiang", result)
            self.assertIn("Ethan Jiang -[approved_budget]-> USD 6 million", result)
            for source in ("ownership.pdf", "project.docx", "manager.txt", "budget.xlsx"):
                self.assertIn(f"Source: {source}", result)

            graph_service = KnowledgeGraphService(root / "documents_graph.json")
            snapshot = graph_service.snapshot(max_nodes=20)
            self.assertEqual(len(snapshot.nodes), 5)
            self.assertEqual(len(snapshot.edges), 4)
            self.assertIn("digraph qwopus_knowledge_graph", graph_service.to_dot(snapshot))

            reloaded = make_test_minirag(storage_path)
            self.assertIn("[Knowledge Graph Path]", reloaded.search(query)[0])

    @staticmethod
    def _create_real_files(root: Path) -> tuple[Path, ...]:
        pdf_path = root / "ownership.pdf"
        pdf = Canvas(str(pdf_path))
        pdf.drawString(
            72,
            750,
            "[[Aurora Holdings|Organization]] -[owns]-> "
            "[[Blue Harbor Ltd|Organization]]",
        )
        pdf.save()

        docx_path = root / "project.docx"
        document = Document()
        document.add_paragraph(
            "[[Blue Harbor Ltd|Organization]] -[participates_in]-> "
            "[[Project Lantern|Project]]"
        )
        document.save(docx_path)

        txt_path = root / "manager.txt"
        txt_path.write_text(
            "[[Project Lantern|Project]] -[managed_by]-> [[Ethan Jiang|Person]]",
            encoding="utf-8",
        )

        xlsx_path = root / "budget.xlsx"
        pd.DataFrame(
            {
                "fact": [
                    "[[Ethan Jiang|Person]] -[approved_budget]-> "
                    "[[USD 6 million|Amount]]"
                ]
            }
        ).to_excel(xlsx_path, index=False)
        return pdf_path, docx_path, txt_path, xlsx_path


if __name__ == "__main__":
    unittest.main()
