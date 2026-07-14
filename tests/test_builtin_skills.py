import asyncio
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qwopus_agent.memory import MiniRAG
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.document_parser import DocumentParserSkill
from qwopus_agent.skills.excel_analysis import ExcelAnalysisSkill
from qwopus_agent.skills.excel_schema import ExcelSchemaSkill
from qwopus_agent.skills.rag_search import RagSearchSkill


class BuiltinSkillTests(unittest.TestCase):
    def test_document_parser_skill_returns_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.md"
            path.write_text("# Notes\n\nProject Alpha", encoding="utf-8")

            response = asyncio.run(
                DocumentParserSkill().run(
                    SkillRequest(query="parse", arguments={"file_path": str(path)})
                )
            )

        self.assertTrue(response.success)
        self.assertIn("Project Alpha", response.content)
        self.assertEqual(response.data["metadata"]["source_type"], "markdown")

    def test_excel_schema_skill_returns_safe_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [10, 20],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelSchemaSkill().run(
                    SkillRequest(query="inspect", arguments={"file_path": str(path)})
                )
            )

        self.assertTrue(response.success)
        sheet = response.data["sheets"]["Sheet1"]
        self.assertEqual(sheet["column_names"], ["region", "revenue"])
        self.assertEqual(len(sheet["sample_rows"]), 2)

    def test_excel_analysis_skill_reuses_local_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [10, 20],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelAnalysisSkill().run(
                    SkillRequest(query="分析收入", arguments={"file_path": str(path)})
                )
            )

        self.assertTrue(response.success)
        self.assertIn("Spreadsheet Analysis", response.content)
        self.assertIn("Sheet1_schema", response.data["table_names"])

    def test_rag_search_skill_uses_injected_minirag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")
            minirag.insert("Project Alpha revenue increased.")

            # 原因：Skill 测试需要隔离持久化知识库，不能污染真实 storage/minirag。
            # 作用：通过依赖注入验证 rag_search 只依赖 MiniRAG.insert/search 公共接口。
            response = asyncio.run(
                RagSearchSkill(minirag=minirag).run(SkillRequest(query="revenue"))
            )

        self.assertTrue(response.success)
        self.assertEqual(response.data["results"], ["Project Alpha revenue increased."])


if __name__ == "__main__":
    unittest.main()
