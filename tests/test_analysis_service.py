import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import (
    DocumentAnalysisRun,
    SmolagentsModelSettings,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.services.analysis_service import (
    UploadedFileInput,
    analyze_uploaded_files,
)


class AnalysisServiceTests(unittest.TestCase):
    def test_analyze_uploaded_files_runs_without_streamlit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")
            settings = SmolagentsModelSettings(
                model_id="test-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            local_result = AnalysisResult(
                markdown_summary="# Local Summary",
                tables={"metadata": pd.DataFrame([{"key": "type", "value": "txt"}])},
                metadata={"source_type": "text"},
                markdown_document="local markdown content",
            )

            with (
                patch("qwopus_agent.services.analysis_service.save_uploaded_bytes") as save_file,
                patch(
                    "qwopus_agent.services.analysis_service.analyze_uploaded_file"
                ) as analyze_file,
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection"
                ) as check_connection,
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_file_analysis_with_debug"
                ) as run_llm,
            ):
                # 原因：service 层测试只验证业务编排，不依赖真实上传目录和模型服务。
                # 作用：证明文件分析流程已经脱离 Streamlit，可被 CLI/API 复用。
                save_file.return_value = SimpleNamespace(
                    original_name="notes.txt",
                    path=Path(tmpdir) / "notes.txt",
                )
                analyze_file.return_value = local_result
                check_connection.return_value = (False, "offline")

                outcome = analyze_uploaded_files(
                    uploaded_files=[UploadedFileInput(name="notes.txt", content=b"hello")],
                    user_question="总结",
                    settings=settings,
                    minirag=minirag,
                )

            self.assertEqual(outcome.analyzed_file_names, ["notes.txt"])
            self.assertIn("notes.txt", outcome.result.markdown_summary)
            self.assertIn("local markdown content", outcome.result.markdown_document)
            self.assertTrue(outcome.result.metadata["minirag_inserted"])
            self.assertEqual(outcome.result.metadata["minirag_search_hits"], 0)
            self.assertTrue(any("本地预处理完成" in step for step in outcome.debug_steps))
            self.assertTrue(any("模型未连接" in step for step in outcome.debug_steps))
            self.assertEqual(
                minirag.search("local markdown"), ["# File: notes.txt\n\nlocal markdown content"]
            )
            run_llm.assert_not_called()

    def test_analyze_uploaded_files_uses_existing_minirag_context_before_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")
            minirag.insert("Prior MiniRAG note about revenue growth.")
            settings = SmolagentsModelSettings(
                model_id="test-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            local_result = AnalysisResult(
                markdown_summary="# Current Summary",
                metadata={"source_type": "text"},
                markdown_document="Current uploaded note about revenue.",
            )
            captured: dict[str, object] = {}

            def fake_llm(**kwargs):
                tools = kwargs["tools"]
                captured["tool_names"] = [tool.name for tool in tools]
                rag_tool = next(tool for tool in tools if tool.name == "rag_search")
                captured["rag_result"] = rag_tool.forward("revenue")
                return DocumentAnalysisRun(
                    answer="Final answer uses prior MiniRAG context.",
                    debug_steps=["fake model finished"],
                    tool_calls=["document_parser", "rag_search"],
                )

            with (
                patch("qwopus_agent.services.analysis_service.save_uploaded_bytes") as save_file,
                patch(
                    "qwopus_agent.services.analysis_service.analyze_uploaded_file"
                ) as analyze_file,
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection"
                ) as check_connection,
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_file_analysis_with_debug",
                    side_effect=fake_llm,
                ),
            ):
                save_file.return_value = SimpleNamespace(
                    original_name="revenue.txt",
                    path=Path(tmpdir) / "revenue.txt",
                )
                analyze_file.return_value = local_result
                check_connection.return_value = (True, "online")

                outcome = analyze_uploaded_files(
                    uploaded_files=[UploadedFileInput(name="revenue.txt", content=b"current")],
                    user_question="revenue",
                    settings=settings,
                    minirag=minirag,
                )

            self.assertEqual(
                outcome.result.llm_analysis, "Final answer uses prior MiniRAG context."
            )
            self.assertEqual(outcome.result.metadata["minirag_search_hits"], 1)
            self.assertTrue(outcome.result.metadata["minirag_inserted"])
            self.assertEqual(captured["tool_names"], ["document_parser", "rag_search"])
            self.assertIn("Prior MiniRAG note", captured["rag_result"])
            self.assertTrue(outcome.result.metadata["minirag_context_used"])
            self.assertTrue(minirag.search("Current uploaded")[0].startswith("# File: revenue.txt"))

    def test_excel_upload_injects_schema_and_sandbox_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame({"region": ["East", "West", "East"], "revenue": [10, 20, 30]}).to_excel(
                path, index=False
            )
            minirag = MiniRAG(storage_path=Path(tmpdir) / "documents.jsonl")
            settings = SmolagentsModelSettings(
                model_id="test-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            local_result = AnalysisResult(
                markdown_summary="# Spreadsheet Analysis",
                tables={},
                metadata={"source_type": "spreadsheet"},
                markdown_document=(
                    "# Spreadsheet Analysis\n\n"
                    "Sheet1 columns: region (object), revenue (int64).\n"
                    "Sample: East 10, West 20."
                ),
            )
            captured: dict[str, object] = {}

            def fake_agent(**kwargs):
                tools = {tool.name: tool for tool in kwargs["tools"]}
                captured["tool_names"] = list(tools)
                captured["schema"] = tools["excel_schema"].forward("sales.xlsx")
                captured["analysis"] = tools["excel_analysis"].forward(
                    "sales.xlsx",
                    (
                        'df = dfs["Sheet1"]\n'
                        'result = df.groupby("region")["revenue"].sum().reset_index()'
                    ),
                )
                return DocumentAnalysisRun(
                    answer="East revenue is 40; West revenue is 20.",
                    debug_steps=["fake Agent finished"],
                    tool_calls=["excel_schema", "excel_analysis"],
                )

            with (
                patch("qwopus_agent.services.analysis_service.save_uploaded_bytes") as save_file,
                patch(
                    "qwopus_agent.services.analysis_service.analyze_uploaded_file"
                ) as analyze_file,
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection",
                    return_value=(True, "online"),
                ),
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_file_analysis_with_debug",
                    side_effect=fake_agent,
                ),
            ):
                save_file.return_value = SimpleNamespace(original_name="sales.xlsx", path=path)
                analyze_file.return_value = local_result
                outcome = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(name="sales.xlsx", content=path.read_bytes())
                    ],
                    user_question="按地区汇总收入",
                    settings=settings,
                    minirag=minirag,
                )

            # 原因：Excel 的 schema 检查和 pandas 计算现在都由 smolagents Tool 调度。
            # 作用：证明服务层注入正确 Tool，且沙箱只把本地聚合结果交回 Agent。
            self.assertEqual(
                captured["tool_names"],
                ["excel_schema", "excel_analysis", "rag_search"],
            )
            self.assertIn("Sample: East 10", captured["schema"])
            self.assertIn("40", captured["analysis"])
            self.assertTrue(outcome.result.metadata["pandas_sandbox_used"])
            self.assertEqual(outcome.result.llm_analysis, "East revenue is 40; West revenue is 20.")


if __name__ == "__main__":
    unittest.main()
