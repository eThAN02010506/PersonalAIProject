import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.documents import build_document_structure, chunk_document_structure
from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    DocumentAnalysisRun,
    SmolagentsModelSettings,
)
from qwopus_agent.services.analysis_service import (
    UploadedFileInput,
    _scope_sections_by_file,
    analyze_uploaded_files,
)
from tests.minirag_fakes import make_test_minirag


class AnalysisServiceTests(unittest.TestCase):
    def test_section_scope_accepts_document_id_and_includes_descendants(self) -> None:
        structure = chunk_document_structure(
            build_document_structure(
                "# Parent\nIntro\n## Child\nDetail\n# Other\nIgnore",
                source="manual.md",
            )
        )
        parent = structure.sections[0]

        # 原因：前端提交 document_id，而 Tool Registry 以文件名保存当前文档。
        # 作用：锁定 id 映射和父章节后代展开，防止章节模式遗漏子标题内容。
        scope = _scope_sections_by_file(
            {"manual.md": structure},
            {structure.document_id: (parent.id,)},
        )

        self.assertEqual(len(scope["manual.md"]), 2)
        self.assertNotIn(structure.sections[-1].id, scope["manual.md"])

        with self.assertRaisesRegex(ValueError, "Unknown section selection"):
            _scope_sections_by_file(
                {"manual.md": structure},
                {structure.document_id: ("stale-section-id",)},
            )

    def test_section_mode_rejects_an_empty_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            local_result = AnalysisResult(
                markdown_summary="# Local Summary",
                metadata={"source_type": "text"},
                markdown_document="# Notes\nScoped content",
            )
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")

            with (
                patch(
                    "qwopus_agent.services.analysis_service.save_uploaded_bytes",
                    return_value=SimpleNamespace(original_name=path.name, path=path),
                ),
                patch(
                    "qwopus_agent.services.analysis_service.analyze_uploaded_file",
                    return_value=local_result,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                self.assertRaisesRegex(ValueError, "requires at least one"),
            ):
                # 原因：章节模式没有选择时，空 allow-list 在 Tool 层等同于允许全文。
                # 作用：锁定服务边界必须拒绝请求，不能静默扩大读取范围。
                analyze_uploaded_files(
                    uploaded_files=[UploadedFileInput(name=path.name, content=b"notes")],
                    user_question="Summarize",
                    settings=SmolagentsModelSettings(
                        model_id="test-model",
                        base_url="http://127.0.0.1:9999/v1",
                    ),
                    minirag=minirag,
                    analysis_mode="section",
                )

    def test_analyze_uploaded_files_runs_without_streamlit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")
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
                minirag.search("local markdown"),
                ["[Source: notes.txt]\nlocal markdown content"],
            )
            run_llm.assert_not_called()

    def test_analyze_uploaded_files_uses_existing_minirag_context_before_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")
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
                document_tool = next(tool for tool in tools if tool.name == "document_search")
                captured["document_result"] = document_tool.forward("revenue.txt", "revenue")
                rag_tool = next(tool for tool in tools if tool.name == "rag_search")
                captured["rag_result"] = rag_tool.forward("revenue")
                return DocumentAnalysisRun(
                    answer="Final answer uses prior MiniRAG context.",
                    debug_steps=["fake model finished"],
                    tool_calls=["document_search", "rag_search"],
                    debug_runs=(
                        AgentDebugRun(
                            label="file_analysis",
                            prompt="analyze revenue.txt",
                            max_steps=4,
                            state="success",
                            output="Final answer uses prior MiniRAG context.",
                            steps=({"observations": "raw document text"},),
                        ),
                    ),
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
                    min_source_relevance=0.25,
                )

            self.assertEqual(
                outcome.result.llm_analysis, "Final answer uses prior MiniRAG context."
            )
            self.assertEqual(outcome.result.metadata["minirag_search_hits"], 1)
            self.assertTrue(outcome.result.metadata["minirag_inserted"])
            self.assertEqual(
                captured["tool_names"],
                [
                    "document_outline",
                    "document_search",
                    "document_read_section",
                    "document_summary",
                    "rag_search",
                ],
            )
            self.assertIn("Current uploaded note", captured["document_result"])
            self.assertIn("Prior MiniRAG note", captured["rag_result"])
            self.assertTrue(outcome.result.metadata["minirag_context_used"])
            self.assertEqual(
                outcome.debug_runs[0].steps[0]["observations"],
                "raw document text",
            )
            self.assertTrue(
                minirag.search("Current uploaded")[0].startswith("[Source: revenue.txt]")
            )

    def test_excel_upload_injects_schema_and_sandbox_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame({"region": ["East", "West", "East"], "revenue": [10, 20, 30]}).to_excel(
                path, index=False
            )
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")
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
