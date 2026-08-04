import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.documents import (
    DocumentStore,
    build_document_structure,
    chunk_document_structure,
)
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
                captured["response_detail"] = kwargs["response_detail"]
                captured["tool_names"] = [tool.name for tool in tools]
                document_tool = next(tool for tool in tools if tool.name == "document_search")
                captured["document_result"] = document_tool.forward("revenue.txt", "revenue")
                rag_tool = next(tool for tool in tools if tool.name == "rag_search")
                captured["rag_result"] = rag_tool.forward("revenue")
                return DocumentAnalysisRun(
                    answer="Final answer uses prior MiniRAG context.",
                    debug_steps=["fake model finished"],
                    tool_calls=["document_search", "rag_search"],
                    inspected_file_names=("revenue.txt",),
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
                    response_detail="concise",
                )

            self.assertEqual(
                outcome.result.llm_analysis, "Final answer uses prior MiniRAG context."
            )
            self.assertEqual(outcome.result.metadata["minirag_search_hits"], 1)
            self.assertEqual(captured["response_detail"], "concise")
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
            self.assertEqual(outcome.result.metadata["generation_mode"], "model")
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
                    inspected_file_names=("sales.xlsx",),
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
                [
                    "excel_schema",
                    "excel_statistics",
                    "excel_modeling",
                    "excel_analysis",
                    "rag_search",
                ],
            )
            self.assertIn("Sample: East 10", captured["schema"])
            self.assertIn("40", captured["analysis"])
            self.assertTrue(outcome.result.metadata["pandas_sandbox_used"])
            self.assertEqual(outcome.result.llm_analysis, "East revenue is 40; West revenue is 20.")

    def test_repeated_upload_reuses_cached_parse_and_minirag_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "cached.xlsx"
            source_path.write_bytes(b"same spreadsheet bytes")
            minirag = make_test_minirag(root / "documents.jsonl")
            document_store = DocumentStore(root / "documents")
            settings = SmolagentsModelSettings(
                model_id="test-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            local_result = AnalysisResult(
                markdown_summary="# Spreadsheet Analysis\n\nCached schema.",
                metadata={"source_type": "spreadsheet"},
                markdown_document="# Spreadsheet Analysis\n\nCached schema.",
            )

            with (
                patch(
                    "qwopus_agent.services.analysis_service.save_uploaded_bytes",
                    return_value=SimpleNamespace(
                        original_name="cached.xlsx",
                        path=source_path,
                    ),
                ) as save_file,
                patch(
                    "qwopus_agent.services.analysis_service.analyze_uploaded_file",
                    return_value=local_result,
                ) as analyze_file,
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection",
                    return_value=(False, "offline"),
                ),
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
            ):
                first = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(
                            name="cached.xlsx",
                            content=source_path.read_bytes(),
                        )
                    ],
                    user_question="summary",
                    settings=settings,
                    minirag=minirag,
                    document_store=document_store,
                )
                second = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(
                            name="cached.xlsx",
                            content=source_path.read_bytes(),
                        )
                    ],
                    user_question="summary",
                    settings=settings,
                    minirag=minirag,
                    document_store=document_store,
                )

            # 原因：相同上传内容已经有持久化解析产物时，不应再次跑昂贵的解析和图谱入库。
            # 作用：锁定第二次请求命中缓存，MiniRAG 仍只有一个稳定 document_id。
            self.assertEqual(save_file.call_count, 1)
            self.assertEqual(analyze_file.call_count, 1)
            self.assertTrue(
                any("命中上传缓存" in step for step in second.debug_steps)
            )
            self.assertFalse(first.result.metadata["files"][0]["metadata"]["cache_hit"])
            self.assertTrue(second.result.metadata["files"][0]["metadata"]["cache_hit"])
            records = (root / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)

    def test_local_folder_files_are_analyzed_without_upload_or_minirag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.md"
            second = root / "second.md"
            first.write_text("# Alpha\nAlpha launch evidence.", encoding="utf-8")
            second.write_text("# Beta\nBeta budget evidence.", encoding="utf-8")
            settings = SmolagentsModelSettings(
                model_id="test-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            captured: dict[str, object] = {}

            def fake_agent(**kwargs):
                tools = {tool.name: tool for tool in kwargs["tools"]}
                captured["tool_names"] = list(tools)
                captured["collection"] = tools["document_collection_summary"].forward()
                captured["search"] = tools["document_search"].forward(
                    "first.md",
                    "Alpha launch",
                )
                return DocumentAnalysisRun(
                    answer="Alpha covers launch; Beta covers budget.",
                    debug_steps=["direct analysis completed"],
                    tool_calls=["document_collection_summary", "document_search"],
                    inspected_file_names=("first.md", "second.md"),
                )

            with (
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection",
                    return_value=(True, "online"),
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_file_analysis_with_debug",
                    side_effect=fake_agent,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.save_uploaded_bytes"
                ) as save_upload,
                patch("qwopus_agent.services.analysis_service.DocumentStore") as document_store,
            ):
                outcome = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(name="first.md", local_path=first),
                        UploadedFileInput(name="second.md", local_path=second),
                    ],
                    user_question="Compare both files",
                    settings=settings,
                    minirag=None,
                    analysis_mode="full",
                )

            # 原因：目录模式的核心契约是“分析原文件”，不能悄悄退回上传或向量入库路径。
            # 作用：证明两份文件由本地 Tool 覆盖，且 storage 上传/文档持久化均未触发。
            save_upload.assert_not_called()
            document_store.assert_not_called()
            self.assertFalse(outcome.result.metadata["minirag_inserted"])
            self.assertFalse(outcome.result.metadata["minirag_context_used"])
            self.assertEqual(outcome.result.metadata["analysis_source"], "local_folder")
            self.assertNotIn("rag_search", captured["tool_names"])
            self.assertIn("# File: first.md", captured["collection"])
            self.assertIn("# File: second.md", captured["collection"])
            self.assertIn("Alpha launch evidence", captured["search"])

    def test_offline_model_still_runs_grounded_report_composer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lesson_21 = root / "lesson-21.md"
            lesson_22 = root / "lesson-22.md"
            lesson_21.write_text(
                "# Lesson 21\n\nTitle: Active humility\n\n"
                "The source distinguishes humility from self-erasure.",
                encoding="utf-8",
            )
            lesson_22.write_text(
                "# Lesson 22\n\nTitle: Free obedience\n\n"
                "The source distinguishes trust from blind compliance.",
                encoding="utf-8",
            )
            settings = SmolagentsModelSettings(
                model_id="offline-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            called: dict[str, object] = {}

            def fake_composer(**kwargs):
                called["file_names"] = kwargs["file_names"]
                return DocumentAnalysisRun(
                    answer="Grounded report composed while the model was offline.",
                    debug_steps=["grounded composer finished"],
                    tool_calls=["document_collection_summary"],
                    inspected_file_names=("lesson-21.md", "lesson-22.md"),
                    generation_mode="grounded_composer",
                )

            question = (
                "请逐一阅读所有文件并完整输出：\n"
                "## 1. 文档理解\n"
                "## 2. 写作整体策略\n"
                "## 3. 详细写作框架\n"
                "## 4. 逐段写作指导\n"
                "## 5. 具体例子\n"
                "## 6. 生成完整报告 Draft"
            )
            with (
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection",
                    return_value=(False, "offline"),
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_file_analysis_with_debug",
                    side_effect=fake_composer,
                ),
            ):
                outcome = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(name="lesson-21.md", local_path=lesson_21),
                        UploadedFileInput(name="lesson-22.md", local_path=lesson_22),
                    ],
                    user_question=question,
                    settings=settings,
                    minirag=None,
                    analysis_mode="full",
                )

            self.assertEqual(
                called["file_names"],
                ["lesson-21.md", "lesson-22.md"],
            )
            self.assertEqual(
                outcome.result.metadata["generation_mode"],
                "grounded_composer",
            )
            self.assertIn("model was offline", outcome.result.llm_analysis)
            self.assertTrue(
                any("本地证据合成" in step for step in outcome.debug_steps)
            )

    def test_analyze_uploaded_files_propagates_recipe_to_runner_and_collection_tool(self) -> None:
        """Selecting a recipe at the service entry reaches both the runner and tool builder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lesson_21 = root / "lesson-21.md"
            lesson_22 = root / "lesson-22.md"
            lesson_21.write_text(
                "# Lesson 21\n\nTitle: Active humility\n\n"
                "The source distinguishes humility from self-erasure.",
                encoding="utf-8",
            )
            lesson_22.write_text(
                "# Lesson 22\n\nTitle: Free obedience\n\n"
                "The source distinguishes trust from blind compliance.",
                encoding="utf-8",
            )
            settings = SmolagentsModelSettings(
                model_id="online-model",
                base_url="http://127.0.0.1:9999/v1",
            )
            captured: dict[str, object] = {}

            def fake_runner(**kwargs):
                captured["runner_recipe"] = kwargs.get("recipe")
                return DocumentAnalysisRun(
                    answer="Recipe-aware analysis completed.",
                    debug_steps=["analysis finished"],
                    tool_calls=[],
                    inspected_file_names=("lesson-21.md", "lesson-22.md"),
                )

            from qwopus_agent.reports.bible_recipe import BIBLE_RECIPE

            with (
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.check_model_connection",
                    return_value=(True, "online"),
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_file_analysis_with_debug",
                    side_effect=fake_runner,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.build_document_collection_summary_tool"
                ) as build_collection_tool,
            ):
                outcome = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(name="lesson-21.md", local_path=lesson_21),
                        UploadedFileInput(name="lesson-22.md", local_path=lesson_22),
                    ],
                    user_question="Compare both files",
                    settings=settings,
                    minirag=None,
                    analysis_mode="full",
                    recipe=BIBLE_RECIPE,
                )

            self.assertEqual(outcome.result.llm_analysis, "Recipe-aware analysis completed.")
            self.assertIs(captured["runner_recipe"], BIBLE_RECIPE)
            _, tool_kwargs = build_collection_tool.call_args
            self.assertIs(tool_kwargs.get("recipe"), BIBLE_RECIPE)


if __name__ == "__main__":
    unittest.main()
