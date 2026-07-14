import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import DocumentAnalysisRun, SmolagentsModelSettings
from qwopus_agent.memory import MiniRAG
from qwopus_agent.services.analysis_service import (
    UploadedFileInput,
    _analyze_file_with_agent,
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
                patch("qwopus_agent.services.analysis_service._analyze_file_with_agent") as analyze_file,
                patch("qwopus_agent.services.analysis_service.check_model_connection") as check_connection,
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch("qwopus_agent.services.analysis_service.run_smolagents_document_analysis_with_debug") as run_llm,
            ):
                # 原因：service 层测试只验证业务编排，不依赖真实上传目录和模型服务。
                # 作用：证明文件分析流程已经脱离 Streamlit，可被 CLI/API 复用。
                save_file.return_value = SimpleNamespace(
                    original_name="notes.txt",
                    path=Path(tmpdir) / "notes.txt",
                )
                analyze_file.return_value = SimpleNamespace(
                    result=local_result,
                    plan_steps=["document_parser"],
                )
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
            self.assertTrue(any("Agent 计划执行" in step for step in outcome.debug_steps))
            self.assertTrue(any("模型未连接" in step for step in outcome.debug_steps))
            self.assertEqual(minirag.search("local markdown"), ["# File: notes.txt\n\nlocal markdown content"])
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
            captured: dict[str, str] = {}

            def fake_llm(**kwargs):
                captured["content"] = kwargs["content"]
                return DocumentAnalysisRun(
                    answer="Final answer uses prior MiniRAG context.",
                    debug_steps=["fake model finished"],
                )

            with (
                patch("qwopus_agent.services.analysis_service.save_uploaded_bytes") as save_file,
                patch("qwopus_agent.services.analysis_service._analyze_file_with_agent") as analyze_file,
                patch("qwopus_agent.services.analysis_service.check_model_connection") as check_connection,
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.run_smolagents_document_analysis_with_debug",
                    side_effect=fake_llm,
                ),
            ):
                save_file.return_value = SimpleNamespace(
                    original_name="revenue.txt",
                    path=Path(tmpdir) / "revenue.txt",
                )
                analyze_file.return_value = SimpleNamespace(
                    result=local_result,
                    plan_steps=["document_parser"],
                )
                check_connection.return_value = (True, "online")

                outcome = analyze_uploaded_files(
                    uploaded_files=[UploadedFileInput(name="revenue.txt", content=b"current")],
                    user_question="revenue",
                    settings=settings,
                    minirag=minirag,
                )

            self.assertEqual(outcome.result.llm_analysis, "Final answer uses prior MiniRAG context.")
            self.assertEqual(outcome.result.metadata["minirag_search_hits"], 1)
            self.assertTrue(outcome.result.metadata["minirag_inserted"])
            self.assertIn("MiniRAG Search Context", captured["content"])
            self.assertIn("Prior MiniRAG note", captured["content"])
            self.assertEqual(len(minirag.search("Current uploaded")), 1)

    def test_analyze_file_with_agent_routes_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("Qwopus Agent uses Planner and Executor.", encoding="utf-8")

            # 原因：服务层不能再绕过 AgentRouter 直接调用底层解析函数。
            # 作用：用真实 txt 文件证明 Planner/Executor/DocumentParserSkill 链路可独立运行。
            routed_analysis = _analyze_file_with_agent(path, user_question="总结")

            self.assertEqual(routed_analysis.plan_steps, ["document_parser"])
            self.assertIn("Qwopus Agent", routed_analysis.result.markdown_document)
            self.assertIn("Qwopus Agent", routed_analysis.result.markdown_summary)


if __name__ == "__main__":
    unittest.main()
