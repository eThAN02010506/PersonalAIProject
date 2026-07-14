import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.memory import MiniRAG
from qwopus_agent.services.analysis_service import UploadedFileInput, analyze_uploaded_files


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
                patch("qwopus_agent.services.analysis_service.analyze_uploaded_file") as analyze_file,
                patch("qwopus_agent.services.analysis_service.check_model_connection") as check_connection,
                patch("qwopus_agent.services.analysis_service.run_smolagents_document_analysis_with_debug") as run_llm,
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
            self.assertTrue(any("模型未连接" in step for step in outcome.debug_steps))
            self.assertEqual(minirag.search("local markdown"), ["# File: notes.txt\n\nlocal markdown content"])
            run_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
