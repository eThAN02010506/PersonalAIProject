import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from qwopus_agent.api.app import create_app
from qwopus_agent.api.model_runtime import RuntimeModelStatus
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.integrations.smolagents_runtime import AgentDebugRun, SmolagentsModelSettings
from qwopus_agent.services.orchestration_models import OrchestrationResult
from qwopus_agent.utils.debug_store import load_debug_records


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "qwopus.db"
        self.debug_directory = Path(self.temp_directory.name) / "debug_runs"
        # 原因：API 测试必须验证 SQLite 持久化，但不能读取或修改用户现有对话。
        # 作用：每个测试使用独立数据库，并关闭旧 JSONL 自动导入。
        self.repository = ConversationRepository(database_path, import_legacy=False)
        # 原因：API 边界单元测试不应加载或写入用户真实的 storage/minirag 索引。
        # 作用：注入隔离替身；真实 MinerU/MiniRAG 链路由端到端用例单独验证。
        self.model_settings = SmolagentsModelSettings(
            model_id="test-model",
            base_url="http://127.0.0.1:9999/v1",
        )
        self.model_status = RuntimeModelStatus(
            mode="remote",
            settings=self.model_settings,
            online=True,
            message="online",
        )
        self.model_runtime = MagicMock()
        self.model_runtime.current_settings.return_value = self.model_settings
        self.model_runtime.status.return_value = self.model_status
        self.model_runtime.configure_remote.return_value = self.model_status
        self.model_runtime.configure_local.return_value = self.model_status
        self.client_context = TestClient(
            create_app(
                self.repository,
                minirag=MagicMock(),
                model_runtime=self.model_runtime,
                debug_directory=self.debug_directory,
            )
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temp_directory.cleanup()

    def test_conversation_crud_and_message_isolation(self) -> None:
        first = self.client.post("/api/conversations", json={"title": "First"})
        second = self.client.post("/api/conversations", json={"title": "Second"})

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        first_id = first.json()["id"]
        second_id = second.json()["id"]
        self.repository.add_message(first_id, "user", "first question")

        first_messages = self.client.get(f"/api/conversations/{first_id}/messages")
        second_messages = self.client.get(f"/api/conversations/{second_id}/messages")
        self.assertEqual(first_messages.json()[0]["content"], "first question")
        self.assertEqual(second_messages.json(), [])

        renamed = self.client.patch(
            f"/api/conversations/{first_id}",
            json={"title": "Renamed"},
        )
        self.assertEqual(renamed.json()["title"], "Renamed")
        self.assertEqual(self.client.delete(f"/api/conversations/{first_id}").status_code, 204)
        self.assertEqual(
            self.client.get(f"/api/conversations/{first_id}/messages").status_code,
            404,
        )

    def test_openapi_exposes_agent_and_document_boundaries(self) -> None:
        schema = self.client.get("/openapi.json").json()

        self.assertIn("/api/conversations/{conversation_id}/runs", schema["paths"])
        self.assertIn("/api/runs/{run_id}", schema["paths"])
        self.assertIn("/api/analysis", schema["paths"])
        self.assertIn("/api/model-settings", schema["paths"])
        self.assertIn("/api/reports/{filename}", schema["paths"])

    def test_model_settings_switches_remote_endpoint_without_env_changes(self) -> None:
        response = self.client.put(
            "/api/model-settings",
            json={"mode": "remote", "base_url": "http://192.168.1.97:8001/v1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["base_url"], self.model_settings.base_url)
        self.model_runtime.configure_remote.assert_called_once_with(
            "http://192.168.1.97:8001/v1"
        )

    def test_analysis_reads_uploaded_bytes_before_orchestration(self) -> None:
        orchestrator = MagicMock()
        orchestrator.run_sync.return_value = OrchestrationResult(
            success=True,
            final_answer="Analyzed",
            route="single_agent",
            debug_runs=(
                AgentDebugRun(
                    label="document",
                    prompt="Summarize",
                    max_steps=2,
                    state="success",
                    output="Analyzed",
                    steps=({"observations": "raw document"},),
                ),
            ),
        )

        # 原因：该测试只验证 FastAPI 上传边界，真实 MinerU/MiniRAG 由端到端测试负责。
        # 作用：防止 await 生成式再次把 UploadFile 变成不可迭代的 async_generator。
        with patch(
            "qwopus_agent.api.app.AgentOrchestrator",
            return_value=orchestrator,
        ):
            response = self.client.post(
                "/api/analysis",
                files={"files": ("sample.txt", b"real bytes", "text/plain")},
                data={"question": "Summarize"},
            )

        self.assertEqual(response.status_code, 200)
        # 原因：raw debug 可能包含完整文件正文，只允许本地 Streamlit 调试台读取。
        # 作用：锁定 FastAPI 正式响应不会把内部 debug_runs 暴露给 React 或其他客户端。
        self.assertNotIn("debug_runs", response.json())
        # 原因：正式响应必须脱敏，但独立 Console 仍要接收同一次文档运行的原始记录。
        # 作用：验证 API 边界同时满足“不外泄”和“可调试”两个方向。
        debug_records = load_debug_records(directory=self.debug_directory)
        self.assertEqual(debug_records[0]["source"], "document")
        self.assertEqual(
            debug_records[0]["debug_runs"][0]["steps"][0]["observations"],
            "raw document",
        )
        request = orchestrator.run_sync.call_args.args[0]
        self.assertEqual(request.uploaded_files[0].name, "sample.txt")
        self.assertEqual(request.uploaded_files[0].content, b"real bytes")


if __name__ == "__main__":
    unittest.main()
