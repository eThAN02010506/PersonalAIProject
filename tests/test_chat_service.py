import queue
import unittest
from unittest.mock import Mock, patch

from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    SmolagentsModelSettings,
)
from qwopus_agent.services.chat_service import BackgroundChatTask, _run_chat_task
from qwopus_agent.services.orchestration_models import (
    OrchestrationResult,
    ProcessEvent,
    SourceCitation,
)


class ChatServiceTests(unittest.TestCase):
    def test_worker_reports_completed_reply_and_progress(self) -> None:
        result_queue: queue.Queue = queue.Queue()
        progress_queue: queue.Queue = queue.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")

        def fake_orchestrator_run(_self, request, progress_callback=None):
            self.assertTrue(request.enable_local_knowledge)
            self.assertEqual(request.min_source_relevance, 0.8)
            progress_callback("planning")
            progress_callback("completed")
            return OrchestrationResult(
                success=True,
                final_answer="finished reply",
                route="multi_agent",
                trace=(
                    ProcessEvent(
                        phase="tool_call",
                        status="completed",
                        agent="knowledge_agent",
                        tool="graph_search",
                    ),
                ),
                citations=(SourceCitation(kind="local", source="notes.txt"),),
                debug_runs=(
                    AgentDebugRun(
                        label="knowledge",
                        prompt="question",
                        max_steps=2,
                        state="success",
                        output="finished reply",
                        steps=({"observations": "raw local evidence"},),
                    ),
                ),
            )

        with patch(
            "qwopus_agent.services.chat_service.AgentOrchestrator.run_sync",
            new=fake_orchestrator_run,
        ):
            _run_chat_task(
                result_queue,
                progress_queue,
                "question",
                [],
                settings,
                True,
                True,
                0.8,
            )

        payload = result_queue.get_nowait()
        self.assertEqual(payload[:2], ("completed", "finished reply"))
        self.assertEqual(payload[2][0]["tool"], "graph_search")
        self.assertEqual(payload[3][0]["source"], "notes.txt")
        # 原因：spawn 子进程不能把 UI 对象或不可序列化 runtime 对象直接传回主进程。
        # 作用：验证完整调试运行已转换为普通 dict，并保留 Tool Observation。
        self.assertEqual(payload[4][0]["steps"][0]["observations"], "raw local evidence")
        self.assertEqual(progress_queue.get_nowait(), "planning")
        self.assertEqual(progress_queue.get_nowait(), "completed")

    def test_worker_returns_failure_without_raising_into_ui(self) -> None:
        result_queue: queue.Queue = queue.Queue()
        progress_queue: queue.Queue = queue.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")

        with patch(
            "qwopus_agent.services.chat_service.AgentOrchestrator.run_sync",
            side_effect=TimeoutError("model timed out"),
        ):
            _run_chat_task(
                result_queue,
                progress_queue,
                "question",
                [],
                settings,
                False,
            )

        status, content = result_queue.get_nowait()[:2]
        self.assertEqual(status, "failed")
        self.assertIn("model timed out", content)

    def test_cancel_terminates_running_process(self) -> None:
        process = Mock()
        process.is_alive.side_effect = [True, False]
        task = BackgroundChatTask(
            process=process,
            result_queue=queue.Queue(),
            progress_queue=queue.Queue(),
            started_at=0.0,
        )

        # 原因：停止入口必须结束正在等待模型的执行单元，而不只是隐藏加载提示。
        # 作用：验证服务会 terminate 并等待子进程退出，避免后台任务继续占用 UI。
        task.cancel()

        process.terminate.assert_called_once_with()
        process.join.assert_called_once_with(timeout=2)
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
