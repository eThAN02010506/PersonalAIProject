import queue
import unittest
from unittest.mock import Mock, patch

from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.services.chat_service import BackgroundChatTask, _run_chat_task


class ChatServiceTests(unittest.TestCase):
    def test_worker_reports_completed_reply_and_progress(self) -> None:
        result_queue: queue.Queue = queue.Queue()
        progress_queue: queue.Queue = queue.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")

        def fake_run_agent_chat_turn(**kwargs):
            kwargs["progress_callback"]("planning")
            kwargs["progress_callback"]("completed")
            return "finished reply"

        with patch(
            "qwopus_agent.services.chat_service.run_agent_chat_turn",
            side_effect=fake_run_agent_chat_turn,
        ):
            _run_chat_task(
                result_queue,
                progress_queue,
                "question",
                [],
                settings,
                True,
            )

        self.assertEqual(result_queue.get_nowait(), ("completed", "finished reply"))
        self.assertEqual(progress_queue.get_nowait(), "planning")
        self.assertEqual(progress_queue.get_nowait(), "completed")

    def test_worker_returns_failure_without_raising_into_ui(self) -> None:
        result_queue: queue.Queue = queue.Queue()
        progress_queue: queue.Queue = queue.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")

        with patch(
            "qwopus_agent.services.chat_service.run_agent_chat_turn",
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

        status, content = result_queue.get_nowait()
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
