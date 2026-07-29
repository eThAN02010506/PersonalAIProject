import multiprocessing
import queue
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    SmolagentsModelSettings,
)
from qwopus_agent.services.chat_service import (
    CHAT_WORKER_REQUEST_SCHEMA_VERSION,
    BackgroundChatTask,
    ChatWorkerRequest,
    _run_chat_task,
    start_chat_task,
)
from qwopus_agent.services.orchestration_models import (
    OrchestrationRequest,
    OrchestrationResult,
    ProcessEvent,
    SourceCitation,
)
from qwopus_agent.skills import WorkflowSpec


class ChatServiceTests(unittest.TestCase):
    def _request(
        self,
        settings: SmolagentsModelSettings,
        **overrides: Any,
    ) -> ChatWorkerRequest:
        request = ChatWorkerRequest(
            conversation_id="conversation-1",
            user_message="question",
            history=(),
            settings=settings,
            enable_web_search=False,
        )
        return replace(request, **overrides)

    def test_worker_reports_completed_reply_and_progress(self) -> None:
        result_queue: queue.Queue[Any] = queue.Queue()
        progress_queue: queue.Queue[Any] = queue.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")
        workflow = WorkflowSpec(
            name="learned_web_search",
            version="0.1.0",
            description="Validated web research workflow.",
            steps=({"skill_name": "web_search"},),
            source_signature="signature",
        ).sealed()

        def fake_orchestrator_run(
            _self: Any,
            request: OrchestrationRequest,
            progress_callback: Callable[[str], None] | None = None,
        ) -> OrchestrationResult:
            self.assertEqual(_self.workflow_specs, (workflow,))
            self.assertTrue(request.enable_browser)
            self.assertTrue(request.enable_local_knowledge)
            self.assertEqual(request.conversation_id, "conversation-1")
            self.assertEqual(request.min_source_relevance, 0.8)
            self.assertEqual(request.response_detail, "balanced")
            assert progress_callback is not None
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
                self._request(
                    settings,
                    enable_web_search=True,
                    enable_browser=True,
                    enable_local_knowledge=True,
                    min_source_relevance=0.8,
                    response_detail="balanced",
                    workflow_specs=(workflow,),
                ),
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
        result_queue: queue.Queue[Any] = queue.Queue()
        progress_queue: queue.Queue[Any] = queue.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")

        with patch(
            "qwopus_agent.services.chat_service.AgentOrchestrator.run_sync",
            side_effect=TimeoutError("model timed out"),
        ):
            _run_chat_task(
                result_queue,
                progress_queue,
                self._request(settings),
            )

        status, content = result_queue.get_nowait()[:2]
        self.assertEqual(status, "failed")
        self.assertIn("model timed out", content)

    def test_start_maps_arguments_into_one_versioned_worker_request(self) -> None:
        context = Mock()
        result_queue = Mock()
        progress_queue = Mock()
        process = Mock()
        context.Queue.side_effect = [result_queue, progress_queue]
        context.Process.return_value = process
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")
        history = [{"role": "assistant", "content": "previous"}]
        workflow = WorkflowSpec(
            name="learned_web_search",
            version="0.1.0",
            description="Validated web research workflow.",
            steps=({"skill_name": "web_search"},),
            source_signature="signature",
        ).sealed()

        with patch(
            "qwopus_agent.services.chat_service.multiprocessing.get_context",
            return_value=context,
        ):
            task = start_chat_task(
                conversation_id="conversation-7",
                user_message="next question",
                history=history,
                settings=settings,
                enable_web_search=True,
                enable_browser=True,
                enable_local_knowledge=True,
                include_global_knowledge=True,
                min_source_relevance=0.72,
                response_detail="concise",
                knowledge_root=Path("/tmp/conversation-knowledge"),
                workflow_specs=(workflow,),
            )

        process_kwargs = context.Process.call_args.kwargs
        self.assertIs(process_kwargs["target"], _run_chat_task)
        self.assertTrue(process_kwargs["daemon"])
        self.assertEqual(len(process_kwargs["args"]), 3)
        self.assertIs(process_kwargs["args"][0], result_queue)
        self.assertIs(process_kwargs["args"][1], progress_queue)
        request = process_kwargs["args"][2]
        self.assertIsInstance(request, ChatWorkerRequest)
        self.assertEqual(request.schema_version, CHAT_WORKER_REQUEST_SCHEMA_VERSION)
        self.assertEqual(request.conversation_id, "conversation-7")
        self.assertEqual(request.user_message, "next question")
        self.assertEqual(request.history, (history[0],))
        self.assertIsNot(request.history[0], history[0])
        self.assertIs(request.settings, settings)
        self.assertTrue(request.enable_web_search)
        self.assertTrue(request.enable_browser)
        self.assertTrue(request.enable_local_knowledge)
        self.assertTrue(request.include_global_knowledge)
        self.assertEqual(request.min_source_relevance, 0.72)
        self.assertEqual(request.response_detail, "concise")
        self.assertEqual(request.knowledge_root, Path("/tmp/conversation-knowledge"))
        self.assertEqual(request.workflow_specs, (workflow,))
        self.assertIs(task.process, process)
        process.start.assert_called_once_with()

    def test_spawned_worker_rejects_unknown_request_schema(self) -> None:
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        progress_queue = context.Queue()
        settings = SmolagentsModelSettings(model_id="test", base_url="http://local/v1")
        request = self._request(
            settings,
            schema_version=CHAT_WORKER_REQUEST_SCHEMA_VERSION + 1,
        )
        process = context.Process(
            target=_run_chat_task,
            args=(result_queue, progress_queue, request),
        )

        process.start()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            self.fail("Spawned chat worker did not exit after schema validation.")

        status, content = result_queue.get(timeout=2)[:2]
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(status, "failed")
        self.assertIn("Unsupported chat worker request schema version", content)

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
