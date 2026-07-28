"""Background chat execution for responsive UI clients."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import asdict, dataclass
from queue import Empty
from typing import Any, Literal, cast

from qwopus_agent.integrations.smolagents_runtime import ChatMessage, SmolagentsModelSettings
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import (
    ConversationTurn,
    OrchestrationRequest,
)

ChatTaskStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class ChatTaskResult:
    """Terminal result returned by one background Agent task."""

    status: ChatTaskStatus
    content: str
    trace: tuple[dict[str, Any], ...] = ()
    citations: tuple[dict[str, Any], ...] = ()
    debug_runs: tuple[dict[str, Any], ...] = ()


@dataclass
class BackgroundChatTask:
    """Handle used by a UI to poll or terminate one Agent process."""

    process: Any
    result_queue: Any
    progress_queue: Any
    started_at: float
    phase: str = "connecting"

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def refresh_phase(self) -> str:
        """Drain progress events and return the most recent phase."""
        while True:
            try:
                self.phase = str(self.progress_queue.get_nowait())
            except Empty:
                return self.phase

    def poll_result(self) -> ChatTaskResult | None:
        """Return a terminal result without blocking the UI."""
        try:
            payload = self.result_queue.get_nowait()
        except Empty:
            if self.process.is_alive():
                return None
            self.process.join(timeout=0.1)
            return ChatTaskResult(
                status="failed",
                content=f"Agent process exited unexpectedly ({self.process.exitcode}).",
            )

        self.process.join(timeout=0.1)
        status, content = payload[:2]
        trace = tuple(payload[2]) if len(payload) > 2 else ()
        citations = tuple(payload[3]) if len(payload) > 3 else ()
        debug_runs = tuple(payload[4]) if len(payload) > 4 else ()
        return ChatTaskResult(
            status=status,
            content=str(content),
            trace=trace,
            citations=citations,
            debug_runs=debug_runs,
        )

    def cancel(self) -> None:
        """Terminate the local worker so the UI stops waiting immediately."""
        if not self.process.is_alive():
            return
        # 原因：线程无法中断正在等待的模型 HTTP 请求，停止按钮会形同虚设。
        # 作用：终止独立 Agent 进程并关闭其连接，让 Web UI 立即恢复交互。
        self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)


def start_chat_task(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings,
    enable_web_search: bool,
    enable_local_knowledge: bool = False,
    min_source_relevance: float = 0.55,
) -> BackgroundChatTask:
    """Start one cancelable Agent request in a spawned process."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    progress_queue = context.Queue()
    process = context.Process(
        target=_run_chat_task,
        args=(
            result_queue,
            progress_queue,
            user_message,
            history,
            settings,
            enable_web_search,
            enable_local_knowledge,
            min_source_relevance,
        ),
        daemon=True,
    )
    process.start()
    return BackgroundChatTask(
        process=process,
        result_queue=result_queue,
        progress_queue=progress_queue,
        started_at=time.monotonic(),
    )


def _run_chat_task(
    result_queue: Any,
    progress_queue: Any,
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings,
    enable_web_search: bool,
    enable_local_knowledge: bool = False,
    min_source_relevance: float = 0.55,
) -> None:
    """Run the synchronous Agent inside the cancelable worker process."""

    def report_progress(phase: str) -> None:
        progress_queue.put(phase)

    try:
        # 原因：聊天、联网和本地知识不能继续绕过统一任务入口各自运行。
        # 作用：后台进程只负责生命周期，所有路由与 Multi-Agent 决策交给 Orchestrator。
        result = AgentOrchestrator(settings=settings).run_sync(
            OrchestrationRequest(
                objective=user_message,
                history=tuple(
                    ConversationTurn(
                        role=cast(Literal["user", "assistant"], item["role"]),
                        content=item["content"],
                    )
                    for item in history
                    if item.get("role") in {"user", "assistant"} and item.get("content")
                ),
                enable_web_search=enable_web_search,
                enable_local_knowledge=enable_local_knowledge,
                min_source_relevance=min_source_relevance,
            ),
            progress_callback=report_progress,
        )
    except Exception as exc:
        result_queue.put(("failed", f"{type(exc).__name__}: {exc}"))
    else:
        status = "completed" if result.success else "failed"
        result_queue.put(
            (
                status,
                result.final_answer,
                [event.model_dump(mode="json") for event in result.trace],
                [citation.model_dump(mode="json") for citation in result.citations],
                [asdict(debug_run) for debug_run in result.debug_runs],
            )
        )
