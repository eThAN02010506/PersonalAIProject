"""Background chat execution for responsive UI clients."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Literal

from qwopus_agent.integrations.smolagents_runtime import (
    ChatMessage,
    SmolagentsModelSettings,
    run_agent_chat_turn,
)

ChatTaskStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class ChatTaskResult:
    """Terminal result returned by one background Agent task."""

    status: ChatTaskStatus
    content: str


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
            status, content = self.result_queue.get_nowait()
        except Empty:
            if self.process.is_alive():
                return None
            self.process.join(timeout=0.1)
            return ChatTaskResult(
                status="failed",
                content=f"Agent process exited unexpectedly ({self.process.exitcode}).",
            )

        self.process.join(timeout=0.1)
        return ChatTaskResult(status=status, content=str(content))

    def cancel(self) -> None:
        """Terminate the local worker so the UI stops waiting immediately."""
        if not self.process.is_alive():
            return
        # 原因：线程无法中断正在等待的模型 HTTP 请求，停止按钮会形同虚设。
        # 作用：终止独立 Agent 进程并关闭其连接，让 Streamlit 立即恢复交互。
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
) -> None:
    """Run the synchronous Agent inside the cancelable worker process."""

    def report_progress(phase: str) -> None:
        progress_queue.put(phase)

    try:
        reply = run_agent_chat_turn(
            user_message=user_message,
            history=history,
            settings=settings,
            enable_web_search=enable_web_search,
            progress_callback=report_progress,
        )
    except Exception as exc:
        result_queue.put(("failed", f"{type(exc).__name__}: {exc}"))
    else:
        result_queue.put(("completed", reply))
