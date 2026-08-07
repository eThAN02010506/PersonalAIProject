"""Background chat execution for responsive UI clients."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Literal, cast

from qwopus_agent.integrations.smolagents_runtime import ChatMessage, SmolagentsModelSettings
from qwopus_agent.memory import DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.intent_resolver import IntentResolver
from qwopus_agent.services.orchestration_models import (
    ConversationTurn,
    OrchestrationFile,
    OrchestrationRequest,
    ProcessEvent,
    ResolvedIntent,
)
from qwopus_agent.skills import WorkflowSpec

ChatTaskStatus = Literal["completed", "failed"]
CHAT_WORKER_REQUEST_SCHEMA_VERSION = 7


@dataclass(frozen=True)
class ChatWorkerRequest:
    """Serializable contract passed across the spawned-process boundary."""

    conversation_id: str
    user_message: str
    history: tuple[ChatMessage, ...]
    settings: SmolagentsModelSettings
    enable_web_search: bool
    enable_browser: bool = False
    resolved_intent: ResolvedIntent | None = None
    enable_local_knowledge: bool = False
    include_global_knowledge: bool = False
    min_source_relevance: float = 0.55
    max_evidence_sources: int = 12
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed"
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
    global_knowledge_path: Path | None = None
    workflow_specs: tuple[WorkflowSpec, ...] = ()
    uploaded_files: tuple[OrchestrationFile, ...] = ()
    schema_version: int = CHAT_WORKER_REQUEST_SCHEMA_VERSION

    def validate_schema(self) -> None:
        """Reject parent/worker code that disagree on the serialized contract."""
        if self.schema_version != CHAT_WORKER_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported chat worker request schema version "
                f"{self.schema_version}; expected {CHAT_WORKER_REQUEST_SCHEMA_VERSION}."
            )


@dataclass(frozen=True)
class ChatTaskResult:
    """Terminal result returned by one background Agent task."""

    status: ChatTaskStatus
    content: str
    trace: tuple[dict[str, Any], ...] = ()
    citations: tuple[dict[str, Any], ...] = ()
    debug_runs: tuple[dict[str, Any], ...] = ()


class _RunTimeoutError(TimeoutError):
    """Raised by the worker deadline watchdog when a run exceeds its budget.

    原因：超时必须是 `TimeoutError` 子类，避免被 `_is_model_connection_error`
    误判成模型连接故障而走错误的重试/分类路径。
    """


def _enforce_run_deadline(seconds: float) -> AbstractContextManager[None]:
    """Return a context manager arming a process-internal SIGALRM deadline.

    原因：`BackgroundChatTask.poll_result` 的超时只在父进程轮询时生效，挂起的
    Agent 不保证被终止。这里在 worker 进程内安装一个必然触发的定时器：
    - 首次触发：抛 `_RunTimeoutError`，正常中断正在执行的编排。
    - 第二次触发：若仍阻塞于 `asyncio.run` 的 executor 清理（挂起 socket 场景），
      `os._exit(1)` 强制收尾，把 worker 生命周期限定在 deadline + ~1s。
    作用：无论父进程是否轮询，worker 都保证在期限内结束。
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return nullcontext()

    @contextmanager
    def enforce() -> Iterator[None]:
        deadline_hit = False

        def _deadline_handler(_signum: int, _frame: Any) -> None:
            nonlocal deadline_hit
            if deadline_hit:
                # 原因：首次异常可能被 asyncio.run 的 executor 清理吞掉或阻塞。
                # 作用：第二次触发说明无法优雅中断，强制退出 worker 进程。
                os._exit(1)  # noqa: SLF001
            deadline_hit = True
            raise _RunTimeoutError(
                f"Agent run exceeded its configured timeout ({seconds:g} seconds)."
            )

        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _deadline_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return enforce()


@dataclass
class BackgroundChatTask:
    """Handle used by a UI to poll or terminate one Agent process."""

    process: Any
    result_queue: Any
    progress_queue: Any
    started_at: float
    timeout_seconds: float = 600.0
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
                if self.elapsed_seconds >= self.timeout_seconds:
                    # 原因：单次 HTTP timeout 不能限制多步 Agent 的累计等待时间。
                    # 作用：整轮超过截止时间时终止隔离 worker，避免 UI 永久轮询。
                    self._terminate_process()
                    return ChatTaskResult(
                        status="failed",
                        content=(
                            "Agent run exceeded its configured timeout "
                            f"({self.timeout_seconds:g} seconds)."
                        ),
                    )
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
        self._terminate_process()
        self._close_queues(wait=False)

    def _terminate_process(self) -> None:
        """Stop a live worker without deciding when its queues should be released."""
        if self.process.is_alive():
            # 原因：线程无法中断正在等待的模型 HTTP 请求，停止按钮会形同虚设。
            # 作用：终止独立 Agent 进程并关闭其连接，让 Web UI 立即恢复交互。
            self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)

    def close(self) -> None:
        """Release process and multiprocessing queue resources after a terminal run."""
        if not self.process.is_alive():
            self.process.join(timeout=0.1)
        self._close_queues(wait=True)

    def _close_queues(self, *, wait: bool) -> None:
        for channel in (self.result_queue, self.progress_queue):
            if not wait:
                cancel_join = getattr(channel, "cancel_join_thread", None)
                if callable(cancel_join):
                    cancel_join()
            close = getattr(channel, "close", None)
            if callable(close):
                close()
            if wait:
                join_thread = getattr(channel, "join_thread", None)
                if callable(join_thread):
                    join_thread()


def start_chat_task(
    conversation_id: str,
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings,
    enable_web_search: bool,
    enable_browser: bool = False,
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    min_source_relevance: float = 0.55,
    max_evidence_sources: int = 12,
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    global_knowledge_path: Path | None = None,
    resolved_intent: ResolvedIntent | None = None,
    workflow_specs: tuple[WorkflowSpec, ...] = (),
    uploaded_files: tuple[OrchestrationFile, ...] = (),
) -> BackgroundChatTask:
    """Start one cancelable Agent request in a spawned process."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    progress_queue = context.Queue()
    # 原因：CLI 或旧测试可直接调用该服务而不经过 FastAPI 的上下文准备阶段。
    # 作用：正式链路使用预解析结果，兼容入口则退化为无历史的确定性解析。
    resolved = resolved_intent or IntentResolver().resolve(
        user_message,
        response_detail=response_detail,
    )
    request = ChatWorkerRequest(
        conversation_id=conversation_id,
        user_message=user_message,
        history=tuple(dict(item) for item in history),
        settings=settings,
        enable_web_search=enable_web_search,
        enable_browser=enable_browser,
        resolved_intent=resolved,
        enable_local_knowledge=enable_local_knowledge,
        include_global_knowledge=include_global_knowledge,
        min_source_relevance=min_source_relevance,
        max_evidence_sources=max_evidence_sources,
        response_detail=response_detail,
        knowledge_root=Path(knowledge_root),
        global_knowledge_path=(
            Path(global_knowledge_path)
            if global_knowledge_path is not None
            else None
        ),
        workflow_specs=workflow_specs,
        uploaded_files=uploaded_files,
    )
    process = context.Process(
        target=_run_chat_task,
        args=(
            result_queue,
            progress_queue,
            request,
        ),
        daemon=True,
    )
    process.start()
    return BackgroundChatTask(
        process=process,
        result_queue=result_queue,
        progress_queue=progress_queue,
        started_at=time.monotonic(),
        timeout_seconds=settings.run_timeout_seconds,
    )


def _run_chat_task(
    result_queue: Any,
    progress_queue: Any,
    request: ChatWorkerRequest,
) -> None:
    """Run the synchronous Agent inside the cancelable worker process."""

    request_started = time.monotonic()

    def report_progress(phase: str) -> None:
        progress_queue.put(phase)

    try:
        request.validate_schema()
        # 原因：聊天、联网和本地知识不能继续绕过统一任务入口各自运行。
        # 作用：后台进程只负责生命周期，所有路由与 Multi-Agent 决策交给 Orchestrator。
        # 整轮超时由进程内 SIGALRM 看门狗强制，不依赖父进程轮询。
        with _enforce_run_deadline(request.settings.run_timeout_seconds):
            result = AgentOrchestrator(
                settings=request.settings,
                knowledge_root=request.knowledge_root,
                global_knowledge_path=request.global_knowledge_path,
                workflow_specs=request.workflow_specs,
            ).run_sync(
                OrchestrationRequest(
                    objective=request.user_message,
                    resolved_intent=request.resolved_intent,
                    interpretation_mode=(
                        request.resolved_intent.interpretation_mode
                        if request.resolved_intent is not None
                        else "contextual"
                    ),
                    conversation_id=request.conversation_id,
                    history=tuple(
                        ConversationTurn(
                            role=cast(Literal["user", "assistant"], item["role"]),
                            content=item["content"],
                        )
                        for item in request.history
                        if item.get("role") in {"user", "assistant"} and item.get("content")
                    ),
                    uploaded_files=request.uploaded_files,
                    enable_web_search=request.enable_web_search,
                    enable_browser=request.enable_browser,
                    enable_local_knowledge=request.enable_local_knowledge,
                    include_global_knowledge=request.include_global_knowledge,
                    min_source_relevance=request.min_source_relevance,
                    max_evidence_sources=request.max_evidence_sources,
                    response_detail=request.response_detail,
                ),
                progress_callback=report_progress,
            )
    except _RunTimeoutError as exc:
        # 原因：超时被看门狗触发时，trace 不能为空，否则阶段耗时指标丢失。
        # 作用：回传一个合成的 run_timeout 阶段事件，让用户和 Debug 都能看到超时与耗时。
        elapsed = max(0.0, time.monotonic() - request_started)
        result_queue.put(
            (
                "failed",
                str(exc),
                (
                    ProcessEvent(
                        phase="run_timeout",
                        status="failed",
                        message="exceeded configured run timeout",
                        duration_seconds=round(elapsed, 3),
                    ).model_dump(mode="json"),
                ),
                (),
                (),
            )
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
