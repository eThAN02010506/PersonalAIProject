"""Background chat execution for responsive UI clients."""

from __future__ import annotations

import multiprocessing
import time
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

    def report_progress(phase: str) -> None:
        progress_queue.put(phase)

    try:
        request.validate_schema()
        # 原因：聊天、联网和本地知识不能继续绕过统一任务入口各自运行。
        # 作用：后台进程只负责生命周期，所有路由与 Multi-Agent 决策交给 Orchestrator。
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
