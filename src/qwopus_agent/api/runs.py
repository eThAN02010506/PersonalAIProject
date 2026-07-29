"""Background chat-run lifecycle shared by HTTP endpoints."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.memory import DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
from qwopus_agent.services.chat_service import BackgroundChatTask, start_chat_task
from qwopus_agent.utils.debug_store import DEFAULT_DEBUG_DIRECTORY, append_debug_record

from .models import RunView
from .repository import ConversationRepository


@dataclass
class _ActiveRun:
    conversation_id: str
    task: BackgroundChatTask


@dataclass(frozen=True)
class _CompletedRun:
    view: RunView
    completed_at: float


DEFAULT_COMPLETED_RUN_TTL_SECONDS = 60 * 60
DEFAULT_MAX_COMPLETED_RUNS = 500


class ChatRunRegistry:
    """Own local worker processes without leaking them into API route logic."""

    def __init__(
        self,
        repository: ConversationRepository,
        debug_directory: Path = DEFAULT_DEBUG_DIRECTORY,
        knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
        completed_ttl_seconds: float = DEFAULT_COMPLETED_RUN_TTL_SECONDS,
        max_completed_runs: int = DEFAULT_MAX_COMPLETED_RUNS,
    ) -> None:
        self.repository = repository
        self.debug_directory = debug_directory
        self.knowledge_root = Path(knowledge_root)
        self.completed_ttl_seconds = completed_ttl_seconds
        self.max_completed_runs = max_completed_runs
        self._runs: dict[str, _ActiveRun] = {}
        self._completed: OrderedDict[str, _CompletedRun] = OrderedDict()
        self._lock = Lock()
        self._poll_lock = Lock()

    def start(
        self,
        conversation_id: str,
        content: str,
        settings: SmolagentsModelSettings,
        *,
        enable_web_search: bool,
        enable_local_knowledge: bool,
        include_global_knowledge: bool = False,
        min_source_relevance: float = 0.55,
        response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    ) -> str:
        self.reap()
        history = self.repository.build_model_history(conversation_id)
        self.repository.add_message(conversation_id, "user", content)
        task = start_chat_task(
            conversation_id=conversation_id,
            user_message=content,
            history=history,
            settings=settings,
            enable_web_search=enable_web_search,
            enable_local_knowledge=enable_local_knowledge,
            include_global_knowledge=include_global_knowledge,
            min_source_relevance=min_source_relevance,
            response_detail=response_detail,
            knowledge_root=self.knowledge_root,
        )
        run_id = uuid4().hex
        with self._lock:
            self._runs[run_id] = _ActiveRun(conversation_id, task)
        return run_id

    def poll(self, run_id: str) -> RunView | None:
        with self._poll_lock:
            self._prune_completed()
            with self._lock:
                completed = self._completed.get(run_id)
                active = self._runs.get(run_id)
            if completed is not None:
                return completed.view
            if active is None:
                return None
            return self._poll_active(run_id, active)

    def cancel(self, run_id: str) -> RunView | None:
        with self._poll_lock:
            self._prune_completed()
            with self._lock:
                active = self._runs.pop(run_id, None)
                completed = self._completed.get(run_id)
            if active is None:
                return completed.view if completed is not None else None
            active.task.cancel()
            view = RunView(run_id=run_id, status="cancelled", phase="cancelled")
            append_debug_record(
                source="chat",
                status="cancelled",
                trace=(),
                debug_runs=(),
                run_id=run_id,
                directory=self.debug_directory,
            )
            self._store_completed(run_id, view)
            return view

    def cancel_conversation(self, conversation_id: str) -> int:
        """Cancel every active run owned by one conversation."""
        with self._poll_lock:
            self._prune_completed()
            with self._lock:
                matched = [
                    (run_id, active)
                    for run_id, active in self._runs.items()
                    if active.conversation_id == conversation_id
                ]
                for run_id, _active in matched:
                    self._runs.pop(run_id, None)

            for run_id, active in matched:
                active.task.cancel()
                view = RunView(
                    run_id=run_id,
                    status="cancelled",
                    phase="cancelled",
                )
                append_debug_record(
                    source="chat",
                    status="cancelled",
                    trace=(),
                    debug_runs=(),
                    run_id=run_id,
                    directory=self.debug_directory,
                )
                self._store_completed(run_id, view)
            # 原因：删除会话后 worker 不能再向其外键消息表写入最终答案。
            # 作用：后端在删除数据前终止该会话全部任务，不依赖某一个前端标签页的状态。
            return len(matched)

    def reap(self) -> None:
        """Collect terminal workers even when their original browser stopped polling."""
        with self._poll_lock:
            self._prune_completed()
            with self._lock:
                active_runs = tuple(self._runs.items())
            for run_id, active in active_runs:
                self._poll_active(run_id, active)

    def close(self) -> None:
        """Cancel every active worker during API shutdown."""
        with self._poll_lock:
            with self._lock:
                active_runs = tuple(self._runs.values())
                self._runs.clear()
            for active in active_runs:
                active.task.cancel()

    def debug_counts(self) -> tuple[int, int]:
        """Return a lock-consistent active/completed run count."""
        self.reap()
        # 原因：Debug API 与聊天轮询可能同时访问运行表，直接读取私有字典会产生竞态。
        # 作用：只暴露数量且沿用现有锁，不让调试页面获得任务对象或取消能力。
        with self._lock:
            return len(self._runs), len(self._completed)

    def _poll_active(self, run_id: str, active: _ActiveRun) -> RunView:
        phase = active.task.refresh_phase()
        result = active.task.poll_result()
        if result is None:
            return RunView(run_id=run_id, status="running", phase=phase)

        with self._lock:
            current = self._runs.get(run_id)
            if current is not active:
                completed = self._completed.get(run_id)
                return (
                    completed.view
                    if completed is not None
                    else RunView(run_id=run_id, status="failed", phase="failed")
                )
            self._runs.pop(run_id, None)

        try:
            if result.status == "completed":
                self.repository.add_message(
                    active.conversation_id,
                    "assistant",
                    result.content,
                )
                view = RunView(
                    run_id=run_id,
                    status="completed",
                    phase="completed",
                    answer=result.content,
                    trace=list(result.trace),
                    citations=list(result.citations),
                )
            else:
                view = RunView(
                    run_id=run_id,
                    status="failed",
                    phase="failed",
                    error=result.content,
                    trace=list(result.trace),
                )
            # 原因：正式 API 不能返回 raw debug_runs，但独立 Console 必须能观察正式请求。
            # 作用：即使浏览器放弃轮询，reap 仍会持久化终态和完整内部诊断。
            append_debug_record(
                source="chat",
                status=result.status,
                result=result.content,
                trace=result.trace,
                debug_runs=result.debug_runs,
                run_id=run_id,
                directory=self.debug_directory,
            )
        finally:
            active.task.close()
        self._store_completed(run_id, view)
        return view

    def _store_completed(self, run_id: str, view: RunView) -> None:
        with self._lock:
            self._completed[run_id] = _CompletedRun(view, time.monotonic())
            self._completed.move_to_end(run_id)
        self._prune_completed()

    def _prune_completed(self) -> None:
        cutoff = time.monotonic() - self.completed_ttl_seconds
        with self._lock:
            expired = [
                run_id
                for run_id, completed in self._completed.items()
                if completed.completed_at < cutoff
            ]
            for run_id in expired:
                self._completed.pop(run_id, None)
            # 原因：SQLite 已保存最终对话，内存中的 RunView 只服务短期 HTTP 轮询。
            # 作用：TTL 和容量共同限制常驻内存，不影响已经持久化的聊天消息。
            while len(self._completed) > self.max_completed_runs:
                self._completed.popitem(last=False)
