"""Background chat-run lifecycle shared by HTTP endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
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


class ChatRunRegistry:
    """Own local worker processes without leaking them into API route logic."""

    def __init__(
        self,
        repository: ConversationRepository,
        debug_directory: Path = DEFAULT_DEBUG_DIRECTORY,
        knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    ) -> None:
        self.repository = repository
        self.debug_directory = debug_directory
        self.knowledge_root = Path(knowledge_root)
        self._runs: dict[str, _ActiveRun] = {}
        self._completed: dict[str, RunView] = {}
        self._lock = Lock()

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
    ) -> str:
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
            knowledge_root=self.knowledge_root,
        )
        run_id = uuid4().hex
        with self._lock:
            self._runs[run_id] = _ActiveRun(conversation_id, task)
        return run_id

    def poll(self, run_id: str) -> RunView | None:
        with self._lock:
            if run_id in self._completed:
                return self._completed[run_id]
            active = self._runs.get(run_id)
        if active is None:
            return None

        phase = active.task.refresh_phase()
        result = active.task.poll_result()
        if result is None:
            return RunView(run_id=run_id, status="running", phase=phase)

        if result.status == "completed":
            self.repository.add_message(active.conversation_id, "assistant", result.content)
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
        # 作用：在结果离开内部 worker 后旁路持久化，用户响应仍只包含安全字段。
        append_debug_record(
            source="chat",
            status=result.status,
            result=result.content,
            trace=result.trace,
            debug_runs=result.debug_runs,
            run_id=run_id,
            directory=self.debug_directory,
        )
        with self._lock:
            self._runs.pop(run_id, None)
            self._completed[run_id] = view
        return view

    def cancel(self, run_id: str) -> RunView | None:
        with self._lock:
            active = self._runs.pop(run_id, None)
        if active is None:
            return self._completed.get(run_id)
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
        with self._lock:
            self._completed[run_id] = view
        return view

    def debug_counts(self) -> tuple[int, int]:
        """Return a lock-consistent active/completed run count."""
        # 原因：Debug API 与聊天轮询可能同时访问运行表，直接读取私有字典会产生竞态。
        # 作用：只暴露数量且沿用现有锁，不让调试页面获得任务对象或取消能力。
        with self._lock:
            return len(self._runs), len(self._completed)
