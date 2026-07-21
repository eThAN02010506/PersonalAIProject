"""Background chat-run lifecycle shared by HTTP endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.services.chat_service import BackgroundChatTask, start_chat_task

from .models import RunView
from .repository import ConversationRepository


@dataclass
class _ActiveRun:
    conversation_id: str
    task: BackgroundChatTask


class ChatRunRegistry:
    """Own local worker processes without leaking them into API route logic."""

    def __init__(self, repository: ConversationRepository) -> None:
        self.repository = repository
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
    ) -> str:
        history = [
            {"role": message.role, "content": message.content}
            for message in self.repository.list_messages(conversation_id)
        ]
        self.repository.add_message(conversation_id, "user", content)
        task = start_chat_task(
            user_message=content,
            history=history,
            settings=settings,
            enable_web_search=enable_web_search,
            enable_local_knowledge=enable_local_knowledge,
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
        with self._lock:
            self._completed[run_id] = view
        return view
