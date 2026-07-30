"""Background chat-run lifecycle shared by HTTP endpoints."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.memory import (
    DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    ConversationKnowledgeManager,
)
from qwopus_agent.services.chat_service import (
    BackgroundChatTask,
    ChatTaskResult,
    start_chat_task,
)
from qwopus_agent.services.intent_resolver import IntentResolver, build_context_snapshot
from qwopus_agent.services.orchestration_models import (
    ContextSnapshot,
    ConversationTaskState,
    InterpretationMode,
    ResolvedIntent,
)
from qwopus_agent.services.skill_growth_service import (
    SkillGrowthService,
    SkillRunTrace,
    SkillTraceStep,
)
from qwopus_agent.skills import SkillCatalog, WorkflowSkill, WorkflowSpec
from qwopus_agent.utils.debug_store import DEFAULT_DEBUG_DIRECTORY, append_debug_record

from .models import RunView
from .repository import ConversationMemoryRecord, ConversationRepository


@dataclass(frozen=True)
class PreparedChatRequest:
    """Resolved request and bounded state captured before a worker starts."""

    resolved_intent: ResolvedIntent
    snapshot: ContextSnapshot
    previous_task_state: ConversationTaskState


@dataclass
class _ActiveRun:
    conversation_id: str
    user_id: str
    username: str
    user_message_id: str
    model_id: str
    task: BackgroundChatTask
    prepared: PreparedChatRequest


@dataclass(frozen=True)
class _CompletedRun:
    view: RunView
    completed_at: float
    conversation_id: str
    user_id: str


DEFAULT_COMPLETED_RUN_TTL_SECONDS = 60 * 60
DEFAULT_MAX_COMPLETED_RUNS = 500
_REUSABLE_TOOL_SKILLS = {
    "tavily_search": "web_search",
    "rag_search": "rag_search",
    "graph_search": "graph_search",
}


class ChatRunRegistry:
    """Own local worker processes without leaking them into API route logic."""

    def __init__(
        self,
        repository: ConversationRepository,
        debug_directory: Path = DEFAULT_DEBUG_DIRECTORY,
        knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
        knowledge_manager: ConversationKnowledgeManager | None = None,
        skill_catalog: SkillCatalog | None = None,
        skill_growth: SkillGrowthService | None = None,
        intent_resolver: IntentResolver | None = None,
        completed_ttl_seconds: float = DEFAULT_COMPLETED_RUN_TTL_SECONDS,
        max_completed_runs: int = DEFAULT_MAX_COMPLETED_RUNS,
    ) -> None:
        self.repository = repository
        self.debug_directory = debug_directory
        self.knowledge_root = Path(knowledge_root)
        self.knowledge_manager = knowledge_manager
        self.skill_catalog = skill_catalog or SkillCatalog()
        self.skill_growth = skill_growth
        self.intent_resolver = intent_resolver or IntentResolver()
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
        enable_browser: bool = False,
        include_global_knowledge: bool = False,
        min_source_relevance: float = 0.55,
        response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
        interpretation_mode: InterpretationMode = "contextual",
        prepared: PreparedChatRequest | None = None,
        user_id: str = "system",
        username: str = "system",
        global_knowledge_path: Path | None = None,
    ) -> str:
        self.reap()
        resolved_request = prepared or self.prepare(
            conversation_id,
            content,
            response_detail=response_detail,
            interpretation_mode=interpretation_mode,
        )
        if resolved_request.resolved_intent.requires_clarification:
            raise ValueError("A clarification request cannot start an Agent worker.")
        history = self.repository.build_model_history(conversation_id)
        user_message = self.repository.add_message(conversation_id, "user", content)
        task = start_chat_task(
            conversation_id=conversation_id,
            user_message=content,
            history=history,
            settings=settings,
            enable_web_search=enable_web_search,
            enable_browser=enable_browser,
            enable_local_knowledge=enable_local_knowledge,
            include_global_knowledge=include_global_knowledge,
            min_source_relevance=min_source_relevance,
            response_detail=response_detail,
            knowledge_root=self.knowledge_root,
            global_knowledge_path=global_knowledge_path,
            resolved_intent=resolved_request.resolved_intent,
            workflow_specs=self._active_workflow_specs(),
        )
        run_id = uuid4().hex
        with self._lock:
            self._runs[run_id] = _ActiveRun(
                conversation_id,
                user_id,
                username,
                user_message.id,
                settings.model_id,
                task,
                resolved_request,
            )
        return run_id

    def _active_workflow_specs(self) -> tuple[WorkflowSpec, ...]:
        """Capture immutable, active workflow versions for one worker request."""
        if self.skill_growth is None:
            return ()
        specs: list[WorkflowSpec] = []
        for manifest in self.skill_growth.catalog.deployed():
            try:
                skill = self.skill_growth.registry.get(manifest.name)
            except KeyError:
                continue
            if (
                isinstance(skill, WorkflowSkill)
                and skill.spec.version == manifest.version
                and skill.spec.checksum == manifest.checksum
                and skill.spec.checksum_is_valid()
            ):
                specs.append(skill.spec)
        # 原因：spawn worker 不能依赖父进程中的可变 Registry，且运行中可能发生 promote/rollback。
        # 作用：每次请求固定使用启动时已激活且通过 checksum 校验的版本。
        return tuple(specs)

    def prepare(
        self,
        conversation_id: str,
        content: str,
        *,
        response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
        interpretation_mode: InterpretationMode = "contextual",
    ) -> PreparedChatRequest:
        """Resolve one request from persisted task state and safe source names."""
        memory = self.repository.get_memory(conversation_id)
        task_state = (
            memory.task_state
            if isinstance(memory, ConversationMemoryRecord)
            else ConversationTaskState()
        )
        sources = (
            self.knowledge_manager.list_sources(conversation_id)
            if self.knowledge_manager is not None
            else ()
        )
        active_skills = tuple(
            manifest.name for manifest in self.skill_catalog.deployed()
        )
        snapshot = build_context_snapshot(
            conversation_id=conversation_id,
            task_state=task_state,
            document_sources=sources,
            active_skill_names=active_skills,
        )
        return PreparedChatRequest(
            resolved_intent=self.intent_resolver.resolve(
                content,
                snapshot=snapshot,
                interpretation_mode=interpretation_mode,
                response_detail=response_detail,
            ),
            snapshot=snapshot,
            previous_task_state=task_state,
        )

    def complete_clarification(
        self,
        conversation_id: str,
        prepared: PreparedChatRequest,
        *,
        user_id: str = "system",
        username: str = "system",
    ) -> str:
        """Persist a clarification without starting or requiring a model."""
        question = prepared.resolved_intent.clarification_question
        if not prepared.resolved_intent.requires_clarification or not question:
            raise ValueError("The prepared request does not require clarification.")
        self.reap()
        self.repository.add_message(
            conversation_id,
            "user",
            prepared.resolved_intent.original_request,
        )
        self.repository.add_message(conversation_id, "assistant", question)
        run_id = uuid4().hex
        view = RunView(
            run_id=run_id,
            status="completed",
            phase="clarification",
            answer=question,
            trace=[
                {
                    "phase": "intent_resolution",
                    "status": "completed",
                    "message": "Clarification required before Agent execution.",
                }
            ],
        )
        append_debug_record(
            source="chat",
            status="completed",
            result=question,
            trace=view.trace,
            debug_runs=(),
            run_id=run_id,
            user_id=user_id,
            username=username,
            directory=self.debug_directory,
        )
        self._store_completed(
            run_id,
            view,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return run_id

    def conversation_id_for(self, run_id: str) -> str | None:
        """Return only the owning conversation id needed for route authorization."""
        with self._lock:
            active = self._runs.get(run_id)
            completed = self._completed.get(run_id)
        if active is not None:
            return active.conversation_id
        return completed.conversation_id if completed is not None else None

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
                user_id=active.user_id,
                username=active.username,
                directory=self.debug_directory,
            )
            self._store_completed(
                run_id,
                view,
                conversation_id=active.conversation_id,
                user_id=active.user_id,
            )
            return view

    def cancel_conversation(self, conversation_id: str) -> int:
        """Cancel every active run owned by one conversation."""
        return self._cancel_matching(
            lambda active: active.conversation_id == conversation_id
        )

    def cancel_user(self, user_id: str) -> int:
        """Cancel every active worker started by one disabled account."""
        return self._cancel_matching(lambda active: active.user_id == user_id)

    def cancel_user_conversation(self, conversation_id: str, user_id: str) -> int:
        """Cancel one member's active workers before their share is revoked."""
        return self._cancel_matching(
            lambda active: (
                active.conversation_id == conversation_id
                and active.user_id == user_id
            )
        )

    def _cancel_matching(
        self,
        matches: Callable[[_ActiveRun], bool],
    ) -> int:
        """Cancel a lock-consistent snapshot selected by an authorization event."""
        with self._poll_lock:
            self._prune_completed()
            with self._lock:
                matched = [
                    (run_id, active)
                    for run_id, active in self._runs.items()
                    if matches(active)
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
                    user_id=active.user_id,
                    username=active.username,
                    directory=self.debug_directory,
                )
                self._store_completed(
                    run_id,
                    view,
                    conversation_id=active.conversation_id,
                    user_id=active.user_id,
                )
            # 原因：删除会话后 worker 不能再向其外键消息表写入最终答案。
            # 作用：权限或数据删除前终止匹配任务，不依赖某一个前端标签页的状态。
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
            public_trace = list(result.trace)
            reusable_trace = (
                _reusable_skill_trace(result.trace, result.content)
                if result.status == "completed"
                else None
            )
            assistant_message_id: str | None = None
            if result.status == "completed":
                assistant_message = self.repository.add_message(
                    active.conversation_id,
                    "assistant",
                    result.content,
                )
                assistant_message_id = assistant_message.id
                self.repository.set_task_state(
                    active.conversation_id,
                    _completed_task_state(active.prepared),
                )
                public_trace.extend(
                    self._observe_skill_growth(
                        run_id,
                        active,
                        result,
                        reusable_trace,
                    )
                )
                view = RunView(
                    run_id=run_id,
                    status="completed",
                    phase="completed",
                    answer=result.content,
                    trace=public_trace,
                    citations=list(result.citations),
                )
            else:
                view = RunView(
                    run_id=run_id,
                    status="failed",
                    phase="failed",
                    error=result.content,
                    trace=public_trace,
                )
            self.repository.save_conversation_run(
                run_id=run_id,
                conversation_id=active.conversation_id,
                user_message_id=active.user_message_id,
                assistant_message_id=assistant_message_id,
                requested_by_user_id=active.user_id,
                objective=active.prepared.resolved_intent.original_request,
                operational_objective=(
                    active.prepared.resolved_intent.operational_objective
                ),
                status=result.status,
                model_id=active.model_id,
                reusable_skills=(
                    tuple(step.skill_name for step in reusable_trace.steps)
                    if reusable_trace is not None
                    else ()
                ),
            )
            # 原因：正式 API 不能返回 raw debug_runs，但独立 Console 必须能观察正式请求。
            # 作用：即使浏览器放弃轮询，reap 仍会持久化终态和完整内部诊断。
            append_debug_record(
                source="chat",
                status=result.status,
                result=result.content,
                trace=public_trace,
                debug_runs=result.debug_runs,
                run_id=run_id,
                user_id=active.user_id,
                username=active.username,
                directory=self.debug_directory,
            )
        finally:
            active.task.close()
        self._store_completed(
            run_id,
            view,
            conversation_id=active.conversation_id,
            user_id=active.user_id,
        )
        return view

    def _observe_skill_growth(
        self,
        run_id: str,
        active: _ActiveRun,
        result: ChatTaskResult,
        trace: SkillRunTrace | None,
    ) -> list[dict[str, str]]:
        """Submit only safe reusable Tool names after a successful final answer."""
        if self.skill_growth is None:
            return []
        if trace is None:
            return []
        try:
            decision = self.skill_growth.observe_trace(
                active.prepared.resolved_intent.operational_objective,
                trace,
                context={"trace_id": run_id},
            )
        except Exception as exc:  # noqa: BLE001 - growth is a non-fatal observer.
            return [
                {
                    "phase": "skill_growth",
                    "status": "warning",
                    "message": f"Skill candidate evaluation failed: {type(exc).__name__}.",
                }
            ]
        return [
            {
                "phase": "skill_growth",
                "status": "completed",
                "message": decision.reason,
            }
        ]

    def _store_completed(
        self,
        run_id: str,
        view: RunView,
        *,
        conversation_id: str,
        user_id: str,
    ) -> None:
        with self._lock:
            self._completed[run_id] = _CompletedRun(
                view,
                time.monotonic(),
                conversation_id,
                user_id,
            )
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


def _completed_task_state(prepared: PreparedChatRequest) -> ConversationTaskState:
    """Create the next state only after the final answer succeeds."""
    previous = prepared.previous_task_state
    intent = prepared.resolved_intent
    # 原因：失败或澄清轮次不能覆盖可继续的最后成功任务。
    # 作用：成功终态一次性保存解析目标、答案契约和可引用文档，同时保留 Skill 版本。
    return ConversationTaskState(
        last_successful_objective=intent.operational_objective,
        last_task_type=intent.task_type,
        last_answer_contract=intent.answer_contract,
        active_document_sources=prepared.snapshot.document_sources,
        last_skill_name=previous.last_skill_name,
        last_skill_version=previous.last_skill_version,
        open_tasks=previous.open_tasks,
        updated_at=datetime.now(UTC).isoformat(),
    )


def _reusable_skill_trace(
    events: tuple[dict[str, object], ...],
    output: str,
) -> SkillRunTrace | None:
    """Map one safe Tool trace to reusable BaseSkill names without observations."""
    steps: list[SkillTraceStep] = []
    for event in events:
        if event.get("phase") != "tool_call" or event.get("status") != "completed":
            continue
        tool_name = event.get("tool")
        if not isinstance(tool_name, str):
            return None
        skill_name = _REUSABLE_TOOL_SKILLS.get(tool_name)
        if skill_name is None:
            # 原因：全局知识、文档读取和文件 Tool 带有本轮权限或路径，不能固化成通用 Skill。
            # 作用：只要轨迹含不可安全复用的 Tool，就放弃整个候选而不是学习残缺子序列。
            return None
        steps.append(SkillTraceStep(skill_name))
    if not steps:
        return None
    return SkillRunTrace(success=True, output=output, steps=tuple(steps))
