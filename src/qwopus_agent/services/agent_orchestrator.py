"""Unified application orchestration for chat, files, research, knowledge, and reports."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwopus_agent.agents import AgentPlanningRequest, Executor, Planner
from qwopus_agent.agents.multi_agent import (
    AgentProfile,
    MultiAgentRun,
    RunnableAgent,
)
from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    ChatAgentRun,
    SmolagentsModelSettings,
    run_agent_chat_turn_with_debug,
)
from qwopus_agent.memory import DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
from qwopus_agent.services.answer_pipeline import (
    build_answer_plan,
    build_evidence_ledger,
    is_internal_pipeline_payload,
    parse_evidence_packet,
    parse_evidence_review,
    render_answer_plan,
    render_evidence_ledger,
    render_evidence_review,
)
from qwopus_agent.services.orchestration_models import (
    AgentOutputRole,
    AnswerContract,
    AnswerPlan,
    EvidenceLedger,
    EvidencePacket,
    EvidenceReview,
    OrchestrationRequest,
    OrchestrationResult,
    ProcessEvent,
    SourceCitation,
)
from qwopus_agent.skills import WorkflowSpec
from qwopus_agent.utils.token_budget import TokenBudgetManager, truncate_to_tokens

if TYPE_CHECKING:
    from qwopus_agent.analysis import AnalysisResult
    from qwopus_agent.documents import DocumentStore
    from qwopus_agent.memory import MiniRAG
    from qwopus_agent.reports import GeneratedReport, ReportGenerator
    from qwopus_agent.services.analysis_service import UploadAnalysisOutcome

ProgressCallback = Callable[[str], None]
ChatRunner = Callable[..., ChatAgentRun]
AnalysisRunner = Callable[..., Any]

_URL_PATTERN = re.compile(r"""https?://[^\s)\]>`"']+""")
_SOURCE_PATTERN = re.compile(
    r"\[Source:\s*(?P<source>[^|\]\n]+?)(?:\s*\|[^\]\n]*)?\]",
    re.I,
)
_EVIDENCE_PATTERN = re.compile(
    r"\[(?P<source>[^\],\n]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls))"
    r"(?:,\s*page\s*(?P<page>[^\]]+))?\]",
    re.I,
)


@dataclass(frozen=True)
class _CapabilityResult:
    """Internal result shape understood by the generic Supervisor helpers."""

    content: str
    success: bool = False
    confidence: float = 0.5
    citations: tuple[SourceCitation, ...] = ()
    analysis_result: AnalysisResult | None = None
    report: GeneratedReport | None = None
    error: str | None = None
    debug_runs: tuple[AgentDebugRun, ...] = ()
    evidence_packet: EvidencePacket | None = None
    evidence_review: EvidenceReview | None = None


@dataclass
class _FunctionAgent:
    """Small adapter that makes an injected async function a RunnableAgent."""

    handler: Callable[[str, dict[str, Any]], Awaitable[Any]]

    async def run(self, question: str, context: dict[str, Any] | None = None) -> Any:
        return await self.handler(question, context or {})


@dataclass
class AgentOrchestrator:
    """Choose the cheapest valid route and coordinate complex requests through Supervisor."""

    settings: SmolagentsModelSettings
    minirag: MiniRAG | None = None
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
    global_knowledge_path: Path | None = None
    document_store: DocumentStore | None = None
    chat_runner: ChatRunner = run_agent_chat_turn_with_debug
    analysis_runner: AnalysisRunner | None = None
    report_generator: ReportGenerator | None = None
    workflow_specs: tuple[WorkflowSpec, ...] = ()
    planner: Planner = dataclass_field(default_factory=Planner)
    executor: Executor = dataclass_field(default_factory=Executor)

    async def run(
        self,
        request: OrchestrationRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        """Plan once and execute the resulting DAG through one Supervisor path."""
        trace: list[ProcessEvent] = []
        try:
            if request.include_global_knowledge and not request.enable_local_knowledge:
                raise ValueError(
                    "Global knowledge requires local knowledge permission."
                )
            # 原因：Planner 若只看到“继续”“再详细一点”，无法知道本轮真正要完成的任务。
            # 作用：规划和能力委派使用已解析目标，原始 objective 继续负责语言与用户审计。
            planning_objective = (
                request.resolved_intent.operational_objective
                if request.resolved_intent is not None
                else request.objective
            ).strip()
            if not planning_objective:
                # 原因：ResolvedIntent 可能由未来入口构造，不能只依赖 request.objective 校验。
                # 作用：空目标永远不会进入 AnswerPlan、Planner 或任何文档 Tool。
                raise ValueError("Planning objective must not be blank.")
            answer_contract = _answer_contract(request)
            answer_plan = build_answer_plan(planning_objective, answer_contract)
            planner_local_knowledge = (
                request.enable_local_knowledge
                and not _is_spreadsheet_only_local_computation_request(request)
            )
            plan = await self.planner.plan(
                AgentPlanningRequest(
                    objective=planning_objective,
                    has_documents=bool(request.uploaded_files),
                    enable_web_search=request.enable_web_search,
                    enable_browser=request.enable_browser,
                    enable_local_knowledge=planner_local_knowledge,
                    generate_report=request.generate_report,
                    complexity=(
                        answer_contract.complexity
                    ),
                    response_detail=answer_contract.response_detail,
                )
            )
            agents = self._build_agents(
                request,
                answer_plan,
                evidence_mode=plan.route == "multi_agent",
                trace=trace,
                progress_callback=progress_callback,
            )
            profiles = {
                name: AgentProfile(
                    name=name,
                    capabilities=(name.removesuffix("_agent"),),
                )
                for name in agents
            }
            if progress_callback is not None:
                progress_callback("planning")
            trace.append(
                ProcessEvent(
                    phase="planning",
                    status="completed",
                    agent="planner",
                    message=f"Planned {len(plan.delegation.tasks)} task(s).",
                )
            )
            trace.append(
                ProcessEvent(
                    phase="answer_plan",
                    status="completed",
                    agent="planner",
                    message=(
                        f"Answer plan contains {len(answer_plan.required_sections)} section(s) "
                        f"and {len(answer_plan.depth_questions)} depth question(s)."
                    ),
                )
            )
            run = await self.executor.execute(
                plan,
                agents=agents,
                profiles=profiles,
                context={
                    "shared_state": {
                        "request": request.model_dump(exclude={"uploaded_files"}),
                        "answer_plan": answer_plan,
                    }
                },
            )
            _, analysis_result, report, debug_runs = _collect_artifacts(run)
            terminal_run = next(
                (item for item in run.runs if item.task_id == plan.terminal_task_id),
                None,
            )
            terminal_result = terminal_run.result if terminal_run is not None else None
            terminal_success = bool(
                terminal_run is not None
                and terminal_run.success
                and isinstance(terminal_result, _CapabilityResult)
                and terminal_result.success
                and terminal_result.content.strip()
            )
            if terminal_success and isinstance(terminal_result, _CapabilityResult):
                citations = terminal_result.citations
                answer = _append_citations(
                    terminal_result.content,
                    citations,
                    request.objective,
                )
            elif plan.route == "single_agent" and isinstance(
                terminal_result, _CapabilityResult
            ):
                # 原因：确定性仲裁在单 Agent 失败时只返回通用错误，会丢失安全的拒答说明。
                # 作用：继续把能力层的无证据提示交给用户，同时 success 保持为 False。
                citations = ()
                answer = terminal_result.content.strip() or _terminal_failure_answer(
                    request.objective
                )
            else:
                # 原因：Arbiter 会从任意成功依赖中选择内容，即使规划的 terminal 已失败或未执行。
                # 作用：多 Agent 的用户答案只归成功 terminal 所有，中间 JSON 仍保留在调试结果中。
                citations = ()
                answer = _terminal_failure_answer(request.objective)
            return OrchestrationResult(
                success=terminal_success,
                final_answer=answer,
                route=plan.route,
                citations=citations,
                trace=tuple(trace),
                analysis_result=analysis_result,
                report=report,
                multi_agent_run=run if plan.route == "multi_agent" else None,
                resolved_intent=request.resolved_intent,
                debug_runs=debug_runs,
            )
        except Exception as exc:  # noqa: BLE001 - return one stable application error envelope.
            trace.append(
                ProcessEvent(
                    phase="orchestration",
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            return OrchestrationResult(
                success=False,
                final_answer=_terminal_failure_answer(request.objective),
                route=(
                    "multi_agent"
                    if sum(
                        (
                            bool(request.uploaded_files),
                            request.enable_web_search,
                            request.enable_browser,
                            request.enable_local_knowledge,
                        )
                    )
                    > 1
                    or request.generate_report
                    else "single_agent"
                ),
                trace=tuple(trace),
                resolved_intent=request.resolved_intent,
            )

    def run_sync(
        self,
        request: OrchestrationRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        """Run from synchronous CLI/UI adapters."""
        return asyncio.run(self.run(request, progress_callback=progress_callback))


    def _build_agents(
        self,
        request: OrchestrationRequest,
        answer_plan: AnswerPlan,
        *,
        evidence_mode: bool,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> dict[str, RunnableAgent]:
        async def chat_handler(
            name: str,
            question: str,
            context: dict[str, Any],
            *,
            web: bool = False,
            browser: bool = False,
            local: bool = False,
        ) -> _CapabilityResult:
            delegated_request = request.model_copy(update={"objective": question})
            task_id = str(
                context.get("multi_agent", {}).get("task_id", name)
            )
            return await self._guarded(
                name,
                trace,
                lambda: self._chat_capability(
                    name,
                    delegated_request,
                    trace,
                    progress_callback,
                    web=web,
                    browser=browser,
                    local=local,
                    output_role="evidence" if evidence_mode else "final",
                    answer_plan=answer_plan,
                    task_id=task_id,
                ),
            )

        agents: dict[str, RunnableAgent] = {}
        if request.uploaded_files:
            agents["document_agent"] = _FunctionAgent(
                lambda _question, _context: self._guarded(
                    "document_agent",
                    trace,
                    lambda: self._document_capability(
                        request,
                        answer_plan,
                        trace,
                        progress_callback,
                        evidence_mode=evidence_mode,
                    ),
                )
            )
        if request.enable_web_search:
            agents["research_agent"] = _FunctionAgent(
                lambda question, context: chat_handler(
                    "research_agent", question, context, web=True
                )
            )
        if request.enable_browser:
            agents["browser_agent"] = _FunctionAgent(
                lambda question, context: chat_handler(
                    "browser_agent", question, context, browser=True
                )
            )
        if request.enable_local_knowledge:
            agents["knowledge_agent"] = _FunctionAgent(
                lambda question, context: chat_handler(
                    "knowledge_agent", question, context, local=True
                )
            )
        if not agents:
            agents["chat_agent"] = _FunctionAgent(
                lambda question, context: chat_handler("chat_agent", question, context)
            )
        agents["review_agent"] = _FunctionAgent(
            lambda question, context: self._review_capability(
                request,
                answer_plan,
                question,
                context,
                trace,
                progress_callback,
            )
        )
        if evidence_mode and (
            request.enable_web_search
            or request.enable_browser
            or request.enable_local_knowledge
        ):
            agents["gap_fill_agent"] = _FunctionAgent(
                lambda question, context: self._gap_fill_capability(
                    request,
                    answer_plan,
                    question,
                    context,
                    trace,
                    progress_callback,
                )
            )
        agents["synthesis_agent"] = _FunctionAgent(
            lambda question, context: self._synthesis_capability(
                request,
                answer_plan,
                question,
                context,
                trace,
                progress_callback,
            )
        )
        if request.generate_report:
            agents["report_agent"] = _FunctionAgent(
                lambda question, context: self._report_capability(
                    request, question, context, trace
                )
            )
        return agents

    async def _chat_capability(
        self,
        agent_name: str,
        request: OrchestrationRequest,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
        *,
        web: bool,
        browser: bool,
        local: bool,
        output_role: AgentOutputRole = "final",
        answer_plan: AnswerPlan | None = None,
        task_id: str | None = None,
        enforce_document_evidence: bool = True,
        answer_contract: AnswerContract | None = None,
        response_language_source: str | None = None,
    ) -> _CapabilityResult:
        started = time.monotonic()
        trace.append(ProcessEvent(phase="execution", status="started", agent=agent_name))
        history = [turn.model_dump() for turn in request.history]
        try:
            run = await asyncio.to_thread(
                self.chat_runner,
                user_message=request.objective,
                history=history,
                settings=self.settings,
                enable_web_search=web,
                enable_browser=browser,
                enable_local_knowledge=local,
                include_global_knowledge=(
                    request.include_global_knowledge if local else False
                ),
                min_source_relevance=request.min_source_relevance,
                max_evidence_sources=request.max_evidence_sources,
                response_detail=request.response_detail,
                knowledge_scope=request.conversation_id,
                knowledge_root=self.knowledge_root,
                global_knowledge_path=self.global_knowledge_path,
                document_evidence_available=bool(request.uploaded_files),
                enforce_document_evidence=enforce_document_evidence,
                response_language_source=(
                    response_language_source
                    or (
                        request.resolved_intent.original_request
                        if request.resolved_intent is not None
                        else request.objective
                    )
                ),
                answer_contract=answer_contract or _answer_contract(request),
                output_role=output_role,
                answer_plan=answer_plan,
                # 原因：worker 不应在执行中重读可变 Catalog，否则一次请求可能混用两个版本。
                # 作用：把父进程已校验的 active WorkflowSpec 快照交给 smolagents 装配。
                promoted_workflows=self.workflow_specs,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            if not _is_model_connection_error(exc):
                raise
            technical_error = f"{type(exc).__name__}: {exc}"
            trace.append(
                ProcessEvent(
                    phase="execution",
                    status="failed",
                    agent=agent_name,
                    message=technical_error,
                    duration_seconds=round(time.monotonic() - started, 3),
                )
            )
            # 原因：模型可能在预检成功后、Tool 检索或最终生成期间断开连接。
            # 作用：正式答案不暴露 smolagents 异常，完整技术原因仍保留在运行轨迹中。
            return _CapabilityResult(
                content=_model_connection_error_answer(request.objective),
                success=False,
                confidence=0.0,
                error=technical_error,
            )
        citations = (
            _citations_from_chat(run)
            if output_role == "final"
            else _parse_citations("\n".join(run.observations))
        )
        confidence = (0.72 if web or browser or local else 0.65) if run.success else 0.0
        evidence_packet = (
            parse_evidence_packet(
                run.answer,
                task_id=task_id or agent_name,
                agent_name=agent_name,
                citations=citations,
                fallback_confidence=confidence,
                trust_declared_sources=False,
                max_sources=request.max_evidence_sources,
            )
            if output_role == "evidence" and run.success
            else None
        )
        evidence_review = (
            parse_evidence_review(run.answer)
            if output_role == "review" and run.success
            else None
        )
        for tool_name in run.tool_calls:
            if tool_name != "final_answer":
                trace.append(
                    ProcessEvent(
                        phase="tool_call",
                        status="completed",
                        agent=agent_name,
                        tool=tool_name,
                        message=f"{agent_name} used {tool_name}.",
                    )
                )
        trace.append(
            ProcessEvent(
                phase="execution",
                status="completed" if run.success else "failed",
                agent=agent_name,
                message=run.error or "",
                duration_seconds=round(time.monotonic() - started, 3),
            )
        )
        return _CapabilityResult(
            content=run.answer,
            success=run.success,
            confidence=confidence,
            citations=citations,
            error=run.error,
            debug_runs=run.debug_runs,
            evidence_packet=evidence_packet,
            evidence_review=evidence_review,
        )

    async def _document_capability(
        self,
        request: OrchestrationRequest,
        answer_plan: AnswerPlan,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
        *,
        evidence_mode: bool,
    ) -> _CapabilityResult:
        direct_local_files = all(
            item.local_path is not None for item in request.uploaded_files
        )
        if self.minirag is None and not direct_local_files:
            raise RuntimeError("Document orchestration requires a MiniRAG instance.")
        if progress_callback is not None:
            progress_callback("analyzing")
        started = time.monotonic()
        trace.append(ProcessEvent(phase="execution", status="started", agent="document_agent"))
        if self.analysis_runner is None:
            from qwopus_agent.services.analysis_service import analyze_uploaded_files

            analysis_runner = analyze_uploaded_files
        else:
            analysis_runner = self.analysis_runner
        from qwopus_agent.services.analysis_service import UploadedFileInput

        analysis_options: dict[str, Any] = {
            "uploaded_files": [
                UploadedFileInput(
                    name=item.name,
                    content=item.content,
                    local_path=item.local_path,
                )
                for item in request.uploaded_files
            ],
            "user_question": request.objective,
            "settings": self.settings,
            "minirag": self.minirag,
            "min_source_relevance": request.min_source_relevance,
            "selected_sections": request.selected_sections,
            "analysis_mode": request.analysis_mode,
            "response_detail": request.response_detail,
        }
        if self.document_store is not None:
            analysis_options["document_store"] = self.document_store
        outcome: UploadAnalysisOutcome = await asyncio.to_thread(
            analysis_runner,
            **analysis_options,
        )
        answer = outcome.result.llm_analysis or outcome.result.markdown_summary
        if request.objective.strip() and not outcome.result.llm_analysis:
            # 原因：本地解析摘要不是用户所要求的生成式写作答案，模型离线时不能冒充完成。
            # 作用：保留解析产物供内部诊断，但让 API/任务状态 fail closed。
            message = (
                "The documents were parsed, but the model did not produce the requested "
                "analysis. Check the model connection and retry."
            )
            trace.append(
                ProcessEvent(
                    phase="execution",
                    status="failed",
                    agent="document_agent",
                    message=message,
                    duration_seconds=round(time.monotonic() - started, 3),
                )
            )
            return _CapabilityResult(
                content=message,
                success=False,
                confidence=0.0,
                error=message,
                analysis_result=outcome.result,
                debug_runs=tuple(getattr(outcome, "debug_runs", ())),
            )
        citations = _file_analysis_citations(
            answer,
            outcome.analyzed_file_names,
        )
        for message in outcome.debug_steps:
            trace.append(
                ProcessEvent(
                    phase="document_analysis",
                    status="completed",
                    agent="document_agent",
                    message=message,
                )
            )
        trace.append(
            ProcessEvent(
                phase="execution",
                status="completed",
                agent="document_agent",
                duration_seconds=round(time.monotonic() - started, 3),
            )
        )
        return _CapabilityResult(
            content=answer,
            success=True,
            confidence=0.8,
            citations=citations,
            analysis_result=outcome.result,
            debug_runs=tuple(getattr(outcome, "debug_runs", ())),
            evidence_packet=(
                parse_evidence_packet(
                    answer,
                    task_id="document",
                    agent_name="document_agent",
                    citations=citations,
                    fallback_confidence=0.8,
                    trust_declared_sources=False,
                    fallback_plan_item_ids=(
                        item.item_id for item in answer_plan.plan_items
                    ),
                    max_sources=request.max_evidence_sources,
                )
                if evidence_mode
                else None
            ),
        )

    async def _review_capability(
        self,
        request: OrchestrationRequest,
        answer_plan: AnswerPlan,
        question: str,
        context: dict[str, Any],
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> _CapabilityResult:
        """Review independent evidence without reopening any Tool."""
        ledger = _evidence_ledger_from_context(
            context,
            max_sources=request.max_evidence_sources,
        )
        budget = TokenBudgetManager(
            context_window=self.settings.context_window_tokens,
            output_reserve=min(self.settings.max_tokens, 1200),
        )
        evidence = truncate_to_tokens(
            render_evidence_ledger(ledger),
            budget.synthesis_budget,
        )
        review_request = request.model_copy(
            update={
                "objective": (
                    f"Original request: {question}\n\n"
                    f"Answer plan:\n{render_answer_plan(answer_plan)}\n\n"
                    f"Independent evidence ledger:\n{evidence}\n\n"
                    "Audit this evidence against every plan item for the final answering agent. "
                    "Return one coverage row per plan item, then identify agreements, factual "
                    "conflicts, unsupported claims, material gaps, and the safest resolution. "
                    "Use supported only when mapped evidence directly establishes the item; use "
                    "partial when useful evidence exists but an important condition is absent."
                ),
                "history": (),
                "enable_web_search": False,
                "enable_browser": False,
                "enable_local_knowledge": False,
                "include_global_knowledge": False,
                "uploaded_files": (),
                "resolved_intent": None,
            }
        )
        # 原因：让原 Evidence Agent 再“辩论”会重复访问文件、网络或知识库。
        # 作用：只用已完成结果执行一次无工具审阅，延迟有界且证据快照保持一致。
        reviewed = await self._chat_capability(
            "review_agent",
            review_request,
            trace,
            progress_callback,
            web=False,
            browser=False,
            local=False,
            output_role="review",
            answer_plan=answer_plan,
            task_id="review",
            enforce_document_evidence=False,
            answer_contract=_answer_contract(request),
            response_language_source=(
                request.resolved_intent.original_request
                if request.resolved_intent is not None
                else request.objective
            ),
        )
        trace.append(
            ProcessEvent(
                phase="reflection",
                status="completed" if reviewed.success else "failed",
                agent="review_agent",
                message="Independent evidence review completed." if reviewed.success else "",
            )
        )
        return _CapabilityResult(
            content=reviewed.content,
            success=reviewed.success,
            confidence=0.85 if reviewed.success else 0.0,
            error=reviewed.error,
            debug_runs=reviewed.debug_runs,
            evidence_review=reviewed.evidence_review,
        )

    async def _gap_fill_capability(
        self,
        request: OrchestrationRequest,
        answer_plan: AnswerPlan,
        question: str,
        context: dict[str, Any],
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> _CapabilityResult:
        """Run at most one targeted evidence pass when Review names material gaps."""
        review = _evidence_review_from_context(context)
        if not review.gaps:
            return _CapabilityResult(
                content="Reviewer found no material evidence gaps.",
                success=True,
                confidence=1.0,
                evidence_packet=EvidencePacket(
                    task_id="gap_fill",
                    agent_name="gap_fill_agent",
                    limitations=("No gap-fill retrieval was required.",),
                ),
            )
        gap_request = request.model_copy(
            update={
                "objective": (
                    f"Original request: {question}\n\n"
                    "Fill only these reviewed evidence gaps:\n"
                    + "\n".join(f"- {gap}" for gap in review.gaps)
                    + "\n\nReturn evidence for the gaps only; do not repeat established facts."
                ),
                "history": (),
            }
        )
        result = await self._chat_capability(
            "gap_fill_agent",
            gap_request,
            trace,
            progress_callback,
            web=request.enable_web_search,
            browser=request.enable_browser,
            local=request.enable_local_knowledge,
            output_role="evidence",
            answer_plan=answer_plan,
            task_id="gap_fill",
        )
        if result.success:
            return result
        # 原因：补证是改进步骤而非原始证据的硬前提，单次零命中不应抹掉已审阅的有效材料。
        # 作用：将失败记录为 limitation 并允许 Synthesizer 明示不确定性，不再循环重试。
        return _CapabilityResult(
            content="Targeted gap-fill did not return usable evidence.",
            success=True,
            confidence=0.0,
            error=result.error,
            debug_runs=result.debug_runs,
            evidence_packet=EvidencePacket(
                task_id="gap_fill",
                agent_name="gap_fill_agent",
                limitations=("Targeted gap-fill did not return usable evidence.",),
            ),
        )

    async def _synthesis_capability(
        self,
        request: OrchestrationRequest,
        answer_plan: AnswerPlan,
        question: str,
        context: dict[str, Any],
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> _CapabilityResult:
        ledger = _evidence_ledger_from_context(
            context,
            max_sources=request.max_evidence_sources,
        )
        review = _evidence_review_from_context(context)
        budget = TokenBudgetManager(
            context_window=self.settings.context_window_tokens,
            output_reserve=self.settings.max_tokens,
        )
        evidence = truncate_to_tokens(
            (
                f"Answer plan:\n{render_answer_plan(answer_plan)}\n\n"
                f"Evidence ledger:\n{render_evidence_ledger(ledger)}\n\n"
                f"Evidence review:\n{render_evidence_review(review)}"
            ),
            budget.synthesis_budget,
        )
        synthesis_request = request.model_copy(
            update={
                "objective": (
                    f"Original request: {question}\n\n"
                    f"Internal synthesis material:\n{evidence}\n\n"
                    "Write one coherent final answer around the plan's central goal. For every "
                    "supported or partial plan item, state the conclusion, provide the specific "
                    "support, explain why that support establishes the conclusion, and give its "
                    "practical implication, condition, example, or limitation when relevant. "
                    "Disclose material missing or conflicted items instead of filling them with "
                    "general knowledge. Resolve reviewed conflicts, distinguish supported facts "
                    "from uncertainty, preserve only citations present in the Evidence Ledger, "
                    "and do not mention internal agents or execution steps. Empty-source facts "
                    "are architectural reasoning, not verified studies; never invent "
                    "publications, percentages, or benchmark measurements. Headings, generic "
                    "advice, and repeated conclusions do not count as detail."
                ),
                "history": (),
                "enable_web_search": False,
                "enable_browser": False,
                "enable_local_knowledge": False,
                "include_global_knowledge": False,
                "uploaded_files": (),
                "resolved_intent": None,
            }
        )
        trusted_citations = _trusted_citations_from_context(
            context,
            max_sources=request.max_evidence_sources,
        )
        try:
            result = await self._chat_capability(
                "synthesis_agent",
                synthesis_request,
                trace,
                progress_callback,
                web=False,
                browser=False,
                local=False,
                output_role="final",
                answer_plan=answer_plan,
                task_id="synthesis",
                enforce_document_evidence=False,
                answer_contract=_answer_contract(request),
                response_language_source=(
                    request.resolved_intent.original_request
                    if request.resolved_intent is not None
                    else request.objective
                ),
            )
            if result.success and is_internal_pipeline_payload(result.content):
                # 原因：弱模型可能把 Evidence/Review JSON 通过 final_answer 成功返回。
                # 作用：用结构化契约识别阻止内部载荷发布，同时保留 raw debug 输出。
                error = "Synthesis returned an internal pipeline payload."
                trace.append(
                    ProcessEvent(
                        phase="synthesis",
                        status="failed",
                        agent="synthesis_agent",
                        message=error,
                    )
                )
                return _CapabilityResult(
                    content="Final synthesis did not complete.",
                    success=False,
                    confidence=0.0,
                    error=error,
                    debug_runs=result.debug_runs,
                )
            final_content = _fallback_from_dependencies_for_empty_answer(
                result.content,
                context,
            )
            return _CapabilityResult(
                content=_preserve_requested_table_output(
                    final_content,
                    request.objective,
                    context,
                ),
                success=result.success,
                confidence=0.95 if result.success else 0.0,
                citations=(
                    _citations_adopted_by_answer(result.content, trusted_citations)
                    if result.success
                    else ()
                ),
                error=result.error,
                debug_runs=result.debug_runs,
            )
        except Exception as exc:  # noqa: BLE001 - preserve completed evidence on synthesis failure.
            trace.append(
                ProcessEvent(
                    phase="synthesis",
                    status="failed",
                    agent="synthesis_agent",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            return _CapabilityResult(
                content="Final synthesis did not complete.",
                success=False,
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _report_capability(
        self,
        request: OrchestrationRequest,
        _question: str,
        context: dict[str, Any],
        trace: list[ProcessEvent],
    ) -> _CapabilityResult:
        multi_agent = context.get("multi_agent", {})
        dependency_results = multi_agent.get("dependency_results", {})
        answer = next(iter(dependency_results.values()), "")
        tables: dict[str, Any] = {}
        citations = [
            citation
            for result in _dependency_capability_results(context)
            for citation in result.citations
        ]
        snapshot = multi_agent.get("shared_state", {})
        for contribution in snapshot.get("contributions", {}).values():
            raw = getattr(contribution, "raw", None)
            if isinstance(raw, _CapabilityResult) and raw.analysis_result is not None:
                tables.update(raw.analysis_result.tables)
        if self.report_generator is None:
            from qwopus_agent.reports import ReportGenerator

            report_generator = ReportGenerator()
        else:
            report_generator = self.report_generator
        started = time.monotonic()
        report = await asyncio.to_thread(
            report_generator.generate,
            title=request.report_title,
            markdown_body=answer,
            tables=tables,
            basename=request.report_basename,
        )
        trace.append(
            ProcessEvent(
                phase="report",
                status="completed",
                agent="report_agent",
                message=f"Generated {len(report.artifacts)} report artifacts.",
                duration_seconds=round(time.monotonic() - started, 3),
            )
        )
        return _CapabilityResult(
            content=answer,
            success=True,
            confidence=1.0,
            citations=_deduplicate_citations(citations),
            report=report,
        )

    async def _guarded(
        self,
        agent_name: str,
        trace: list[ProcessEvent],
        operation: Callable[[], Awaitable[_CapabilityResult]],
    ) -> _CapabilityResult:
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 - one provider must not cancel sibling evidence.
            # 原因：异常文本若沿用默认 success=True，会被 Supervisor 当成有效证据。
            # 作用：发布可审计的真实失败状态；可选依赖降级由 DAG 策略显式处理。
            trace.append(
                ProcessEvent(
                    phase="execution",
                    status="warning",
                    agent=agent_name,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            return _CapabilityResult(
                content="The requested Agent capability was unavailable.",
                success=False,
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )


def _answer_contract(request: OrchestrationRequest) -> AnswerContract:
    """Return one contract even when an older caller did not resolve intent."""
    if request.resolved_intent is not None:
        return request.resolved_intent.answer_contract
    return AnswerContract(response_detail=request.response_detail)


def _is_spreadsheet_only_local_computation_request(
    request: OrchestrationRequest,
) -> bool:
    """Return whether local spreadsheet tools can fully answer the request.

    原因：Excel 计算已由 document_agent 的本地 Tool 完成，
    再进 MiniRAG/Review/Synthesis 会变慢且可能改差。
    作用：纯表格计算问题规划为单 Agent；Web、Browser、Report、全局知识仍走完整编排。
    """
    if (
        not request.uploaded_files
        or request.enable_web_search
        or request.enable_browser
        or request.generate_report
        or request.include_global_knowledge
    ):
        return False
    if not all(_is_spreadsheet_file(file.name) for file in request.uploaded_files):
        return False
    normalized = request.objective.casefold()
    return any(
        term in normalized
        for term in (
            "anova",
            "average",
            "calculate",
            "correlation",
            "covariance",
            "describe",
            "excel",
            "mean",
            "median",
            "outlier",
            "regression",
            "spreadsheet",
            "table",
            "t-test",
            "variance",
            "workbook",
            "z-score",
            "异常",
            "表格",
            "方差",
            "回归",
            "均值",
            "离群",
            "平均",
            "统计",
        )
    )


def _is_spreadsheet_file(name: str) -> bool:
    """Detect spreadsheet files by extension at the orchestration boundary."""
    return Path(name).suffix.casefold() in {".csv", ".xls", ".xlsx"}


def _dependency_capability_results(
    context: dict[str, Any],
) -> tuple[_CapabilityResult, ...]:
    """Read typed dependency artifacts without copying user-facing transcripts."""
    multi_agent = context.get("multi_agent", {})
    # 原因：set 会随机改变并行 Worker 的结果顺序，使引用和 Debug 输出偶发重排。
    # 作用：沿用 Supervisor 记录的依赖顺序，让相同输入产生稳定的 Ledger 与引用顺序。
    dependency_ids = tuple(multi_agent.get("dependency_results", {}))
    contributions = multi_agent.get("shared_state", {}).get("contributions", {})
    results: list[_CapabilityResult] = []
    for task_id in dependency_ids:
        contribution = contributions.get(task_id)
        raw = getattr(contribution, "raw", None)
        if isinstance(raw, _CapabilityResult) and raw.success:
            results.append(raw)
    return tuple(results)


def _preserve_requested_table_output(
    answer: str,
    objective: str,
    context: dict[str, Any],
) -> str:
    """Append verified local computation tables when synthesis drops them.

    原因：多 Agent synthesis 有时把 document_agent 的 Markdown 表格改写成纯文字。
    作用：用户明确要求表格时，最终答案仍保留本地 Tool 产生的可核验计算表。
    """
    if not _requests_table_output(objective) or _contains_markdown_table(answer):
        return answer
    table_block = _verified_table_block_from_dependencies(context)
    if not table_block:
        return answer
    return f"{answer.rstrip()}\n\n{table_block}".strip()


def _fallback_from_dependencies_for_empty_answer(
    answer: str,
    context: dict[str, Any],
) -> str:
    """Use verified dependency prose when synthesis returns only empty scaffolding.

    原因：弱模型偶尔会输出空标题而丢掉已经通过 Tool 验证的实质答案。
    作用：保留 synthesis 的正常结果；只在内容近似为空时回退到成功依赖输出。
    """
    if not _is_empty_scaffold_answer(answer):
        return answer
    fallback = _first_substantive_dependency_answer(context)
    return fallback or answer


def _is_empty_scaffold_answer(answer: str) -> bool:
    """Detect heading-only answers without penalizing short but real answers."""
    substantive_lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip() and not _looks_like_markdown_heading(line.strip())
    ]
    substantive_text = " ".join(
        re.sub(r"[*_`>\\-]", "", line).strip()
        for line in substantive_lines
    ).strip()
    return len(substantive_text) < 40 and "|" not in answer


def _looks_like_markdown_heading(line: str) -> bool:
    """Treat common section labels as structure, not user-facing substance."""
    if line.startswith("#"):
        return True
    return re.fullmatch(r"\*{1,2}[^*\n]{1,80}\*{1,2}", line) is not None


def _first_substantive_dependency_answer(context: dict[str, Any]) -> str:
    """Prefer the first successful worker answer that already contains real content."""
    for result in _dependency_capability_results(context):
        if not _is_empty_scaffold_answer(result.content):
            return result.content
    return ""


def _requests_table_output(objective: str) -> bool:
    """Detect explicit table-output requests without invoking another model."""
    normalized = objective.casefold()
    return any(
        term in normalized
        for term in ("table", "markdown table", "表格", "列表")
    )


def _contains_markdown_table(answer: str) -> bool:
    """Return true when answer already contains a GitHub-Flavored Markdown table."""
    lines = answer.splitlines()
    for index, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        separator = lines[index + 1].strip()
        if "|" in separator and set(separator.replace("|", "").strip()) <= {"-", ":"}:
            return True
    return False


def _verified_table_block_from_dependencies(context: dict[str, Any]) -> str:
    """Use successful capability output as the source of deterministic table blocks."""
    for result in _dependency_capability_results(context):
        content = result.content
        for marker in (
            "## Local calculation table",
            "## 本地计算表格",
            "## Verified local computation",
        ):
            if marker in content:
                return content[content.index(marker):].strip()
        if _contains_markdown_table(content):
            return content.strip()
    return ""


def _evidence_ledger_from_context(
    context: dict[str, Any],
    *,
    max_sources: int,
) -> EvidenceLedger:
    # 原因：完整 Worker 文本会重复占用上下文，并让最终模型继承不同写作风格。
    # 作用：Reviewer 和 Synthesizer 只接收已验证、去重且有长度边界的事实集合。
    packets = tuple(
        result.evidence_packet
        for result in _dependency_capability_results(context)
        if result.evidence_packet is not None
    )
    return build_evidence_ledger(packets, max_sources=max_sources)


def _evidence_review_from_context(context: dict[str, Any]) -> EvidenceReview:
    for result in _dependency_capability_results(context):
        if result.evidence_review is not None:
            return result.evidence_review
    return EvidenceReview(
        resolution="No separate evidence review was required or available."
    )


def _trusted_citations_from_context(
    context: dict[str, Any],
    *,
    max_sources: int,
) -> tuple[SourceCitation, ...]:
    """Collect only citations attached to successful typed dependencies."""
    citations = [
        citation
        for result in _dependency_capability_results(context)
        for citation in result.citations
    ]
    return _deduplicate_citations(citations)[:max_sources]


def _citations_from_chat(run: ChatAgentRun) -> tuple[SourceCitation, ...]:
    """Return answer-adopted sources that also exist in Tool Observations."""
    trusted = _parse_citations("\n".join(run.observations))
    return _citations_adopted_by_answer(run.answer, trusted)


def _citations_adopted_by_answer(
    answer: str,
    trusted: tuple[SourceCitation, ...],
) -> tuple[SourceCitation, ...]:
    """Intersect model citations with exact Tool-grounded source identifiers."""
    declared = _parse_citations(answer)
    declared_urls = {
        citation.url
        for citation in declared
        if citation.kind == "web" and citation.url is not None
    }
    declared_local = {
        citation.source.casefold()
        for citation in declared
        if citation.kind == "local"
    }
    selected: list[SourceCitation] = []
    for citation in trusted:
        if citation.kind == "web":
            if citation.url is not None and citation.url in declared_urls:
                selected.append(citation)
            continue
        exact_source = citation.source.casefold()
        source_mentioned = re.search(
            rf"(?<!\w){re.escape(citation.source)}(?!\w)",
            answer,
            re.I,
        )
        if exact_source in declared_local or source_mentioned is not None:
            selected.append(citation)
    return _deduplicate_citations(selected)


def _file_analysis_citations(
    answer: str,
    analyzed_file_names: list[str],
) -> tuple[SourceCitation, ...]:
    """Return uploaded files named by the final analysis answer."""
    normalized_answer = answer.casefold()
    mentioned = [
        name
        for name in analyzed_file_names
        if Path(name).name.casefold() in normalized_answer
    ]
    # 原因：上传成功只表示文件可用，不表示最终答案实际采用了该文件。
    # 作用：优先展示答案明确引用的来源；旧模型完全不写文件名时仍保留兼容回退。
    cited_names = mentioned or analyzed_file_names
    return tuple(
        SourceCitation(kind="local", source=name)
        for name in cited_names
    )


def _parse_citations(evidence: str) -> tuple[SourceCitation, ...]:
    """Extract normalized web and local citations from one answer or evidence block."""
    citations: list[SourceCitation] = []
    for url in _URL_PATTERN.findall(evidence):
        normalized_url = url.rstrip(".,;:")
        citations.append(
            SourceCitation(
                kind="web",
                source=normalized_url,
                url=normalized_url,
            )
        )
    for match in _SOURCE_PATTERN.finditer(evidence):
        citations.append(SourceCitation(kind="local", source=match.group("source").strip()))
    for match in _EVIDENCE_PATTERN.finditer(evidence):
        citations.append(
            SourceCitation(
                kind="local",
                source=match.group("source").strip(),
                page=(match.group("page") or "").strip() or None,
            )
        )
    return _deduplicate_citations(citations)


def _collect_artifacts(
    run: MultiAgentRun,
) -> tuple[
    tuple[SourceCitation, ...],
    AnalysisResult | None,
    GeneratedReport | None,
    tuple[AgentDebugRun, ...],
]:
    citations: list[SourceCitation] = []
    analysis_result = None
    report = None
    debug_runs: list[AgentDebugRun] = []
    for named_run in run.runs:
        result = named_run.result
        if not isinstance(result, _CapabilityResult):
            continue
        citations.extend(result.citations)
        analysis_result = result.analysis_result or analysis_result
        report = result.report or report
        debug_runs.extend(result.debug_runs)
    return _deduplicate_citations(citations), analysis_result, report, tuple(debug_runs)


def _deduplicate_citations(citations: list[SourceCitation]) -> tuple[SourceCitation, ...]:
    unique: dict[tuple[str, str, str | None, str | None], SourceCitation] = {}
    for citation in citations:
        key = (citation.kind, citation.source, citation.page, citation.url)
        unique[key] = citation
    return tuple(unique.values())


def _is_model_connection_error(exc: BaseException) -> bool:
    """Recognize provider connection failures through wrapped exception chains."""
    current: BaseException | None = exc
    while current is not None:
        text = f"{type(current).__name__}: {current}".casefold()
        if any(
            marker in text
            for marker in (
                "connection error",
                "connecterror",
                "connection refused",
                "host is down",
                "failed to connect",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _model_connection_error_answer(objective: str) -> str:
    """Return a stable user-facing message in the current question's language."""
    if re.search(r"[\u3400-\u9fff]", objective):
        return "模型服务连接已中断。请检查模型地址和服务状态，确认模型在线后重试。"
    return (
        "The model service connection was interrupted. Check the configured model "
        "address and service status, then retry."
    )


def _terminal_failure_answer(objective: str) -> str:
    """Return a safe failure while preserving technical details in debug artifacts."""
    if re.search(r"[\u3400-\u9fff]", objective):
        return "最终回答未能生成。请重试；详细失败原因已保留在调试记录中。"
    return (
        "The final answer could not be generated. Please retry; detailed failure "
        "information remains available in the debug record."
    )


def _append_citations(
    answer: str,
    citations: tuple[SourceCitation, ...],
    objective: str,
) -> str:
    missing = [
        citation
        for citation in citations
        if citation.source not in answer and (citation.url is None or citation.url not in answer)
    ]
    if not missing:
        return answer.strip()
    heading = "来源" if re.search(r"[\u3400-\u9fff]", objective) else "Sources"
    lines = []
    for citation in missing:
        if citation.url:
            lines.append(f"- [{citation.source}]({citation.url})")
        else:
            page = f", page {citation.page}" if citation.page else ""
            lines.append(f"- {citation.source}{page}")
    return f"{answer.strip()}\n\n### {heading}\n\n" + "\n".join(lines)
