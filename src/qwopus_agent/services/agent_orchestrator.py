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
from qwopus_agent.services.orchestration_models import (
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
            )
            plan = await self.planner.plan(
                AgentPlanningRequest(
                    objective=planning_objective,
                    has_documents=bool(request.uploaded_files),
                    enable_web_search=request.enable_web_search,
                    enable_browser=request.enable_browser,
                    enable_local_knowledge=request.enable_local_knowledge,
                    generate_report=request.generate_report,
                )
            )
            agents = self._build_agents(request, trace, progress_callback)
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
            run = await self.executor.execute(
                plan,
                agents=agents,
                profiles=profiles,
                context={
                    "shared_state": {
                        "request": request.model_dump(exclude={"uploaded_files"})
                    }
                },
            )
            citations, analysis_result, report, debug_runs = _collect_artifacts(run)
            terminal_run = next(
                (item for item in run.runs if item.task_id == plan.terminal_task_id),
                None,
            )
            terminal_result = terminal_run.result if terminal_run is not None else None
            answer_content = run.final_answer
            if plan.route == "single_agent" and isinstance(
                terminal_result,
                _CapabilityResult,
            ):
                # 原因：确定性仲裁在单 Agent 失败时只返回通用错误，会丢失安全的拒答说明。
                # 作用：继续把能力层的无证据提示交给用户，同时 success 保持为 False。
                answer_content = terminal_result.content
            answer = _append_citations(answer_content, citations, request.objective)
            return OrchestrationResult(
                # 原因：中间证据或错误文本非空，不代表规划的最终任务已经完成。
                # 作用：整体状态只取决于 DAG 的 terminal task，避免部分结果伪装成成功。
                success=bool(answer.strip()) and bool(
                    terminal_run is not None and terminal_run.success
                ),
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
                final_answer=f"Agent execution failed: {type(exc).__name__}: {exc}",
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
                ),
            )

        agents: dict[str, RunnableAgent] = {}
        if request.uploaded_files:
            agents["document_agent"] = _FunctionAgent(
                lambda _question, _context: self._guarded(
                    "document_agent",
                    trace,
                    lambda: self._document_capability(request, trace, progress_callback),
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
                request, question, context, trace, progress_callback
            )
        )
        agents["synthesis_agent"] = _FunctionAgent(
            lambda question, context: self._synthesis_capability(
                request, question, context, trace, progress_callback
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
                response_detail=request.response_detail,
                knowledge_scope=request.conversation_id,
                knowledge_root=self.knowledge_root,
                global_knowledge_path=self.global_knowledge_path,
                document_evidence_available=bool(request.uploaded_files),
                response_language_source=(
                    request.resolved_intent.original_request
                    if request.resolved_intent is not None
                    else request.objective
                ),
                answer_contract=(
                    request.resolved_intent.answer_contract
                    if request.resolved_intent is not None
                    else None
                ),
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
        citations = _citations_from_chat(run)
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
            confidence=(0.72 if web or browser or local else 0.65) if run.success else 0.0,
            citations=citations,
            error=run.error,
            debug_runs=run.debug_runs,
        )

    async def _document_capability(
        self,
        request: OrchestrationRequest,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
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
        )

    async def _review_capability(
        self,
        request: OrchestrationRequest,
        question: str,
        context: dict[str, Any],
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> _CapabilityResult:
        """Review independent evidence without reopening any Tool."""
        dependency_results = context.get("multi_agent", {}).get("dependency_results", {})
        budget = TokenBudgetManager(
            context_window=self.settings.context_window_tokens,
            output_reserve=min(self.settings.max_tokens, 1200),
        )
        evidence = truncate_to_tokens(
            "\n\n".join(
                f"[{task_id}]\n{content}"
                for task_id, content in dependency_results.items()
            ),
            budget.synthesis_budget,
        )
        review_request = request.model_copy(
            update={
                "objective": (
                    f"Original request: {question}\n\n"
                    f"Independent evidence:\n{evidence}\n\n"
                    "Audit this evidence for the final answering agent. Identify agreements, "
                    "factual conflicts, unsupported claims, and the safest resolution. Do not "
                    "call tools and do not answer the user directly."
                ),
                "resolved_intent": None,
                "history": (),
                "enable_web_search": False,
                "enable_browser": False,
                "enable_local_knowledge": False,
                "include_global_knowledge": False,
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
        )

    async def _synthesis_capability(
        self,
        request: OrchestrationRequest,
        question: str,
        context: dict[str, Any],
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> _CapabilityResult:
        dependency_results = context.get("multi_agent", {}).get("dependency_results", {})
        budget = TokenBudgetManager(
            context_window=self.settings.context_window_tokens,
            output_reserve=self.settings.max_tokens,
        )
        evidence = truncate_to_tokens(
            "\n\n".join(
                f"[{task_id}]\n{content}"
                for task_id, content in dependency_results.items()
            ),
            budget.synthesis_budget,
        )
        synthesis_request = request.model_copy(
            update={
                "objective": (
                    f"Original request: {question}\n\n"
                    f"Agent evidence:\n{evidence}\n\n"
                    "Synthesize one final answer. Resolve conflicts, preserve source citations, "
                    "and do not mention internal agents or execution steps."
                ),
                "history": (),
                "enable_web_search": False,
                "enable_browser": False,
                "enable_local_knowledge": False,
                "include_global_knowledge": False,
            }
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
            )
            return _CapabilityResult(
                content=result.content,
                success=result.success,
                confidence=0.95 if result.success else 0.0,
                error=result.error,
                debug_runs=result.debug_runs,
            )
        except Exception as exc:  # noqa: BLE001 - preserve completed evidence on synthesis failure.
            trace.append(
                ProcessEvent(
                    phase="synthesis",
                    status="warning",
                    agent="synthesis_agent",
                    message=f"Synthesis fallback: {type(exc).__name__}",
                )
            )
            return _CapabilityResult(
                content=evidence or f"No evidence was available: {exc}",
                success=False,
                confidence=0.0,
                error=str(exc),
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
        citations: list[SourceCitation] = []
        snapshot = multi_agent.get("shared_state", {})
        for contribution in snapshot.get("contributions", {}).values():
            raw = getattr(contribution, "raw", None)
            if isinstance(raw, _CapabilityResult):
                citations.extend(raw.citations)
                if raw.analysis_result is not None:
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
                content=f"{agent_name} was unavailable: {type(exc).__name__}: {exc}",
                success=False,
                confidence=0.0,
                error=str(exc),
            )


def _citations_from_chat(run: ChatAgentRun) -> tuple[SourceCitation, ...]:
    """Return sources cited by the final answer, with Observation as a fallback."""
    citations = _parse_citations(run.answer)
    if citations:
        # 原因：一次 RAG 查询会返回多个候选 chunk，但最终答案通常只采用其中一部分。
        # 作用：只展示模型实际引用的来源，避免把未使用的 Observation 全部追加给用户。
        return citations
    observation_citations = _parse_citations("\n".join(run.observations))
    mentioned_local = tuple(
        citation
        for citation in observation_citations
        if citation.kind == "local"
        and Path(citation.source).name.casefold() in run.answer.casefold()
    )
    if mentioned_local:
        # 原因：部分模型会直接写出文件名，却不遵循结构化 Source 标记。
        # 作用：仍能识别真实采用的本地文档，同时排除 Observation 内未使用的 URL。
        return mentioned_local
    return observation_citations


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
