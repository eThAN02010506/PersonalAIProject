"""Unified application orchestration for chat, files, research, knowledge, and reports."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qwopus_agent.agents.multi_agent import (
    AgentProfile,
    DelegatedTask,
    DelegationPlan,
    MultiAgentRun,
    MultiAgentSupervisor,
    RunnableAgent,
)
from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    ChatAgentRun,
    SmolagentsModelSettings,
    run_agent_chat_turn_with_debug,
)
from qwopus_agent.services.orchestration_models import (
    OrchestrationRequest,
    OrchestrationResult,
    ProcessEvent,
    SourceCitation,
)

if TYPE_CHECKING:
    from qwopus_agent.analysis import AnalysisResult
    from qwopus_agent.memory import MiniRAG
    from qwopus_agent.reports import GeneratedReport, ReportGenerator
    from qwopus_agent.services.analysis_service import UploadAnalysisOutcome

ProgressCallback = Callable[[str], None]
ChatRunner = Callable[..., ChatAgentRun]
AnalysisRunner = Callable[..., Any]

_URL_PATTERN = re.compile(r"https?://[^\s)\]>]+")
_SOURCE_PATTERN = re.compile(r"\[Source:\s*(?P<source>[^\]\n]+)\]", re.I)
_EVIDENCE_PATTERN = re.compile(
    r"\[(?P<source>[^\],\n]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls))"
    r"(?:,\s*page\s*(?P<page>[^\]]+))?\]",
    re.I,
)


@dataclass(frozen=True)
class _CapabilityResult:
    """Internal result shape understood by the generic Supervisor helpers."""

    content: str
    success: bool = True
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
    chat_runner: ChatRunner = run_agent_chat_turn_with_debug
    analysis_runner: AnalysisRunner | None = None
    report_generator: ReportGenerator | None = None

    async def run(
        self,
        request: OrchestrationRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        """Execute one request through a single fast path or supervised multi-agent path."""
        trace: list[ProcessEvent] = []
        try:
            if not _requires_supervisor(request):
                return await self._run_single(request, trace, progress_callback)
            return await self._run_supervised(request, trace, progress_callback)
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
                route="single_agent" if not _requires_supervisor(request) else "multi_agent",
                trace=tuple(trace),
            )

    def run_sync(
        self,
        request: OrchestrationRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        """Run from synchronous CLI/UI adapters."""
        return asyncio.run(self.run(request, progress_callback=progress_callback))

    async def _run_single(
        self,
        request: OrchestrationRequest,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> OrchestrationResult:
        if request.uploaded_files:
            capability = await self._document_capability(request, trace, progress_callback)
        else:
            capability = await self._chat_capability(
                "chat_agent",
                request,
                trace,
                progress_callback,
                web=request.enable_web_search,
                local=request.enable_local_knowledge,
            )
        answer = _append_citations(capability.content, capability.citations, request.objective)
        return OrchestrationResult(
            success=capability.success,
            final_answer=answer,
            route="single_agent",
            citations=capability.citations,
            trace=tuple(trace),
            analysis_result=capability.analysis_result,
            report=capability.report,
            debug_runs=capability.debug_runs,
        )

    async def _run_supervised(
        self,
        request: OrchestrationRequest,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> OrchestrationResult:
        plan = _build_delegation_plan(request)
        agents = self._build_agents(request, trace, progress_callback)
        profiles = {
            name: AgentProfile(name=name, capabilities=(name.removesuffix("_agent"),))
            for name in agents
        }
        if progress_callback is not None:
            progress_callback("planning")
        trace.append(
            ProcessEvent(
                phase="planning",
                status="completed",
                agent="supervisor",
                message=f"Delegated {len(plan.tasks)} tasks.",
            )
        )
        supervisor = MultiAgentSupervisor(
            agents=agents,
            profiles=profiles,
            max_parallel=3,
            # 原因：生产请求先通过依赖任务和确定性仲裁收敛，默认辩论会额外调用所有模型。
            # 作用：保留 Supervisor 冲突仲裁能力，同时避免每次组合查询产生无必要的延迟。
            debate_rounds=0,
        )
        run = await supervisor.run(
            request.objective,
            context={
                "delegation_plan": plan,
                "shared_state": {"request": request.model_dump(exclude={"uploaded_files"})},
            },
        )
        citations, analysis_result, report, debug_runs = _collect_artifacts(run)
        answer = _append_citations(run.final_answer, citations, request.objective)
        return OrchestrationResult(
            success=bool(answer.strip()) and any(item.success for item in run.runs),
            final_answer=answer,
            route="multi_agent",
            citations=citations,
            trace=tuple(trace),
            analysis_result=analysis_result,
            report=report,
            multi_agent_run=run,
            debug_runs=debug_runs,
        )

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
        local: bool,
    ) -> _CapabilityResult:
        started = time.monotonic()
        trace.append(ProcessEvent(phase="execution", status="started", agent=agent_name))
        history = [turn.model_dump() for turn in request.history]
        run = await asyncio.to_thread(
            self.chat_runner,
            user_message=request.objective,
            history=history,
            settings=self.settings,
            enable_web_search=web,
            enable_local_knowledge=local,
            min_source_relevance=request.min_source_relevance,
            progress_callback=progress_callback,
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
                status="completed",
                agent=agent_name,
                duration_seconds=round(time.monotonic() - started, 3),
            )
        )
        return _CapabilityResult(
            content=run.answer,
            confidence=0.72 if web or local else 0.65,
            citations=citations,
            debug_runs=run.debug_runs,
        )

    async def _document_capability(
        self,
        request: OrchestrationRequest,
        trace: list[ProcessEvent],
        progress_callback: ProgressCallback | None,
    ) -> _CapabilityResult:
        if self.minirag is None:
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

        outcome: UploadAnalysisOutcome = await asyncio.to_thread(
            analysis_runner,
            uploaded_files=[
                UploadedFileInput(name=item.name, content=item.content)
                for item in request.uploaded_files
            ],
            user_question=request.objective,
            settings=self.settings,
            minirag=self.minirag,
            min_source_relevance=request.min_source_relevance,
            selected_sections=request.selected_sections,
            analysis_mode=request.analysis_mode,
        )
        answer = outcome.result.llm_analysis or outcome.result.markdown_summary
        citations = tuple(
            SourceCitation(kind="local", source=name)
            for name in outcome.analyzed_file_names
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
            confidence=0.8,
            citations=citations,
            analysis_result=outcome.result,
            debug_runs=tuple(getattr(outcome, "debug_runs", ())),
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
        evidence = "\n\n".join(
            f"[{task_id}]\n{content}" for task_id, content in dependency_results.items()
        )[:18_000]
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
                "enable_local_knowledge": False,
            }
        )
        try:
            result = await self._chat_capability(
                "synthesis_agent",
                synthesis_request,
                trace,
                progress_callback,
                web=False,
                local=False,
            )
            return _CapabilityResult(content=result.content, confidence=0.95)
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
                confidence=0.75,
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
            # 原因：研究或知识服务可能独立失败，严格失败会跳过依赖它的综合任务。
            # 作用：发布可审计的降级结果，让 Supervisor 继续使用其他成功证据。
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
                confidence=0.05,
                error=str(exc),
            )


def _requires_supervisor(request: OrchestrationRequest) -> bool:
    capabilities = sum(
        (
            bool(request.uploaded_files),
            request.enable_web_search,
            request.enable_local_knowledge,
        )
    )
    return capabilities > 1 or request.generate_report


def _build_delegation_plan(request: OrchestrationRequest) -> DelegationPlan:
    tasks: list[DelegatedTask] = []
    evidence_ids: list[str] = []
    if request.uploaded_files:
        tasks.append(DelegatedTask("document", request.objective, "document_agent"))
        evidence_ids.append("document")
    if request.enable_web_search:
        tasks.append(DelegatedTask("research", request.objective, "research_agent"))
        evidence_ids.append("research")
    if request.enable_local_knowledge:
        dependencies = ("document",) if request.uploaded_files else ()
        tasks.append(
            DelegatedTask(
                "knowledge",
                request.objective,
                "knowledge_agent",
                dependencies,
            )
        )
        evidence_ids.append("knowledge")
    if not evidence_ids:
        tasks.append(DelegatedTask("chat", request.objective, "chat_agent"))
        evidence_ids.append("chat")
    tasks.append(
        DelegatedTask(
            "synthesis",
            request.objective,
            "synthesis_agent",
            tuple(evidence_ids),
        )
    )
    if request.generate_report:
        tasks.append(
            DelegatedTask(
                "report",
                request.objective,
                "report_agent",
                ("synthesis",),
            )
        )
    return DelegationPlan(objective=request.objective, tasks=tuple(tasks))


def _citations_from_chat(run: ChatAgentRun) -> tuple[SourceCitation, ...]:
    citations: list[SourceCitation] = []
    evidence = "\n".join((*run.observations, run.answer))
    for url in _URL_PATTERN.findall(evidence):
        citations.append(SourceCitation(kind="web", source=url, url=url))
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
