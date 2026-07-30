"""smolagents runtime integration for Qwopus-Agent.

This module connects Qwopus-Agent with an OpenAI-compatible local LLM server
such as optiq serve / mlx_lm.server.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qwopus_agent.integrations import (
    smolagents_debug,
    smolagents_knowledge,
    smolagents_model,
)
from qwopus_agent.integrations.skill_tools import (
    build_promoted_workflow_tools,
    build_registered_skill_tools,
)
from qwopus_agent.memory import DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
from qwopus_agent.prompts import smolagents as smolagents_prompts
from qwopus_agent.reports import contract as report_contract
from qwopus_agent.reports import grounded
from qwopus_agent.services.answer_quality import (
    AnswerQualityEvaluator,
    strip_unsupported_evidence_lines,
)
from qwopus_agent.services.orchestration_models import (
    AgentOutputRole,
    AnswerContract,
    AnswerPlan,
)
from qwopus_agent.skills import SkillRegistry, WorkflowSpec
from qwopus_agent.utils.token_budget import (
    TokenBudgetManager,
    truncate_to_tokens,
)

AgentDebugRun = smolagents_debug.AgentDebugRun
SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings
check_model_connection = smolagents_model.check_model_connection
resolve_model_settings = smolagents_model.resolve_model_settings
ChatMessage = smolagents_prompts.ChatMessage
build_chat_messages = smolagents_prompts.build_chat_messages
format_agent_chat_prompt = smolagents_prompts.format_agent_chat_prompt
format_chat_prompt = smolagents_prompts.format_chat_prompt
LocalKnowledgeTools = smolagents_knowledge.LocalKnowledgeTools
build_local_knowledge_tools = smolagents_knowledge.build_local_knowledge_tools
build_browser_open_tool = smolagents_knowledge.build_browser_open_tool
build_tavily_search_tool = smolagents_knowledge.build_tavily_search_tool

_agent_debug_steps = smolagents_debug.agent_debug_steps
_build_agent_debug_run = smolagents_debug.build_agent_debug_run
_extract_collection_covered_file_names = (
    smolagents_debug.extract_collection_covered_file_names
)
_extract_agent_observations = smolagents_debug.extract_agent_observations
_extract_agent_tool_calls = smolagents_debug.extract_agent_tool_calls
_extract_final_answer = smolagents_debug.extract_final_answer
_extract_inspected_file_names = smolagents_debug.extract_inspected_file_names
_looks_like_tool_observation = smolagents_debug.looks_like_tool_observation
_required_file_tools = smolagents_debug.required_file_tools
_unpack_agent_run_result = smolagents_debug.unpack_agent_run_result
_document_evidence_required_answer = (
    smolagents_prompts.document_evidence_required_answer
)
_has_usable_knowledge_evidence = smolagents_prompts.has_usable_knowledge_evidence
_LOCAL_KNOWLEDGE_TOOLS = smolagents_prompts.LOCAL_KNOWLEDGE_TOOLS
_no_knowledge_evidence_answer = smolagents_prompts.no_knowledge_evidence_answer
_requires_document_evidence = smolagents_prompts.requires_document_evidence
_QUALITY_REPAIR_INSTRUCTIONS = {
    "insufficient_depth": (
        "develop every relevant requirement with distinct supporting detail"
    ),
    "insufficient_specificity": (
        "replace generic claims with concrete evidence, causal explanation, examples, "
        "conditions, verification, or limitations"
    ),
    "missing_ordered_steps": "add ordered, verifiable steps and failure recovery",
    "missing_comparison": (
        "compare the options under shared criteria and state when each should be chosen"
    ),
    "missing_analysis_structure": (
        "separate the main finding, supporting analysis, risks, and implications"
    ),
    "fragmented_answer": (
        "connect the evidence into a coherent argument instead of a bullet stack"
    ),
    "missing_source_attribution": "cite only the supplied source names, pages, or URLs",
    "ungrounded_source_reference": (
        "remove every file name or URL that does not appear in supplied tool evidence"
    ),
    "unsupported_evidence_framing": (
        "remove evidence headings, case studies, and claims that documents prove something "
        "when no tool evidence was supplied; present only design reasoning"
    ),
    "unsupported_empirical_claims": (
        "remove every percentage, benchmark, study, experiment, report, publication, and "
        "measured-improvement claim that has no supplied source; replace it with design "
        "reasoning or a future verification method"
    ),
}
_apply_grounded_report_fallbacks = (
    report_contract._apply_grounded_report_fallbacks
)
_collection_grounding_evidence = report_contract._collection_grounding_evidence
_is_model_generation_failure_output = (
    report_contract._is_model_generation_failure_output
)
_lesson_slot_manifest = report_contract._lesson_slot_manifest
_merge_numbered_section_refinement = (
    report_contract._merge_numbered_section_refinement
)
_missing_requested_sections = report_contract._missing_requested_sections
_report_quality_issues = report_contract._report_quality_issues

_ALL_SOURCE_REQUEST_PATTERN = grounded._ALL_SOURCE_REQUEST_PATTERN
_SCRIPTURE_REFERENCE_PATTERN = grounded._SCRIPTURE_REFERENCE_PATTERN
_LessonGroundingSpec = grounded._LessonGroundingSpec
_canonical_lesson_heading = grounded._canonical_lesson_heading
_chinese_integer = grounded._chinese_integer
_collection_manifest_sources = grounded._collection_manifest_sources
_collection_source_blocks = grounded._collection_source_blocks
_grounded_application_claim = grounded._grounded_application_claim
_grounded_evidence_claim = grounded._grounded_evidence_claim
_lesson_answer_aliases = grounded._lesson_answer_aliases
_lesson_answer_label = grounded._lesson_answer_label
_lesson_evidence = grounded._lesson_evidence
_lesson_grounding_specs = grounded._lesson_grounding_specs
_lesson_number_from_label = grounded._lesson_number_from_label
_lesson_scripture = grounded._lesson_scripture
_lesson_topic = grounded._lesson_topic
_normalized_fact_text = grounded._normalized_fact_text
_render_deterministic_grounded_report = (
    grounded._render_deterministic_grounded_report
)
_render_grounded_checklist = grounded._render_grounded_checklist
_render_grounded_draft_review = grounded._render_grounded_draft_review
_render_grounded_examples = grounded._render_grounded_examples
_render_grounded_full_draft = grounded._render_grounded_full_draft
_render_grounded_lesson_fallback = (
    grounded._render_grounded_lesson_fallback
)
_render_grounded_outline = grounded._render_grounded_outline
_render_grounded_paragraph_guidance = (
    grounded._render_grounded_paragraph_guidance
)
_render_grounded_source_inventory = (
    grounded._render_grounded_source_inventory
)
_render_grounded_strategy = grounded._render_grounded_strategy
_render_grounded_understanding = grounded._render_grounded_understanding
_requested_numbered_sections = grounded._requested_numbered_sections
_scripture_reference_is_supported = (
    grounded._scripture_reference_is_supported
)
_scripture_reference_key = grounded._scripture_reference_key
_source_answer_label = grounded._source_answer_label
_source_application_excerpt = grounded._source_application_excerpt
_source_evidence_excerpt = grounded._source_evidence_excerpt
_source_fact_values = grounded._source_fact_values
_source_tagged_excerpt = grounded._source_tagged_excerpt
_title_is_source_understanding = grounded._title_is_source_understanding
_title_requires_full_draft = grounded._title_requires_full_draft
_topic_payload = grounded._topic_payload
_validated_grounded_collection = grounded._validated_grounded_collection
should_use_grounded_report_composer = (
    grounded.should_use_grounded_report_composer
)


class SmolagentsDependencyError(RuntimeError):
    """Raised when smolagents is required but missing."""


@dataclass(frozen=True)
class DocumentAnalysisRun:
    """Document analysis answer with a UI-visible debug trace."""

    answer: str

    debug_steps: list[str]

    tool_calls: list[str] = field(default_factory=list)

    inspected_file_names: tuple[str, ...] = ()

    debug_runs: tuple[AgentDebugRun, ...] = ()

    generation_mode: str = "model"


@dataclass(frozen=True)
class ChatAgentRun:
    """Safe structured result from one smolagents chat run."""

    answer: str
    tool_calls: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    state: str | None = None
    debug_runs: tuple[AgentDebugRun, ...] = ()
    success: bool = True
    error: str | None = None


def build_smolagents_model(settings: SmolagentsModelSettings | None = None) -> Any:
    settings = settings or SmolagentsModelSettings.from_env()

    try:
        from smolagents import OpenAIModel

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("smolagents is not installed") from exc

    return OpenAIModel(
        model_id=settings.model_id,
        api_base=settings.base_url,
        api_key=settings.api_key,
        client_kwargs={
            "timeout": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        },
        temperature=settings.temperature,
        # 原因：smolagents Agent.run 不会把应用层 max_tokens 自动传给模型。
        # 作用：长文档报告使用显式输出预算，不再由兼容服务的短默认值截成“略”。
        max_tokens=settings.max_tokens,
    )


def build_smolagents_code_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
    final_answer_checks: list[Callable[..., bool]] | None = None,
) -> Any:
    try:
        from smolagents import CodeAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

    model = build_smolagents_model(settings)

    return CodeAgent(
        tools=tools or [],
        model=model,
        # 原因：聊天 Tool 不需要任意文件、进程或 shell 访问；授权 os/subprocess 会扩大风险。
        # 作用：Code 兼容模式只能组合已注册 Tool，数据计算继续使用独立 pandas 沙箱。
        additional_authorized_imports=[],
        final_answer_checks=final_answer_checks,
    )


def build_smolagents_tool_calling_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
    final_answer_checks: list[Callable[..., bool]] | None = None,
) -> Any:
    """Build the smolagents Agent runtime used as Qwopus' chat driver."""
    try:
        from smolagents import ToolCallingAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

    settings = settings or SmolagentsModelSettings.from_env()
    if settings.capabilities.agent_mode == "code":
        return build_smolagents_code_agent(
            settings=settings,
            tools=tools,
            final_answer_checks=final_answer_checks,
        )

    model = build_smolagents_model(settings)
    # 原因：smolagents 是整体 Agent 驱动入口，工具选择应由 Agent runtime 处理。
    # 作用：Streamlit 不再手动先搜索再拼 prompt，而是把受控 Tool 交给 Agent。
    return ToolCallingAgent(
        tools=tools or [],
        model=model,
        final_answer_checks=final_answer_checks,
    )


def run_smolagents_smoke_test(
    prompt: str,
    settings: SmolagentsModelSettings | None = None,
) -> str:
    agent = build_smolagents_code_agent(
        settings=settings,
        tools=[],
    )

    return str(agent.run(prompt))


def run_agent_chat_turn(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
    enable_web_search: bool = False,
    enable_browser: bool = False,
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    min_source_relevance: float = 0.55,
    max_evidence_sources: int = 12,
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    knowledge_scope: str | None = None,
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    global_knowledge_path: Path | None = None,
    document_evidence_available: bool = False,
    enforce_document_evidence: bool = True,
    response_language_source: str | None = None,
    answer_contract: AnswerContract | None = None,
    output_role: AgentOutputRole = "final",
    answer_plan: AnswerPlan | None = None,
    promoted_workflows: tuple[WorkflowSpec, ...] = (),
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Run one chat turn through smolagents as the Agent driver."""
    return run_agent_chat_turn_with_debug(
        user_message=user_message,
        history=history,
        settings=settings,
        enable_web_search=enable_web_search,
        enable_browser=enable_browser,
        enable_local_knowledge=enable_local_knowledge,
        include_global_knowledge=include_global_knowledge,
        min_source_relevance=min_source_relevance,
        max_evidence_sources=max_evidence_sources,
        response_detail=response_detail,
        knowledge_scope=knowledge_scope,
        knowledge_root=knowledge_root,
        global_knowledge_path=global_knowledge_path,
        document_evidence_available=document_evidence_available,
        enforce_document_evidence=enforce_document_evidence,
        response_language_source=response_language_source,
        answer_contract=answer_contract,
        output_role=output_role,
        answer_plan=answer_plan,
        promoted_workflows=promoted_workflows,
        progress_callback=progress_callback,
    ).answer


def run_agent_chat_turn_with_debug(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
    enable_web_search: bool = False,
    enable_browser: bool = False,
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    min_source_relevance: float = 0.55,
    max_evidence_sources: int = 12,
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    knowledge_scope: str | None = None,
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    global_knowledge_path: Path | None = None,
    document_evidence_available: bool = False,
    enforce_document_evidence: bool = True,
    response_language_source: str | None = None,
    answer_contract: AnswerContract | None = None,
    output_role: AgentOutputRole = "final",
    answer_plan: AnswerPlan | None = None,
    promoted_workflows: tuple[WorkflowSpec, ...] = (),
    progress_callback: Callable[[str], None] | None = None,
) -> ChatAgentRun:
    """Run chat and retain only the safe Tool metadata needed by orchestration."""
    effective_settings = settings or SmolagentsModelSettings.from_env()
    budget = TokenBudgetManager(
        context_window=effective_settings.context_window_tokens,
        output_reserve=effective_settings.max_tokens,
    )
    # 原因：普通独立 Skill 已由 Registry 扫描，不应再要求维护一份 smolagents 工具清单。
    # 作用：默认只自动装配无敏感权限的 Skill；Web、Knowledge、Browser 仍由本轮授权控制。
    tools: list[Any] = build_registered_skill_tools(
        SkillRegistry.discover(),
        enabled_permissions={"always"},
        max_output_tokens=budget.observation_budget,
        progress_callback=progress_callback,
    )
    if include_global_knowledge and not enable_local_knowledge:
        raise ValueError("Global knowledge requires local knowledge permission.")
    if enable_web_search:
        tools.append(
            build_tavily_search_tool(
                progress_callback=progress_callback,
                max_results=max_evidence_sources,
            )
        )
    if enable_browser:
        tools.append(
            build_browser_open_tool(
                progress_callback=progress_callback,
                max_output_tokens=budget.observation_budget,
            )
        )
    knowledge_tools: list[Any] = []
    private_sources: tuple[str, ...] | None = None
    knowledge_primary_scope: Literal["private", "global", "none"] = "none"
    if enable_local_knowledge:
        if not knowledge_scope:
            raise ValueError("knowledge_scope is required when local knowledge is enabled")
        knowledge_tools = build_local_knowledge_tools(
            knowledge_scope,
            user_message=user_message,
            progress_callback=progress_callback,
            min_source_relevance=min_source_relevance,
            max_results=max_evidence_sources,
            knowledge_root=knowledge_root,
            global_knowledge_path=global_knowledge_path,
            include_global_knowledge=include_global_knowledge,
            budget_manager=budget,
        )
        # 原因：测试/第三方注入的旧工厂可能仍返回普通 list；生产工厂返回带来源清单的兼容子类。
        # 作用：已知清单用于确定性预检，未知清单保持旧扩展点行为而不误拒绝请求。
        private_sources = getattr(knowledge_tools, "private_sources", None)
        knowledge_primary_scope = getattr(
            knowledge_tools,
            "primary_scope",
            "private" if knowledge_tools else "none",
        )
        tools.extend(knowledge_tools)

    promoted_tools = (
        build_promoted_workflow_tools(
            promoted_workflows,
            tools,
            max_output_tokens=budget.observation_budget,
        )
        if promoted_workflows
        else []
    )
    # 原因：已晋升工作流只有进入 smolagents 的本轮 Tool 列表才是真正可复用能力。
    # 作用：工作流复用现有授权 Tool；缺少 Web/Knowledge 权限时适配器会拒绝装配。
    tools.extend(promoted_tools)
    promoted_local_tools = {
        spec.name
        for spec in promoted_workflows
        if any(
            step.skill_name in {"rag_search", "graph_search"}
            for step in spec.steps
        )
        and any(getattr(tool, "name", None) == spec.name for tool in promoted_tools)
    }

    accessible_document_evidence = (
        document_evidence_available
        or bool(private_sources)
        or (private_sources is None and bool(knowledge_tools))
        or (enable_local_knowledge and include_global_knowledge)
    )
    # 原因：Review/Synthesis 已消费上游 ledger，不应把内部提示重新当作用户附件请求。
    # 作用：用户入口默认继续执行证据预检，只有明确的内部阶段调用可以跳过。
    if (
        enforce_document_evidence
        and _requires_document_evidence(user_message)
        and not accessible_document_evidence
    ):
        # 原因：提示模型“不要编造”仍会让无 Tool 聊天根据历史或常识生成貌似完整的文件分析。
        # 作用：在 Agent/模型构造前检查本轮实际可访问的来源，缺证据时返回稳定失败。
        if progress_callback is not None:
            progress_callback("completed")
        return ChatAgentRun(
            answer=_document_evidence_required_answer(user_message),
            state="preflight_rejected",
            success=False,
            error=(
                "Document evidence is required, but this turn has no uploaded attachment, "
                "conversation source, or authorized global knowledge."
            ),
        )

    # 原因：部分兼容模型只知道 final_answer_check 失败，却看不到具体问题，会原样重复短答。
    # 作用：首轮不挂布尔检查，应用层随后携带明确 issues 进行最多一次无工具修正。
    agent = build_smolagents_tool_calling_agent(
        settings=effective_settings,
        tools=tools,
        final_answer_checks=[],
    )
    prompt = format_agent_chat_prompt(
        history=history,
        user_message=user_message,
        enable_web_search=enable_web_search,
        enable_browser=enable_browser,
        enable_local_knowledge=enable_local_knowledge,
        include_global_knowledge=include_global_knowledge,
        knowledge_primary_scope=knowledge_primary_scope,
        history_max_tokens=budget.history_budget,
        response_detail=response_detail,
        response_language_source=response_language_source,
        answer_contract=answer_contract,
        output_role=output_role,
        answer_plan=answer_plan,
    )
    if progress_callback is not None:
        progress_callback("planning")
    # 原因：部分模型会忽略提示并重复调用已经成功的检索 Tool，直到耗尽较大的步数上限。
    # 作用：每类获准能力最多预留一次调用，再留一步生成最终答案。
    capability_count = sum((enable_web_search, enable_browser, enable_local_knowledge))
    max_steps = (
        2
        + max(0, capability_count - 1)
        + int(
            include_global_knowledge
            and knowledge_primary_scope == "private"
        )
    )
    run_max_steps = max_steps if tools else 2
    run_result = agent.run(
        prompt,
        max_steps=run_max_steps,
        return_full_result=True,
    )
    answer, state, steps = _unpack_agent_run_result(run_result)
    debug_runs = [
        _build_agent_debug_run(
            label="chat",
            prompt=prompt,
            max_steps=run_max_steps,
            state=state,
            output=answer,
            steps=steps,
        )
    ]
    tool_calls = _extract_agent_tool_calls(steps)
    observations = _extract_agent_observations(steps)
    final_answer = _extract_final_answer(answer)
    has_grounded_sources = _answer_has_grounded_source(final_answer, observations)
    quality_issues = (
        _answer_quality_issues(
            final_answer,
            answer_contract,
            answer_plan,
            has_citations=has_grounded_sources,
        )
        if output_role == "final"
        else ()
    )

    local_tool_used = bool(
        (_LOCAL_KNOWLEDGE_TOOLS | promoted_local_tools).intersection(tool_calls)
    )
    if (
        enable_local_knowledge
        and not enable_web_search
        and (
            not local_tool_used
            or not _has_usable_knowledge_evidence(observations)
        )
    ):
        # 原因：知识专用请求在零命中后交给无工具 finalizer，会退化成模型常识回答。
        # 作用：保留原始 Tool 审计信息，但明确返回失败且不再进行任何开放式生成。
        if progress_callback is not None:
            progress_callback("completed")
        error = "No relevant local knowledge evidence was found for this request."
        return ChatAgentRun(
            answer=_no_knowledge_evidence_answer(user_message),
            tool_calls=tuple(dict.fromkeys(tool_calls)),
            observations=tuple(dict.fromkeys(observations)),
            state=state,
            debug_runs=tuple(debug_runs),
            success=False,
            error=error,
        )
    needs_refinement = (
        not final_answer
        or _looks_like_tool_observation(final_answer)
        or state == "max_steps_error"
        or (local_tool_used and len(final_answer) < 80)
        or bool(quality_issues)
    )
    if needs_refinement:
        evidence = "\n\n".join(
            (
                *observations,
                f"Previous draft:\n{final_answer}" if final_answer else "",
            )
        ).strip() or "No usable tool evidence."
        # 原因：继续复用带 Tool 的 Agent 仍可能无视提示并再次检索，造成长时间循环。
        # 作用：把已取得的 Observation 交给无工具 finalizer，只允许它生成最终自然语言答案。
        finalizer = build_smolagents_tool_calling_agent(
            settings=effective_settings,
            tools=[],
        )
        retry_prompt = _role_refinement_prompt(
            output_role=output_role,
            user_message=user_message,
            evidence=truncate_to_tokens(evidence, budget.synthesis_budget),
            quality_issues=quality_issues,
        )
        retry_result = finalizer.run(
            retry_prompt,
            max_steps=2,
            return_full_result=True,
        )
        retry_answer, retry_state, retry_steps = _unpack_agent_run_result(retry_result)
        debug_runs.append(
            _build_agent_debug_run(
                label="chat_finalizer",
                prompt=retry_prompt,
                max_steps=2,
                state=retry_state,
                output=retry_answer,
                steps=retry_steps,
            )
        )
        tool_calls.extend(_extract_agent_tool_calls(retry_steps))
        observations.extend(_extract_agent_observations(retry_steps))
        state = retry_state or state
        final_answer = _extract_final_answer(retry_answer)
        if output_role == "final":
            has_grounded_sources = _answer_has_grounded_source(
                final_answer,
                observations,
            )
            remaining_issues = _answer_quality_issues(
                final_answer,
                answer_contract,
                answer_plan,
                has_citations=has_grounded_sources,
            )
            if (
                {
                    "unsupported_empirical_claims",
                    "unsupported_evidence_framing",
                    "ungrounded_source_reference",
                }.intersection(remaining_issues)
                and not has_grounded_sources
            ):
                final_answer = strip_unsupported_evidence_lines(final_answer)

    if progress_callback is not None:
        progress_callback("completed")
    if not final_answer or _looks_like_tool_observation(final_answer):
        raise RuntimeError("smolagents did not produce a final chat answer after tool execution.")
    # 原因：正式界面只应看到安全结果，但本地 Debug Console 需要复现模型与 Tool 交互。
    # 作用：安全字段继续供业务编排使用，原始步骤放入独立 debug_runs，API 不序列化它。
    return ChatAgentRun(
        answer=final_answer,
        tool_calls=tuple(dict.fromkeys(tool_calls)),
        observations=tuple(dict.fromkeys(observations)),
        state=state,
        debug_runs=tuple(debug_runs),
        success=True,
    )


def _role_refinement_prompt(
    *,
    output_role: AgentOutputRole,
    user_message: str,
    evidence: str,
    quality_issues: tuple[str, ...] = (),
) -> str:
    """Finish a stalled Tool run without changing its orchestration role."""
    common = (
        f"Original task:\n{user_message}\n\n"
        f"Available tool evidence:\n{evidence}\n\n"
    )
    if output_role == "evidence":
        return common + (
            "Convert only this evidence into one JSON object and return it through final_answer. "
            'Use exactly: {"facts":[{"claim":"...","support":"...","sources":["..."],'
            '"confidence":"low|medium|high","plan_item_ids":["P1"]}],'
            '"limitations":["..."]}. '
            "Do not write a user-facing answer or expose Observation, Thought, logs, or drafts."
        )
    if output_role == "review":
        return common + (
            "Review only this evidence and return one JSON object through final_answer. Use "
            'exactly: {"agreements":["..."],"conflicts":["..."],'
            '"unsupported_claims":["..."],"gaps":["..."],"resolution":"...",'
            '"coverage":[{"plan_item_id":"P1","status":"supported|partial|missing|conflicted",'
            '"finding":"..."}]}. '
            "Do not write a user-facing answer or expose Thought or drafts."
        )
    issue_instruction = (
        "Correct these detected answer defects exactly once:\n"
        + "\n".join(
            f"- {issue}: {_QUALITY_REPAIR_INSTRUCTIONS.get(issue, 'correct this defect')}"
            for issue in quality_issues
        )
        + "\n\n"
        if quality_issues
        else ""
    )
    return common + issue_instruction + (
        "Answer the original user question now. Return only a complete natural-language final "
        "answer in the user's language. State the conclusion, explain the relevant relationship "
        "or evidence, and explicitly cite every available local source file and page. Never "
        "invent citations, measurements, or study results. If no source evidence is available, "
        "frame claims as architectural reasoning. Never expose Observation, Thought, tool logs, "
        "or drafts."
    )


def _build_answer_quality_checks(
    contract: AnswerContract | None,
) -> list[Callable[..., bool]]:
    """Build a request-scoped smolagents final-answer validator."""
    if (
        contract is None
        or contract.response_detail == "concise"
        or contract.complexity == "simple"
    ):
        return []
    evaluator = AnswerQualityEvaluator()

    def qwopus_answer_quality(
        final_answer: Any,
        _memory: Any,
        *,
        agent: Any = None,
    ) -> bool:
        del agent
        answer = str(final_answer)
        report = evaluator.evaluate(
            answer,
            contract,
            has_citations=_answer_contains_source(answer),
        )
        return report.passed

    return [qwopus_answer_quality]


def _answer_quality_issues(
    answer: str,
    contract: AnswerContract | None,
    answer_plan: AnswerPlan | None = None,
    *,
    has_citations: bool | None = None,
) -> tuple[str, ...]:
    if (
        not answer
        or contract is None
        or contract.response_detail == "concise"
        or contract.complexity == "simple"
    ):
        return ()
    return AnswerQualityEvaluator().evaluate(
        answer,
        contract,
        has_citations=(
            _answer_contains_source(answer)
            if has_citations is None
            else has_citations
        ),
        answer_plan=answer_plan,
    ).issues


def _answer_contains_source(answer: str) -> bool:
    return bool(
        re.search(r"https?://", answer, re.I)
        or re.search(
            r"\b[^\s/\\]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls)\b",
            answer,
            re.I,
        )
    )


def _answer_has_grounded_source(
    answer: str,
    observations: list[str],
) -> bool:
    """Require a source token to appear in both the answer and Tool evidence."""
    answer_sources = _source_tokens(answer)
    observation_sources = _source_tokens("\n".join(observations))
    return bool(answer_sources.intersection(observation_sources))


def _source_tokens(content: str) -> set[str]:
    return {
        match.casefold()
        for match in re.findall(
            r"https?://[^\s)\]>`\"']+|"
            r"\b[^\s/\\]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls)\b",
            content,
            re.I,
        )
    }


def run_smolagents_file_analysis_with_debug(
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    tools: list[Any],
    settings: SmolagentsModelSettings | None = None,
    analysis_mode: str = "question",
) -> DocumentAnalysisRun:
    """Run uploaded-file analysis through the smolagents ToolCallingAgent."""
    if not file_names:
        raise ValueError("file_names must not be empty.")
    if not tools:
        raise ValueError("At least one file-analysis tool is required.")

    effective_settings = settings or SmolagentsModelSettings.from_env()
    budget = TokenBudgetManager(
        context_window=effective_settings.context_window_tokens,
        output_reserve=effective_settings.max_tokens,
    )
    collection_tools = [
        tool
        for tool in tools
        if getattr(tool, "name", "") == "document_collection_summary"
    ]
    if len(collection_tools) > 1:
        raise ValueError("Only one document_collection_summary tool is allowed.")
    has_collection_summary = bool(collection_tools)
    parser_files = set(file_names).difference(spreadsheet_names)
    requires_collection_summary = _requires_collection_summary(
        available=has_collection_summary,
        file_count=len(parser_files),
        user_question=user_question,
        analysis_mode=analysis_mode,
    )
    requested_sections = _requested_numbered_sections(user_question)
    prompt = format_file_analysis_agent_prompt(
        file_names=file_names,
        spreadsheet_names=spreadsheet_names,
        user_question=user_question,
        analysis_mode=analysis_mode,
        has_collection_summary=has_collection_summary,
    )
    collection_tool = collection_tools[0] if collection_tools else None
    use_grounded_report_composer = should_use_grounded_report_composer(
        file_names=file_names,
        spreadsheet_names=spreadsheet_names,
        user_question=user_question,
        has_collection_summary=collection_tool is not None,
    )
    if use_grounded_report_composer:
        assert collection_tool is not None
        # 原因：一次 8k-token 长生成在本地大模型切换或显存紧张时会超时/重启，
        # 而且弱模型容易把相邻课程压成一个泛化清单。
        # 作用：完整报告先由 collection Tool 确定性覆盖全部来源，再按逐课事实槽位组装；
        # 这一完整性优先路径不依赖模型长生成，也不会使用文件外知识。
        collection_evidence = str(collection_tool.forward())
        lesson_specs = _validated_grounded_collection(
            file_names=file_names,
            collection_evidence=collection_evidence,
        )
        final_answer = _render_deterministic_grounded_report(
            requested=requested_sections,
            file_names=file_names,
            collection_evidence=collection_evidence,
            lesson_specs=lesson_specs,
        )
        missing_sections = _missing_requested_sections(
            final_answer,
            requested_sections,
        )
        quality_issues = _report_quality_issues(
            answer=final_answer,
            requested=requested_sections,
            file_names=file_names,
            user_question=user_question,
            collection_evidence=collection_evidence,
        )
        if missing_sections or quality_issues:
            raise RuntimeError(
                "Grounded report composer produced an invalid report contract."
            )
        debug_steps = [
            "已用 document_collection_summary 读取并核对全部来源。",
            "长篇全来源任务使用逐来源、逐课确定性报告合成，未调用不稳定的单次长生成。",
        ]
        return DocumentAnalysisRun(
            answer=final_answer,
            debug_steps=debug_steps,
            tool_calls=["document_collection_summary"],
            inspected_file_names=tuple(file_names),
            debug_runs=(
                _build_agent_debug_run(
                    label="grounded_report_composer",
                    prompt=prompt,
                    max_steps=0,
                    state="success",
                    output=final_answer,
                    steps=[],
                ),
            ),
            generation_mode="grounded_composer",
        )

    agent = build_smolagents_tool_calling_agent(
        settings=effective_settings,
        tools=tools,
    )
    # 原因：固定至少八步会让不遵循提示的模型反复读取同一文件，显著增加等待时间。
    # 作用：按文件数提供一次读取和收尾预算，遗漏文件由下方精确校验触发补充轮。
    max_steps = (
        4
        if requires_collection_summary
        else min(max(len(file_names) + 2, 4), 12)
    )
    # 原因：上传分析需要由 smolagents 自己选择解析、RAG 或 Excel 沙箱工具。
    # 作用：返回完整运行状态供可选调试区审计，主界面仍只使用最终 answer。
    run_result = agent.run(
        prompt,
        max_steps=max_steps,
        return_full_result=True,
    )
    answer, state, steps = _unpack_agent_run_result(run_result)
    debug_runs = [
        _build_agent_debug_run(
            label="file_analysis",
            prompt=prompt,
            max_steps=max_steps,
            state=state,
            output=answer,
            steps=steps,
        )
    ]
    tool_calls = _extract_agent_tool_calls(steps)
    all_steps = list(steps)
    debug_steps = _agent_debug_steps(state=state, steps=steps, tool_calls=tool_calls)
    required_tools = _required_file_tools(
        spreadsheet_names=spreadsheet_names,
    )
    if requires_collection_summary:
        # 原因：逐文件调用受 Agent 步数限制，提示语不能保证大型文档集合真的全部进入上下文。
        # 作用：多文档任务必须执行一次带 coverage manifest 的平衡证据 Tool。
        required_tools.add("document_collection_summary")
    missing_tools = required_tools.difference(tool_calls)
    inspected_files = _extract_inspected_file_names(steps)
    inspected_files.update(_extract_collection_covered_file_names(steps))
    missing_files = set(file_names).difference(inspected_files)

    final_answer = _extract_final_answer(answer)
    if _is_model_generation_failure_output(final_answer):
        # 原因：smolagents 在最终模型请求失败时会把错误包装成普通 text content。
        # 作用：保留完整 debug run 和已检查来源，但不让 transport error 成为用户答案。
        debug_steps.append("模型最终答案生成失败；错误详情仅保留在 Debug Console。")
        return DocumentAnalysisRun(
            answer="",
            debug_steps=debug_steps,
            tool_calls=list(dict.fromkeys(tool_calls)),
            inspected_file_names=tuple(
                file_name for file_name in file_names if file_name in inspected_files
            ),
            debug_runs=tuple(debug_runs),
        )
    missing_sections = _missing_requested_sections(
        final_answer,
        requested_sections,
    )
    collection_evidence = _collection_grounding_evidence(all_steps)
    lesson_specs = _lesson_grounding_specs(file_names, collection_evidence)
    quality_issues = _report_quality_issues(
        answer=final_answer,
        requested=requested_sections,
        file_names=file_names,
        user_question=user_question,
        collection_evidence=collection_evidence,
    )
    refinement_numbers = set(missing_sections).union(quality_issues)
    refinement_sections = {
        number: title
        for number, title in requested_sections.items()
        if number in refinement_numbers
    }
    if (
        not final_answer
        or _looks_like_tool_observation(final_answer)
        or missing_tools
        or missing_files
        or refinement_sections
    ):
        # 原因：少数模型会把最后一次 Tool Observation 当作回答，或者在步数内没有调用 final_answer。
        # 作用：保留同一个 Agent memory 再收敛一轮，禁止原始工具输出进入 Streamlit 主结果。
        section_only_refinement = bool(
            final_answer
            and not _looks_like_tool_observation(final_answer)
            and not missing_tools
            and not missing_files
            and refinement_sections
        )
        if missing_tools:
            missing_names = ", ".join(sorted(missing_tools))
            debug_steps.append(f"Agent 尚未调用必要 Tool：{missing_names}；触发补充执行。")
        elif section_only_refinement:
            debug_steps.append(
                "Agent 报告仅有部分章节缺失或不足；只补齐目标章节并保留其余答案。"
            )
        else:
            debug_steps.append("Agent 尚未形成最终答案，保留工具上下文后触发收敛步骤。")
        missing_tool_instruction = ", ".join(sorted(missing_tools)) or "none"
        missing_file_instruction = ", ".join(sorted(missing_files)) or "none"
        missing_section_instruction = (
            ", ".join(
                f"{number}. {title}" for number, title in refinement_sections.items()
            )
            or "none"
        )
        quality_issue_instruction = (
            "; ".join(
                f"{number}. {' | '.join(messages)}"
                for number, messages in quality_issues.items()
            )
            or "none"
        )
        lesson_slot_instruction = _lesson_slot_manifest(lesson_specs)
        if section_only_refinement:
            grounded_context = truncate_to_tokens(
                collection_evidence,
                budget.synthesis_budget,
            )
            retry_prompt = (
                "The previous answer is mostly complete. Do not rewrite, summarize, or repeat "
                "any accepted section because the runtime will retain it verbatim. Return ONLY "
                "the following missing or underdeveloped numbered sections, each with its exact "
                f"Markdown number and title: {missing_section_instruction}. "
                "Fully develop those sections from the existing inspected-file evidence. "
                "Correct every grounded-deliverable defect listed here: "
                f"{quality_issue_instruction}. "
                "Use SOURCE_FACTS as the only authority for lesson titles and scripture "
                "references. If QWOPUS_EXPLICIT_RUBRIC_FOUND is false, explicitly say that no "
                "rubric was supplied and do not invent points, weights, or totals. "
                "Never use placeholders such as '略', 'omitted', or 'to be completed'. "
                "Do not add a preface, conclusion, Observation, Thought, tool log, or any "
                "section not listed above. Follow the language of the user's question.\n\n"
                f"{lesson_slot_instruction}\n\n"
                "Grounding evidence from the completed collection read follows. Treat each "
                "# File block as isolated and use no outside knowledge:\n"
                f"{grounded_context}"
            )
        else:
            retry_prompt = (
                "Continue from the existing tool observations and answer the original "
                "user question now. "
                f"Before answering, call every missing required tool: {missing_tool_instruction}. "
                "Inspect every missing file with document_search, document_read_section, "
                f"or document_collection_summary: "
                f"{missing_file_instruction}. "
                "For each missing spreadsheet, call excel_schema and excel_analysis with that "
                "file name; generate restricted pandas code from its schema observation. "
                "Rewrite the complete answer and fully deliver every requested numbered section; "
                f"missing or underdeveloped sections: {missing_section_instruction}. "
                "Correct every grounded-deliverable defect listed here: "
                f"{quality_issue_instruction}. "
                "Never use placeholders such as '略', 'omitted', or 'to be completed'. "
                "Return only a complete natural-language final "
                "answer. Do not repeat Observation, tool output, Thought, code drafts, or "
                "internal steps. Follow the language of the user's question."
            )
        if section_only_refinement and collection_evidence:
            # 原因：原 Agent memory 已累计工具结果、outline 和旧 Draft；弱模型会在长上下文里
            # 重复相邻课次并漏掉某个课次。
            # 作用：只把受控 collection evidence 和精确课次槽位交给无工具的新 Agent，
            # 接受章节仍由确定性 merge 保留。
            retry_agent = build_smolagents_tool_calling_agent(
                settings=effective_settings,
                tools=[],
            )
            retry_max_steps = 2
            retry_result = retry_agent.run(
                retry_prompt,
                max_steps=retry_max_steps,
                return_full_result=True,
            )
        else:
            retry_agent = agent
            retry_max_steps = min(
                max(len(missing_files) + len(missing_tools) + 2, 3),
                12,
            )
            retry_result = retry_agent.run(
                retry_prompt,
                reset=False,
                max_steps=retry_max_steps,
                return_full_result=True,
            )
        retry_answer, retry_state, retry_steps = _unpack_agent_run_result(retry_result)
        debug_runs.append(
            _build_agent_debug_run(
                label=(
                    "file_analysis_section_refinement"
                    if section_only_refinement
                    else "file_analysis_refinement"
                ),
                prompt=retry_prompt,
                max_steps=retry_max_steps,
                state=retry_state,
                output=retry_answer,
                steps=retry_steps,
            )
        )
        retry_tool_calls = _extract_agent_tool_calls(retry_steps)
        tool_calls.extend(retry_tool_calls)
        all_steps.extend(retry_steps)
        inspected_files.update(_extract_inspected_file_names(retry_steps))
        inspected_files.update(_extract_collection_covered_file_names(retry_steps))
        debug_steps.extend(
            _agent_debug_steps(
                state=retry_state,
                steps=retry_steps,
                tool_calls=retry_tool_calls,
                prefix="收敛轮",
            )
        )
        retry_final_answer = _extract_final_answer(retry_answer)
        if section_only_refinement:
            if not _looks_like_tool_observation(retry_final_answer):
                final_answer = _merge_numbered_section_refinement(
                    final_answer,
                    retry_final_answer,
                    requested_sections,
                    refinement_sections,
                    lesson_specs,
                )
            final_answer = _apply_grounded_report_fallbacks(
                answer=final_answer,
                refinement=retry_final_answer,
                requested=requested_sections,
                target_sections=refinement_sections,
                quality_issues=quality_issues,
                file_names=file_names,
                collection_evidence=collection_evidence,
                lesson_specs=lesson_specs,
            )
        else:
            final_answer = retry_final_answer
        missing_sections = _missing_requested_sections(
            final_answer,
            requested_sections,
        )
        collection_evidence = _collection_grounding_evidence(all_steps)
        lesson_specs = _lesson_grounding_specs(file_names, collection_evidence)
        quality_issues = _report_quality_issues(
            answer=final_answer,
            requested=requested_sections,
            file_names=file_names,
            user_question=user_question,
            collection_evidence=collection_evidence,
        )

    missing_tools = required_tools.difference(tool_calls)
    if missing_tools:
        missing_names = ", ".join(sorted(missing_tools))
        raise RuntimeError(f"smolagents did not call required file tools: {missing_names}.")
    missing_files = set(file_names).difference(inspected_files)
    if missing_files:
        missing_names = ", ".join(sorted(missing_files))
        raise RuntimeError(f"smolagents did not inspect uploaded files: {missing_names}.")
    if missing_sections:
        missing_names = ", ".join(
            f"{number}. {title}" for number, title in missing_sections.items()
        )
        raise RuntimeError(
            "smolagents did not complete requested report sections: "
            f"{missing_names}."
        )
    if quality_issues:
        issue_names = "; ".join(
            f"{number}. {' | '.join(messages)}"
            for number, messages in quality_issues.items()
        )
        raise RuntimeError(
            "smolagents did not satisfy the grounded report contract: "
            f"{issue_names}."
        )
    if not final_answer or _looks_like_tool_observation(final_answer):
        raise RuntimeError("smolagents did not produce a final answer after tool execution.")

    return DocumentAnalysisRun(
        answer=final_answer,
        debug_steps=debug_steps,
        tool_calls=list(dict.fromkeys(tool_calls)),
        inspected_file_names=tuple(
            file_name for file_name in file_names if file_name in inspected_files
        ),
        debug_runs=tuple(debug_runs),
    )


def format_file_analysis_agent_prompt(
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    analysis_mode: str = "question",
    has_collection_summary: bool = False,
) -> str:
    """Build the task prompt for the smolagents uploaded-file driver."""
    question = user_question.strip() or "Summarize the uploaded files."
    requires_collection_summary = _requires_collection_summary(
        available=has_collection_summary,
        file_count=len(file_names),
        user_question=question,
        analysis_mode=analysis_mode,
    )
    file_list = "\n".join(f"- {file_name}" for file_name in file_names)
    lines = [
        "You are Qwopus-Agent's uploaded-file analysis agent.",
        "Use the available tools to inspect the current uploaded files before answering.",
        "Never invent file content and never return raw Tool Observation text as the final answer.",
        (
            "The final answer must follow the language of the user's question; "
            "if unclear, follow the files' main language."
        ),
        (
            "Give a complete natural-language answer with enough detail for the request, "
            "not a fixed short bullet list."
        ),
        "Use rag_search only when previously indexed local knowledge is relevant.",
        (
            "For current documents, use document_search for a specific question. Use "
            "document_outline and document_read_section for chapter tasks, and document_summary "
            "for a whole-document summary. "
            "Never assume that the beginning of a file represents the whole document."
        ),
        "Current uploaded files:",
        file_list,
    ]
    if requires_collection_summary:
        lines.append(
            "For a folder-wide task, call document_collection_summary first so every "
            "selected document contributes evidence before drilling into individual files."
        )
        lines.append(
            "Treat every # File block as an isolated source. Copy lesson titles and scripture "
            "references only from that file's SOURCE_FACTS; never add a second remembered "
            "reference, merge neighboring lessons, or invent quotations. If the collection "
            "marker says no explicit rubric was found, state that fact instead of creating "
            "scores or weights."
        )
        if _ALL_SOURCE_REQUEST_PATTERN.search(question):
            lines.append(
                "The user explicitly requested all sources: the document-understanding section "
                "must name and substantively summarize every listed file. If a complete Draft "
                "is requested, write every lesson subsection in full; phrases such as 'the "
                "remaining sections follow the same format' are a failed answer."
            )
    if analysis_mode == "section":
        # 原因：章节分析必须遵守用户在前端选定的范围，不能退回全文泛化总结。
        # 作用：要求 Agent 优先读取受限章节工具，并围绕章节内容组织最终答案。
        lines.append(
            "Analysis mode: selected sections. Use document_read_section and answer only from "
            "the sections available through that scoped tool."
        )
    elif analysis_mode == "full":
        # 原因：长文档全文不能直接进入模型上下文。
        # 作用：强制使用已分层压缩的 document_summary，再按需检索证据补充细节。
        summary_tool = (
            "document_collection_summary"
            if requires_collection_summary
            else "document_summary"
        )
        lines.append(
            f"Analysis mode: whole document. Call {summary_tool} first, then use "
            "document_search only when details need supporting evidence."
        )
    if spreadsheet_names:
        spreadsheet_list = ", ".join(spreadsheet_names)
        lines.extend(
            [
                f"Spreadsheets: {spreadsheet_list}.",
                (
                    "For a spreadsheet, call excel_schema first. If computation is needed, "
                    "then call excel_analysis."
                ),
                (
                    "Use restricted pandas code with dfs and pd, and assign the final value "
                    "to result."
                ),
                "Never request or reproduce the entire spreadsheet.",
            ]
        )
    lines.extend(
        [
            "",
            f"User question: {question}",
            "",
            "Produce the final answer after using the needed tools.",
        ]
    )
    return "\n".join(lines)


def _requires_collection_summary(
    *,
    available: bool,
    file_count: int,
    user_question: str,
    analysis_mode: str,
) -> bool:
    """Require collection coverage only for exhaustive multi-document tasks."""
    if not available or file_count <= 1:
        return False
    # 原因：具体事实问题可逐文件检索；无条件强制 collection 会浪费步骤并导致弱模型失败。
    # 作用：全文模式和明确要求全部来源时仍保证覆盖，其余任务允许按问题选择文件工具。
    return (
        analysis_mode == "full"
        or _ALL_SOURCE_REQUEST_PATTERN.search(user_question) is not None
    )


def run_smolagents_chat_turn(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
) -> str:
    model = build_smolagents_model(settings)
    response = model.generate(
        build_chat_messages(history, user_message),
        max_tokens=(settings or SmolagentsModelSettings.from_env()).max_tokens,
    )

    # 原因：不同 smolagents 版本返回 ChatMessage 对象或 dict-like 结构。
    # 作用：把返回值统一成 Streamlit 可展示的纯文本。
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(response, dict):
        return str(response.get("content", response))
    return str(response)
