"""Tool-driven chat runner for smolagents integration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from qwopus_agent.integrations import (
    smolagents_answering,
    smolagents_debug,
    smolagents_factory,
    smolagents_knowledge,
    smolagents_model,
)
from qwopus_agent.integrations.skill_tools import (
    build_promoted_workflow_tools,
    build_registered_skill_tools,
)
from qwopus_agent.integrations.smolagents_results import ChatAgentRun
from qwopus_agent.memory import DEFAULT_CONVERSATION_KNOWLEDGE_ROOT
from qwopus_agent.prompts import smolagents as smolagents_prompts
from qwopus_agent.services.answer_quality import strip_unsupported_evidence_lines
from qwopus_agent.services.orchestration_models import (
    AgentOutputRole,
    AnswerContract,
    AnswerPlan,
)
from qwopus_agent.skills import SkillRegistry, WorkflowSpec
from qwopus_agent.utils.token_budget import TokenBudgetManager, truncate_to_tokens

SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings
ChatMessage = smolagents_prompts.ChatMessage
_LOCAL_KNOWLEDGE_TOOLS = smolagents_prompts.LOCAL_KNOWLEDGE_TOOLS
_document_evidence_required_answer = smolagents_prompts.document_evidence_required_answer
_format_agent_chat_prompt = smolagents_prompts.format_agent_chat_prompt
_has_usable_knowledge_evidence = smolagents_prompts.has_usable_knowledge_evidence
_no_knowledge_evidence_answer = smolagents_prompts.no_knowledge_evidence_answer
_requires_document_evidence = smolagents_prompts.requires_document_evidence
_answer_has_grounded_source = smolagents_answering.answer_has_grounded_source
_answer_quality_issues = smolagents_answering.answer_quality_issues
_role_refinement_prompt = smolagents_answering.role_refinement_prompt
_build_agent_debug_run = smolagents_debug.build_agent_debug_run
_extract_agent_observations = smolagents_debug.extract_agent_observations
_extract_agent_tool_calls = smolagents_debug.extract_agent_tool_calls
_extract_final_answer = smolagents_debug.extract_final_answer
_looks_like_tool_observation = smolagents_debug.looks_like_tool_observation
_unpack_agent_run_result = smolagents_debug.unpack_agent_run_result
_build_browser_open_tool = smolagents_knowledge.build_browser_open_tool
_build_local_knowledge_tools = smolagents_knowledge.build_local_knowledge_tools
_build_tavily_search_tool = smolagents_knowledge.build_tavily_search_tool
_build_smolagents_tool_calling_agent = smolagents_factory.build_smolagents_tool_calling_agent


def _discover_default_skill_registry() -> SkillRegistry:
    """Use a no-argument wrapper so dependency overrides have one stable shape."""
    return SkillRegistry.discover()


_discover_skill_registry: Callable[[], SkillRegistry] = _discover_default_skill_registry


def configure_runtime_dependencies(
    *,
    browser_tool_builder: Callable[..., Any] | None = None,
    local_knowledge_tool_builder: Callable[..., Any] | None = None,
    tavily_search_tool_builder: Callable[..., Any] | None = None,
    tool_calling_agent_builder: Callable[..., Any] | None = None,
    skill_registry_discover: Callable[[], SkillRegistry] | None = None,
) -> None:
    """Override runner dependencies from the legacy runtime facade.

    原因：runtime.py 是历史公共入口，测试和扩展代码会 patch 那里的工厂。
    作用：拆分后仍允许旧入口控制依赖，同时避免 runner 直接知道 runtime 模块。
    """
    global _build_browser_open_tool
    global _build_local_knowledge_tools
    global _build_tavily_search_tool
    global _build_smolagents_tool_calling_agent
    global _discover_skill_registry

    if browser_tool_builder is not None:
        _build_browser_open_tool = browser_tool_builder
    if local_knowledge_tool_builder is not None:
        _build_local_knowledge_tools = local_knowledge_tool_builder
    if tavily_search_tool_builder is not None:
        _build_tavily_search_tool = tavily_search_tool_builder
    if tool_calling_agent_builder is not None:
        _build_smolagents_tool_calling_agent = tool_calling_agent_builder
    if skill_registry_discover is not None:
        _discover_skill_registry = skill_registry_discover


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
        _discover_skill_registry(),
        enabled_permissions={"always"},
        max_output_tokens=budget.observation_budget,
        progress_callback=progress_callback,
    )
    if include_global_knowledge and not enable_local_knowledge:
        raise ValueError("Global knowledge requires local knowledge permission.")
    if enable_web_search:
        tools.append(
            _build_tavily_search_tool(
                progress_callback=progress_callback,
                max_results=max_evidence_sources,
            )
        )
    if enable_browser:
        tools.append(
            _build_browser_open_tool(
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
        knowledge_tools = _build_local_knowledge_tools(
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
    agent = _build_smolagents_tool_calling_agent(
        settings=effective_settings,
        tools=tools,
        final_answer_checks=[],
    )
    prompt = _format_agent_chat_prompt(
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
        finalizer = _build_smolagents_tool_calling_agent(
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
