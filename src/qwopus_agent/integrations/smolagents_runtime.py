"""smolagents runtime integration for Qwopus-Agent.

This module connects Qwopus-Agent with an OpenAI-compatible local LLM server
such as optiq serve / mlx_lm.server.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qwopus_agent.integrations import smolagents_debug, smolagents_model
from qwopus_agent.memory import (
    DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    conversation_knowledge_path,
)
from qwopus_agent.utils.token_budget import (
    TokenBudgetManager,
    estimate_tokens,
    truncate_to_tokens,
)

AgentDebugRun = smolagents_debug.AgentDebugRun
SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings
check_model_connection = smolagents_model.check_model_connection
resolve_model_settings = smolagents_model.resolve_model_settings

_agent_debug_steps = smolagents_debug.agent_debug_steps
_build_agent_debug_run = smolagents_debug.build_agent_debug_run
_extract_agent_observations = smolagents_debug.extract_agent_observations
_extract_agent_tool_calls = smolagents_debug.extract_agent_tool_calls
_extract_final_answer = smolagents_debug.extract_final_answer
_extract_inspected_file_names = smolagents_debug.extract_inspected_file_names
_looks_like_tool_observation = smolagents_debug.looks_like_tool_observation
_required_file_tools = smolagents_debug.required_file_tools
_unpack_agent_run_result = smolagents_debug.unpack_agent_run_result


class SmolagentsDependencyError(RuntimeError):
    """Raised when smolagents is required but missing."""


ChatMessage = dict[str, str]
CHAT_HISTORY_MAX_MESSAGES = 8
_LOCAL_KNOWLEDGE_TOOLS = frozenset(
    {
        "rag_search",
        "graph_search",
        "global_rag_search",
        "global_graph_search",
    }
)
_NO_KNOWLEDGE_EVIDENCE_MARKERS = (
    "no relevant minirag results",
    "no matching knowledge-graph path was found",
    "no relevant knowledge",
    "no relevant evidence",
    "no usable tool evidence",
    "tool execution failed",
    "error executing tool",
    "error while executing tool",
)


def build_tavily_search_tool(
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    """Load the Tavily Tool factory only when web search is enabled."""
    # 原因：普通聊天不需要 Excel、文档或 MiniRAG Tool 依赖。
    # 作用：关闭联网时避免加载完整工具模块，同时保留可测试的工厂注入点。
    from qwopus_agent.integrations.smolagents_tools import (
        build_tavily_search_tool as create_tool,
    )

    return create_tool(progress_callback=progress_callback)


def build_local_knowledge_tools(
    knowledge_scope: str,
    progress_callback: Callable[[str], None] | None = None,
    min_source_relevance: float = 0.55,
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    include_global_knowledge: bool = False,
    budget_manager: TokenBudgetManager | None = None,
) -> list[Any]:
    """Build chat tools over the persisted MiniRAG and knowledge graph stores."""
    from qwopus_agent.integrations.smolagents_tools import (
        build_graph_search_tool,
        build_minirag_search_tool,
    )
    from qwopus_agent.memory import MiniRAG

    budget = budget_manager or TokenBudgetManager()
    # 原因：聊天运行在独立 spawn 进程，不能安全复用 Streamlit session 中的原生向量对象。
    # 作用：每次启用本地知识时只加载当前 conversation_id 的向量和配套图谱快照。
    minirag = MiniRAG(
        storage_path=conversation_knowledge_path(
            knowledge_scope,
            root=knowledge_root,
        )
    )
    tools = [
        build_minirag_search_tool(
            minirag,
            min_relevance=min_source_relevance,
            progress_callback=progress_callback,
            budget_manager=budget,
        ),
        build_graph_search_tool(
            minirag.graph_index,
            progress_callback=progress_callback,
            budget_manager=budget,
        ),
    ]
    if include_global_knowledge:
        global_minirag = MiniRAG()
        # 原因：全局检索是额外权限，复用 rag_search 名称会让 Agent 和日志无法区分来源范围。
        # 作用：仅在显式授权后增加独立命名的全局语义与图路径 Tool。
        tools.extend(
            [
                build_minirag_search_tool(
                    global_minirag,
                    min_relevance=min_source_relevance,
                    progress_callback=progress_callback,
                    budget_manager=budget,
                    tool_name="global_rag_search",
                    description=(
                        "Search the explicitly authorized global MiniRAG store. Use this only "
                        "when current-conversation evidence is insufficient."
                    ),
                ),
                build_graph_search_tool(
                    global_minirag.graph_index,
                    progress_callback=progress_callback,
                    budget_manager=budget,
                    tool_name="global_graph_search",
                    description=(
                        "Search explicit relationships in the authorized global knowledge graph."
                    ),
                ),
            ]
        )
    return tools


@dataclass(frozen=True)
class DocumentAnalysisRun:
    """Document analysis answer with a UI-visible debug trace."""

    answer: str

    debug_steps: list[str]

    tool_calls: list[str] = field(default_factory=list)

    debug_runs: tuple[AgentDebugRun, ...] = ()


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


def format_chat_prompt(
    history: list[ChatMessage],
    user_message: str,
) -> str:
    lines = [
        """
    You are a CodeAgent.

    Always answer using:

    Thought: ...

    final_answer

    Never output plain text.
    """,
        "",
        "历史对话:",
    ]

    for message in history:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            lines.append(f"用户：{content}")

        elif role == "assistant":
            lines.append(f"助手：{content}")

    lines.extend(
        [
            "",
            f"用户：{user_message}",
            "",
            "助手：",
        ]
    )

    return "\n".join(lines)


def build_chat_messages(
    history: list[ChatMessage],
    user_message: str,
) -> list[ChatMessage]:
    """Build plain chat messages for direct model generation."""
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": (
                "你是 Qwopus-Agent 的本地办公助手。"
                "请直接、清晰地回答用户问题。"
                # 原因：普通聊天只收到对话历史，无法读取上传文件或 MiniRAG 内容。
                # 作用：禁止模型假装分析旧文件，并引导用户使用文档分析流程。
                "普通聊天无法自动访问用户之前上传的文件或 MiniRAG。"
                "如果用户要求分析未附带内容的历史文件，请明确说明无法读取，"
                "并请用户在文档分析页面重新选择文件；不要编造文件内容。"
                "不要输出 Thought、代码块或 final_answer 包装，除非用户明确要求。"
            ),
        }
    ]

    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            # 原因：普通对话要保留上下文，但不应该把 Streamlit 内部状态泄漏给模型。
            # 作用：只传递模型需要理解上下文的 role/content。
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


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
        client_kwargs={"timeout": settings.timeout_seconds},
        temperature=settings.temperature,
    )


def build_smolagents_code_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
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
    )


def build_smolagents_tool_calling_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Build the smolagents Agent runtime used as Qwopus' chat driver."""
    try:
        from smolagents import ToolCallingAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

    settings = settings or SmolagentsModelSettings.from_env()
    if settings.capabilities.agent_mode == "code":
        return build_smolagents_code_agent(settings=settings, tools=tools)

    model = build_smolagents_model(settings)
    # 原因：smolagents 是整体 Agent 驱动入口，工具选择应由 Agent runtime 处理。
    # 作用：Streamlit 不再手动先搜索再拼 prompt，而是把受控 Tool 交给 Agent。
    return ToolCallingAgent(
        tools=tools or [],
        model=model,
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
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    min_source_relevance: float = 0.55,
    knowledge_scope: str | None = None,
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Run one chat turn through smolagents as the Agent driver."""
    return run_agent_chat_turn_with_debug(
        user_message=user_message,
        history=history,
        settings=settings,
        enable_web_search=enable_web_search,
        enable_local_knowledge=enable_local_knowledge,
        include_global_knowledge=include_global_knowledge,
        min_source_relevance=min_source_relevance,
        knowledge_scope=knowledge_scope,
        knowledge_root=knowledge_root,
        progress_callback=progress_callback,
    ).answer


def run_agent_chat_turn_with_debug(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
    enable_web_search: bool = False,
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    min_source_relevance: float = 0.55,
    knowledge_scope: str | None = None,
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    progress_callback: Callable[[str], None] | None = None,
) -> ChatAgentRun:
    """Run chat and retain only the safe Tool metadata needed by orchestration."""
    effective_settings = settings or SmolagentsModelSettings.from_env()
    budget = TokenBudgetManager(
        context_window=effective_settings.context_window_tokens,
        output_reserve=effective_settings.max_tokens,
    )
    tools: list[Any] = []
    if include_global_knowledge and not enable_local_knowledge:
        raise ValueError("Global knowledge requires local knowledge permission.")
    if enable_web_search:
        tools.append(build_tavily_search_tool(progress_callback=progress_callback))
    if enable_local_knowledge:
        if not knowledge_scope:
            raise ValueError("knowledge_scope is required when local knowledge is enabled")
        tools.extend(
            build_local_knowledge_tools(
                knowledge_scope,
                progress_callback=progress_callback,
                min_source_relevance=min_source_relevance,
                knowledge_root=knowledge_root,
                include_global_knowledge=include_global_knowledge,
                budget_manager=budget,
            )
        )
    agent = build_smolagents_tool_calling_agent(settings=effective_settings, tools=tools)
    prompt = format_agent_chat_prompt(
        history=history,
        user_message=user_message,
        enable_web_search=enable_web_search,
        enable_local_knowledge=enable_local_knowledge,
        include_global_knowledge=include_global_knowledge,
        history_max_tokens=budget.history_budget,
    )
    if progress_callback is not None:
        progress_callback("planning")
    # 原因：部分模型会忽略提示并重复调用已经成功的检索 Tool，直到耗尽较大的步数上限。
    # 作用：单类检索只允许一次调用加一次收尾；两类检索最多各调用一次再收尾。
    max_steps = (
        2
        + int(enable_web_search and enable_local_knowledge)
        + int(include_global_knowledge)
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

    local_tool_used = bool(_LOCAL_KNOWLEDGE_TOOLS.intersection(tool_calls))
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
    )
    if needs_refinement:
        evidence = "\n\n".join(observations) or final_answer or "No usable tool evidence."
        # 原因：继续复用带 Tool 的 Agent 仍可能无视提示并再次检索，造成长时间循环。
        # 作用：把已取得的 Observation 交给无工具 finalizer，只允许它生成最终自然语言答案。
        finalizer = build_smolagents_tool_calling_agent(
            settings=effective_settings,
            tools=[],
        )
        retry_prompt = (
            f"Original user question:\n{user_message}\n\n"
            "Available tool evidence:\n"
            f"{truncate_to_tokens(evidence, budget.synthesis_budget)}\n\n"
            "Answer the original user question now. Return only a complete "
            "natural-language final answer in the user's language. State the conclusion, "
            "explain the relevant relationship or evidence, and explicitly cite every "
            "available local source file and page. Never expose Observation, Thought, tool "
            "logs, or drafts."
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


def format_agent_chat_prompt(
    history: list[ChatMessage],
    user_message: str,
    enable_web_search: bool,
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    history_max_tokens: int = 1024,
) -> str:
    """Build a single task prompt for smolagents Agent chat."""
    lines = [
        "You are Qwopus-Agent's local office assistant.",
        "Return only the final answer. Do not expose Tool logs, Observation, Thought, or drafts.",
        # 原因：历史对话或系统提示的中文可能让模型忽略当前英文等其他语言输入。
        # 作用：只根据当前问题选择回答语言，混合输入则跟随主要语言或用户明确要求。
        (
            "The final answer MUST use the same language as the CURRENT USER QUESTION below. "
            "Determine it only from that question, not from this prompt or conversation history. "
            "Do not default to Chinese or English. For mixed-language input, use its dominant "
            "language unless the user explicitly requests another language."
        ),
        # 原因：只要求“最终答案”容易让模型把搜索结果压缩成一小段。
        # 作用：在未要求简答时生成有结论、细节、解释和来源的完整多段回答。
        (
            "Unless the user explicitly asks for brevity, provide a detailed, complete answer: "
            "answer directly, then explain key facts, context or practical implications, and "
            "include available source links. Do not reduce the answer to a short bullet list."
        ),
    ]
    if enable_web_search:
        # 原因：部分模型会把“详细回答”仍压缩成几条短句，尤其是中文输出。
        # 作用：为联网答案规定可检查的内容范围和篇幅，同时保留用户主动要求简答的权利。
        lines.append(
            "Use tavily_search when current or external information is needed, then synthesize "
            "the evidence into the final answer. Unless brevity was requested, organize the "
            "answer into substantial sections covering the direct answer, how it works, key "
            "features or evidence, practical uses, limitations or cautions, and 2-5 actual "
            "source URLs when they are useful. Match the depth and length to the question; do "
            "not enforce a fixed minimum length, and respect explicit requests for a shorter "
            "or longer response. For a simple question, call tavily_search only once; after a "
            "successful Observation, use that evidence and call final_answer instead of "
            "repeating the search."
        )
    else:
        lines.append("Internet search is disabled; do not claim that you searched the web.")

    if enable_local_knowledge:
        # 原因：语义检索和图路径检索解决不同问题，模型需要明确的工具选择边界。
        # 作用：内容问题使用 rag_search，实体关系问题使用 graph_search，并在检索后总结。
        lines.append(
            "Local knowledge uploaded in this conversation is available. Use rag_search for "
            "semantic document evidence. Use graph_search for named-entity relationships, "
            "cross-document links, or multi-hop paths. Do not call both unless both evidence "
            "types are necessary. "
            "After a successful Observation, synthesize it into the final answer and cite the "
            "available local source names or pages; never expose raw Observation text. If the "
            "knowledge tools return no relevant evidence, do not answer from general knowledge."
        )
        if include_global_knowledge:
            lines.append(
                "The user explicitly allowed global knowledge for this turn. Use "
                "global_rag_search or global_graph_search only when the current conversation "
                "does not contain enough evidence, and clearly preserve global source citations."
            )
        else:
            lines.append(
                "Global knowledge is not authorized. Never claim to use files from other "
                "conversations or call a global knowledge tool."
            )
    else:
        lines.append(
            "Local knowledge access is disabled. Chat cannot access previously uploaded files or "
            "MiniRAG; ask the user to enable local knowledge or upload files on the document "
            "analysis page when their content is needed."
        )

    if history:
        lines.append("\nRECENT CONVERSATION:")
        for message in _bounded_chat_history(
            history,
            max_tokens=history_max_tokens,
        ):
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                lines.append(f"{role}: {content}")

    lines.extend(
        [
            "",
            "CURRENT USER QUESTION (the only source for response language):",
            user_message,
            "",
            "Now produce the complete final answer in that same language.",
        ]
    )
    return "\n".join(lines)


def _has_usable_knowledge_evidence(observations: list[str]) -> bool:
    """Distinguish source-bearing Tool evidence from explicit empty/error observations."""
    for observation in observations:
        normalized = " ".join(observation.casefold().split())
        if normalized and not any(
            marker in normalized for marker in _NO_KNOWLEDGE_EVIDENCE_MARKERS
        ):
            return True
    return False


def _no_knowledge_evidence_answer(user_message: str) -> str:
    """Return a deterministic refusal without asking an LLM to fill the evidence gap."""
    if any("\u3400" <= character <= "\u9fff" for character in user_message):
        return (
            "当前会话的本地知识中没有检索到足够的相关证据，因此我没有使用常识补全答案。"
            "请确认文件已上传到当前会话，或降低相关性阈值后重试。"
        )
    return (
        "I could not find enough relevant evidence in this conversation's local knowledge, "
        "so I did not fill the gap with general knowledge. Confirm that the files were uploaded "
        "to this conversation or retry with a lower relevance threshold."
    )


def _bounded_chat_history(
    history: list[ChatMessage],
    *,
    max_tokens: int,
) -> list[ChatMessage]:
    """Keep recent chat context inside a predictable token budget."""
    selected: list[ChatMessage] = []
    remaining_tokens = max(0, max_tokens)

    for message in reversed(history[-CHAT_HISTORY_MAX_MESSAGES:]):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        content_tokens = estimate_tokens(content)
        if content_tokens > remaining_tokens:
            # 原因：单条长报告也可能超过整个上下文预算，拖慢模型首 token。
            # 作用：保留最近消息的开头并停止加入更旧内容，让延迟保持可预测。
            if remaining_tokens > 0:
                selected.append(
                    {
                        "role": role,
                        "content": (
                            f"{truncate_to_tokens(content, remaining_tokens)} [truncated]"
                        ),
                    }
                )
            break
        selected.append({"role": role, "content": content})
        remaining_tokens -= content_tokens

    return list(reversed(selected))


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

    agent = build_smolagents_tool_calling_agent(settings=settings, tools=tools)
    prompt = format_file_analysis_agent_prompt(
        file_names=file_names,
        spreadsheet_names=spreadsheet_names,
        user_question=user_question,
        analysis_mode=analysis_mode,
    )
    # 原因：固定至少八步会让不遵循提示的模型反复读取同一文件，显著增加等待时间。
    # 作用：按文件数提供一次读取和收尾预算，遗漏文件由下方精确校验触发补充轮。
    max_steps = min(max(len(file_names) + 2, 4), 12)
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
    debug_steps = _agent_debug_steps(state=state, steps=steps, tool_calls=tool_calls)
    required_tools = _required_file_tools(
        spreadsheet_names=spreadsheet_names,
    )
    missing_tools = required_tools.difference(tool_calls)
    parser_files = set(file_names).difference(spreadsheet_names)
    inspected_files = _extract_inspected_file_names(steps)
    missing_files = parser_files.difference(inspected_files)

    final_answer = _extract_final_answer(answer)
    if (
        not final_answer
        or _looks_like_tool_observation(final_answer)
        or missing_tools
        or missing_files
    ):
        # 原因：少数模型会把最后一次 Tool Observation 当作回答，或者在步数内没有调用 final_answer。
        # 作用：保留同一个 Agent memory 再收敛一轮，禁止原始工具输出进入 Streamlit 主结果。
        if missing_tools:
            missing_names = ", ".join(sorted(missing_tools))
            debug_steps.append(f"Agent 尚未调用必要 Tool：{missing_names}；触发补充执行。")
        else:
            debug_steps.append("Agent 尚未形成最终答案，保留工具上下文后触发收敛步骤。")
        missing_tool_instruction = ", ".join(sorted(missing_tools)) or "none"
        missing_file_instruction = ", ".join(sorted(missing_files)) or "none"
        retry_prompt = (
            "Continue from the existing tool observations and answer the original "
            "user question now. "
            f"Before answering, call every missing required tool: {missing_tool_instruction}. "
            f"Inspect every missing file with document_search or document_read_section: "
            f"{missing_file_instruction}. "
            "For excel_analysis, generate restricted pandas code from the existing "
            "excel_schema observation. Return only a complete natural-language final "
            "answer. Do not repeat Observation, tool output, Thought, code drafts, or "
            "internal steps. Follow the language of the user's question."
        )
        retry_max_steps = min(max(len(missing_files) + len(missing_tools) + 2, 3), 12)
        retry_result = agent.run(
            retry_prompt,
            reset=False,
            max_steps=retry_max_steps,
            return_full_result=True,
        )
        retry_answer, retry_state, retry_steps = _unpack_agent_run_result(retry_result)
        debug_runs.append(
            _build_agent_debug_run(
                label="file_analysis_refinement",
                prompt=retry_prompt,
                max_steps=retry_max_steps,
                state=retry_state,
                output=retry_answer,
                steps=retry_steps,
            )
        )
        retry_tool_calls = _extract_agent_tool_calls(retry_steps)
        tool_calls.extend(retry_tool_calls)
        inspected_files.update(_extract_inspected_file_names(retry_steps))
        debug_steps.extend(
            _agent_debug_steps(
                state=retry_state,
                steps=retry_steps,
                tool_calls=retry_tool_calls,
                prefix="收敛轮",
            )
        )
        final_answer = _extract_final_answer(retry_answer)

    missing_tools = required_tools.difference(tool_calls)
    if missing_tools:
        missing_names = ", ".join(sorted(missing_tools))
        raise RuntimeError(f"smolagents did not call required file tools: {missing_names}.")
    missing_files = parser_files.difference(inspected_files)
    if missing_files:
        missing_names = ", ".join(sorted(missing_files))
        raise RuntimeError(f"smolagents did not inspect uploaded files: {missing_names}.")
    if not final_answer or _looks_like_tool_observation(final_answer):
        raise RuntimeError("smolagents did not produce a final answer after tool execution.")

    return DocumentAnalysisRun(
        answer=final_answer,
        debug_steps=debug_steps,
        tool_calls=list(dict.fromkeys(tool_calls)),
        debug_runs=tuple(debug_runs),
    )


def format_file_analysis_agent_prompt(
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    analysis_mode: str = "question",
) -> str:
    """Build the task prompt for the smolagents uploaded-file driver."""
    question = user_question.strip() or "Summarize the uploaded files."
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
        lines.append(
            "Analysis mode: whole document. Call document_summary first, then use "
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
