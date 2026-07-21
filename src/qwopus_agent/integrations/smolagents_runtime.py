"""smolagents runtime integration for Qwopus-Agent.

This module connects Qwopus-Agent with an OpenAI-compatible local LLM server
such as optiq serve / mlx_lm.server.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class SmolagentsDependencyError(RuntimeError):
    """Raised when smolagents is required but missing."""


ChatMessage = dict[str, str]
CHAT_HISTORY_MAX_MESSAGES = 8
CHAT_HISTORY_MAX_CHARS = 4000


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
    progress_callback: Callable[[str], None] | None = None,
) -> list[Any]:
    """Build chat tools over the persisted MiniRAG and knowledge graph stores."""
    from qwopus_agent.integrations.smolagents_tools import (
        build_graph_search_tool,
        build_minirag_search_tool,
    )
    from qwopus_agent.memory import MiniRAG
    from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
    from qwopus_agent.memory.graph_extraction import RuleBasedGraphExtractor
    from qwopus_agent.memory.knowledge_graph import (
        DEFAULT_KNOWLEDGE_GRAPH_PATH,
        KnowledgeGraphIndex,
    )

    # 原因：聊天运行在独立 spawn 进程，不能安全复用 Streamlit session 中的原生向量对象。
    # 作用：每次启用本地知识时从持久化文件加载只属于当前聊天进程的检索实例。
    minirag = MiniRAG()
    graph_index = KnowledgeGraphIndex(
        graph=PersistentKnowledgeGraph(DEFAULT_KNOWLEDGE_GRAPH_PATH),
        extractor=RuleBasedGraphExtractor(),
    )
    return [
        build_minirag_search_tool(minirag, progress_callback=progress_callback),
        build_graph_search_tool(graph_index, progress_callback=progress_callback),
    ]


@dataclass(frozen=True)
class SmolagentsModelSettings:
    """Configuration for OpenAI-compatible model server."""

    model_id: str

    base_url: str

    api_key: str = "sk-optiq-local"

    timeout_seconds: int = 120

    temperature: float = 0.2

    max_tokens: int = 1024

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.getenv(
                "QWOPUS_MLX_MODEL",
                "gemma-4-12B-it-qat-OptiQ-4bit",
            ),
            base_url=os.getenv("QWOPUS_MLX_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("QWOPUS_SMOLAGENTS_API_KEY", "sk-optiq-local"),
            timeout_seconds=int(os.getenv("QWOPUS_SMOLAGENTS_TIMEOUT_SECONDS", "120")),
            temperature=float(os.getenv("QWOPUS_SMOLAGENTS_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("QWOPUS_SMOLAGENTS_MAX_TOKENS", "1024")),
        )


@dataclass(frozen=True)
class AgentDebugRun:
    """One complete local smolagents run retained only for the debug console."""

    label: str
    prompt: str
    max_steps: int
    state: str | None
    output: str
    steps: tuple[dict[str, Any], ...] = ()


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


def _models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def resolve_model_settings(
    settings: SmolagentsModelSettings | None = None,
) -> SmolagentsModelSettings:
    """Return settings updated with the model currently exposed by the server."""
    settings = settings or SmolagentsModelSettings.from_env()
    try:
        status, payload = _request_models(settings)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return settings

    model_id = _extract_server_model_id(payload) if 200 <= status < 300 else None
    if not model_id:
        return settings

    # 原因：服务器加载的模型会变化，.env 中的静态名称可能已经过期。
    # 作用：每次请求模型列表后使用实时 id，同时保留其他连接参数不变。
    return replace(settings, model_id=model_id)


def check_model_connection(
    settings: SmolagentsModelSettings | None = None,
) -> tuple[bool, str]:
    settings = settings or SmolagentsModelSettings.from_env()

    try:
        status, payload = _request_models(settings)
        if 200 <= status < 300:
            model_id = _extract_server_model_id(payload) or settings.model_id
            return True, (
                f"模型服务在线: {settings.base_url} (当前模型: {_display_model_name(model_id)})"
            )
        return False, f"模型服务异常: {status}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", str(exc))
        return False, (f"无法连接模型服务: {settings.base_url} ({reason})")


def _request_models(settings: SmolagentsModelSettings) -> tuple[int, dict[str, Any]]:
    """Request the OpenAI-compatible model list once."""
    request = urllib.request.Request(
        _models_endpoint(settings.base_url),
        headers={"Authorization": f"Bearer {settings.api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload


def _extract_server_model_id(payload: dict[str, Any]) -> str | None:
    """Read a model id from common OpenAI-compatible response shapes."""
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        model_id = data[0].get("id")
        if isinstance(model_id, str) and model_id:
            return model_id

    models = payload.get("models")
    if isinstance(models, list) and models and isinstance(models[0], dict):
        for key in ("model", "name"):
            model_id = models[0].get(key)
            if isinstance(model_id, str) and model_id:
                return model_id
    return None


def _display_model_name(model_id: str) -> str:
    """Return a readable filename for Unix or Windows model paths."""
    return model_id.replace("\\", "/").rsplit("/", 1)[-1]


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
):
    try:
        from smolagents import CodeAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

    model = build_smolagents_model(settings)

    return CodeAgent(
        tools=tools or [],
        model=model,
        additional_authorized_imports=[
            "os",
            "pathlib",
            "glob",
            "json",
            "datetime",
            "re",
            "subprocess",
        ],
    )


def build_smolagents_tool_calling_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
):
    """Build the smolagents Agent runtime used as Qwopus' chat driver."""
    try:
        from smolagents import ToolCallingAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

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
):
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
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Run one chat turn through smolagents as the Agent driver."""
    return run_agent_chat_turn_with_debug(
        user_message=user_message,
        history=history,
        settings=settings,
        enable_web_search=enable_web_search,
        enable_local_knowledge=enable_local_knowledge,
        progress_callback=progress_callback,
    ).answer


def run_agent_chat_turn_with_debug(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
    enable_web_search: bool = False,
    enable_local_knowledge: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> ChatAgentRun:
    """Run chat and retain only the safe Tool metadata needed by orchestration."""
    tools: list[Any] = []
    if enable_web_search:
        tools.append(build_tavily_search_tool(progress_callback=progress_callback))
    if enable_local_knowledge:
        tools.extend(build_local_knowledge_tools(progress_callback=progress_callback))
    agent = build_smolagents_tool_calling_agent(settings=settings, tools=tools)
    prompt = format_agent_chat_prompt(
        history=history,
        user_message=user_message,
        enable_web_search=enable_web_search,
        enable_local_knowledge=enable_local_knowledge,
    )
    if progress_callback is not None:
        progress_callback("planning")
    # 原因：部分模型会忽略提示并重复调用已经成功的检索 Tool，直到耗尽较大的步数上限。
    # 作用：单类检索只允许一次调用加一次收尾；两类检索最多各调用一次再收尾。
    max_steps = 3 if enable_web_search and enable_local_knowledge else 2
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

    local_tool_used = bool({"rag_search", "graph_search"}.intersection(tool_calls))
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
        finalizer = build_smolagents_tool_calling_agent(settings=settings, tools=[])
        retry_prompt = (
            f"Original user question:\n{user_message}\n\n"
            f"Available tool evidence:\n{evidence[:12_000]}\n\n"
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
    )


def format_agent_chat_prompt(
    history: list[ChatMessage],
    user_message: str,
    enable_web_search: bool,
    enable_local_knowledge: bool = False,
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
            "Previously uploaded local knowledge is available. Use rag_search for semantic "
            "document evidence. Use graph_search for named-entity relationships, cross-document "
            "links, or multi-hop paths. Do not call both unless both evidence types are necessary. "
            "After a successful Observation, synthesize it into the final answer and cite the "
            "available local source names or pages; never expose raw Observation text."
        )
    else:
        lines.append(
            "Local knowledge access is disabled. Chat cannot access previously uploaded files or "
            "MiniRAG; ask the user to enable local knowledge or upload files on the document "
            "analysis page when their content is needed."
        )

    if history:
        lines.append("\nRECENT CONVERSATION:")
        for message in _bounded_chat_history(history):
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


def _bounded_chat_history(history: list[ChatMessage]) -> list[ChatMessage]:
    """Keep recent chat context inside a predictable character budget."""
    selected: list[ChatMessage] = []
    remaining_chars = CHAT_HISTORY_MAX_CHARS

    for message in reversed(history[-CHAT_HISTORY_MAX_MESSAGES:]):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        if len(content) > remaining_chars:
            # 原因：单条长报告也可能超过整个上下文预算，拖慢模型首 token。
            # 作用：保留最近消息的开头并停止加入更旧内容，让延迟保持可预测。
            if remaining_chars > 0:
                selected.append(
                    {"role": role, "content": f"{content[:remaining_chars]} [truncated]"}
                )
            break
        selected.append({"role": role, "content": content})
        remaining_chars -= len(content)

    return list(reversed(selected))


def run_smolagents_file_analysis_with_debug(
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    tools: list[Any],
    settings: SmolagentsModelSettings | None = None,
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
        file_names=file_names,
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
            f"Read every missing file with document_parser exactly once: "
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
        "Current uploaded files:",
        file_list,
    ]
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
):
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


def _unpack_agent_run_result(run_result: Any) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Normalize smolagents RunResult and older direct return values."""
    if hasattr(run_result, "output"):
        output = getattr(run_result, "output", None)
        state = getattr(run_result, "state", None)
        steps = getattr(run_result, "steps", None)
        return str(output or ""), str(state) if state is not None else None, steps or []
    return str(run_result), None, []


def _build_agent_debug_run(
    *,
    label: str,
    prompt: str,
    max_steps: int,
    state: str | None,
    output: str,
    steps: list[dict[str, Any]],
) -> AgentDebugRun:
    """Copy a JSON-safe raw run so the local debug console can inspect every step."""
    # 原因：smolagents step 中可能混入 Pydantic 对象或其他不可序列化值。
    # 作用：保留完整可读内容，同时确保 spawn Queue 和 JSON 下载都能稳定传输。
    normalized_steps = tuple(
        {
            str(key): _normalize_debug_value(value)
            for key, value in step.items()
        }
        if isinstance(step, dict)
        else {"value": _normalize_debug_value(step)}
        for step in steps
    )
    return AgentDebugRun(
        label=label,
        prompt=prompt,
        max_steps=max_steps,
        state=state,
        output=output,
        steps=normalized_steps,
    )


def _normalize_debug_value(value: Any) -> Any:
    """Convert nested debug values to JSON-safe primitives without dropping content."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_debug_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize_debug_value(model_dump(mode="json"))
    return str(value)


def _extract_agent_tool_calls(steps: list[dict[str, Any]]) -> list[str]:
    """Extract Tool names from smolagents' succinct step records."""
    names: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for tool_call in step.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
    return names


def _extract_agent_observations(steps: list[dict[str, Any]]) -> list[str]:
    """Return observations produced by non-final tools without model thoughts."""
    observations: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_names = _extract_agent_tool_calls([step])
        observation = step.get("observations")
        if (
            isinstance(observation, str)
            and observation.strip()
            and any(name != "final_answer" for name in tool_names)
        ):
            observations.append(observation.strip())
    return observations


def _extract_inspected_file_names(steps: list[dict[str, Any]]) -> set[str]:
    """Return file names passed to document_parser Tool calls."""
    inspected: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        for tool_call in step.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "document_parser":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if isinstance(arguments, dict) and isinstance(arguments.get("file_name"), str):
                inspected.add(arguments["file_name"])
    return inspected


def _required_file_tools(file_names: list[str], spreadsheet_names: list[str]) -> set[str]:
    """Return the minimum Tool chain required before a file answer is accepted."""
    required: set[str] = set()
    if len(file_names) > len(spreadsheet_names):
        required.add("document_parser")
    if spreadsheet_names:
        # 原因：模型可能直接根据 sample 心算，绕过本地 pandas 沙箱。
        # 作用：所有 Excel 回答必须先看 schema，再执行本地代码；完整表格始终不进入 LLM。
        required.update({"excel_schema", "excel_analysis"})
    return required


def _agent_debug_steps(
    state: str | None,
    steps: list[dict[str, Any]],
    tool_calls: list[str],
    prefix: str = "smolagents",
) -> list[str]:
    """Build a safe trace without exposing Tool observations or model reasoning."""
    trace = [f"{prefix} 运行状态：{state or 'completed'}；步骤数：{len(steps)}。"]
    for tool_name in tool_calls:
        # 原因：用户可以选择查看 Agent 过程，但原始 Observation 可能包含整段文件内容。
        # 作用：调试区只展示调用了哪个 Tool，不展示参数、推理文本或 Tool 返回正文。
        trace.append(f"{prefix} 调用 Tool：{tool_name}")
    for step in steps:
        if isinstance(step, dict) and step.get("error"):
            step_number = step.get("step_number", "?")
            trace.append(f"{prefix} 第 {step_number} 步发生错误，Agent 已按运行策略处理。")
    return trace


def _looks_like_tool_observation(text: str) -> bool:
    """Detect model output that is still exposing tool observations."""
    lowered = text.lower()
    return "observation:" in lowered or "document analysis:" in lowered or "## preview" in lowered


def _extract_final_answer(text: str) -> str:
    """Extract final_answer(...) when a CodeAgent-style answer leaks through."""
    match = re.search(r"final_answer\((?P<quote>['\"])(?P<answer>.*?)(?P=quote)\)", text, re.S)
    if match:
        return match.group("answer").strip()
    return text.strip()
