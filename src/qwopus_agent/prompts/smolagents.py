"""Prompt construction and evidence policies for smolagents chat runs."""

from __future__ import annotations

import re
from typing import Literal

from qwopus_agent.services.orchestration_models import (
    AgentOutputRole,
    AnswerContract,
    AnswerPlan,
)
from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens

ChatMessage = dict[str, str]
CHAT_HISTORY_MAX_MESSAGES = 8
LOCAL_KNOWLEDGE_TOOLS = frozenset(
    {
        "rag_search",
        "graph_search",
        "global_rag_search",
        "global_graph_search",
    }
)
NO_KNOWLEDGE_EVIDENCE_MARKERS = (
    "no relevant minirag results",
    "no matching knowledge-graph path was found",
    "no relevant knowledge",
    "no relevant evidence",
    "no usable tool evidence",
    "tool execution failed",
    "error executing tool",
    "error while executing tool",
)
DOCUMENT_EVIDENCE_ACTION_PATTERN = re.compile(
    (
        r"(?:仔细)?(?:阅读|读取|查看|检查|分析|总结|汇总|比较|对比|引用|"
        r"基于|根据|完成|撰写|写作)|"
        r"\b(?:read|review|inspect|analy[sz]e|summari[sz]e|compare|cite|"
        r"write|draft|use|based\s+on)\b"
    ),
    re.IGNORECASE,
)
DOCUMENT_EVIDENCE_REFERENCE_PATTERNS = (
    re.compile(
        r"(?:我|我们).{0,16}(?:上传|附上|提交).{0,32}(?:文件|文档|资料|材料|附件|课)",
        re.DOTALL,
    ),
    re.compile(
        r"(?:所有|全部|这些|上述|前述|已上传的|上传的|附件中的).{0,16}"
        r"(?:文件|文档|资料|材料|附件)",
        re.DOTALL,
    ),
    re.compile(
        r"\b(?:i|we)\b.{0,40}\b(?:uploaded|attached)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:all(?:\s+of)?(?:\s+the)?|these|those|my|our|the\s+attached|"
        r"the\s+uploaded)\s+(?:(?:uploaded|attached)\s+)?"
        r"(?:files?|documents?|materials?|attachments?)\b",
        re.IGNORECASE,
    ),
)


def format_chat_prompt(
    history: list[ChatMessage],
    user_message: str,
) -> str:
    """Build the legacy CodeAgent chat prompt."""
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
    lines.extend(["", f"用户：{user_message}", "", "助手："])
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
            # 原因：普通对话要保留上下文，但不应该把 UI 内部状态泄漏给模型。
            # 作用：只传递模型需要理解上下文的 role/content。
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def format_agent_chat_prompt(
    history: list[ChatMessage],
    user_message: str,
    enable_web_search: bool,
    enable_browser: bool = False,
    enable_local_knowledge: bool = False,
    include_global_knowledge: bool = False,
    knowledge_primary_scope: Literal["private", "global", "none"] | None = None,
    history_max_tokens: int = 1024,
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    response_language_source: str | None = None,
    answer_contract: AnswerContract | None = None,
    output_role: AgentOutputRole = "final",
    answer_plan: AnswerPlan | None = None,
) -> str:
    """Build one bounded, capability-aware task prompt for Agent chat."""
    language_source = response_language_source or user_message
    resolved_knowledge_scope = knowledge_primary_scope or (
        "private" if enable_local_knowledge else "none"
    )
    lines = _output_role_instructions(output_role)
    if output_role == "final":
        lines.extend(
            [
                # 原因：历史或系统提示的语言可能让模型忽略当前问题的语言。
                # 作用：仅以当前问题决定最终回答语言。
                (
                    "The final answer MUST use the same language as the CURRENT USER QUESTION "
                    "below. Determine it only from that question, not from this prompt or "
                    "conversation history. Do not default to Chinese or English. For "
                    "mixed-language input, use its dominant language unless the user explicitly "
                    "requests another language."
                ),
                _response_detail_instruction(response_detail),
            ]
        )
    if answer_contract is not None:
        lines.append(_answer_contract_instruction(answer_contract))
    if answer_plan is not None:
        # 原因：各 Agent 自行猜测内容结构会产生重复和互不衔接的小节。
        # 作用：Evidence 围绕同一问题收集材料，Synthesizer 再按同一主线组织最终答案。
        lines.extend(
            [
                "ANSWER PLAN (internal; never mention it to the user):",
                answer_plan.model_dump_json(indent=2),
                (
                    "QUALITY EXAMPLE: weak='This approach has risks.' "
                    "strong='Risk: concurrent writes can lose updates when two workers save the "
                    "same record; use a transaction and verify with a collision test.'"
                ),
                (
                    "If no sources are supplied, do not create an evidence section or describe "
                    "completed studies and measurements. Use design reasoning and propose future "
                    "verification."
                ),
            ]
        )
    if enable_web_search:
        lines.append(
            "Use tavily_search when current or external information is needed. For a simple "
            "question, call tavily_search only once; after a successful Observation, use that "
            f"evidence and {_role_completion_instruction(output_role)} instead of repeating the "
            "search. Match depth to the task and do not enforce a fixed minimum length."
        )
    else:
        lines.append("Internet search is disabled; do not claim that you searched the web.")
    if enable_browser:
        lines.append(
            "Use browser_open only for a specific public HTTP(S) page or when JavaScript "
            "rendering is necessary. Read its rendered text once, then "
            f"{_role_completion_instruction(output_role)}; do not retry blocked private/local "
            "URLs."
        )
    else:
        lines.append("Browser automation is disabled; do not claim that you opened a page.")

    _append_knowledge_instructions(
        lines,
        enable_local_knowledge=enable_local_knowledge,
        include_global_knowledge=include_global_knowledge,
        resolved_knowledge_scope=resolved_knowledge_scope,
        output_role=output_role,
    )
    if history:
        lines.append("\nRECENT CONVERSATION:")
        for message in bounded_chat_history(history, max_tokens=history_max_tokens):
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                lines.append(f"{role}: {content}")
    lines.extend(
        [
            "",
            "CURRENT USER QUESTION (the only source for response language):",
            language_source,
        ]
    )
    if user_message != language_source:
        lines.extend(
            [
                "",
                # 原因：续写请求的原文可能只有“继续”，不足以让 Agent 完成正确任务。
                # 作用：把上下文解析后的目标单独交给 Agent，同时不污染语言判断来源。
                "RESOLVED TASK TO EXECUTE:",
                user_message,
            ]
        )
    lines.extend(
        [
            "",
            _role_final_instruction(output_role),
        ]
    )
    return "\n".join(lines)


def _output_role_instructions(output_role: AgentOutputRole) -> list[str]:
    """Give each orchestration role one non-conflicting output contract."""
    if output_role == "evidence":
        return [
            "You are a Qwopus-Agent evidence worker, not the final answer writer.",
            "Use authorized tools when needed, then return only one JSON object through "
            "final_answer. Do not write a user-facing answer, introduction, conclusion, "
            "Markdown, Thought, Observation, or drafts.",
            'Required schema: {"facts":[{"claim":"...","support":"...","sources":["..."],'
            '"confidence":"low|medium|high","plan_item_ids":["P1"]}],'
            '"limitations":["..."]}.',
            "Include only evidence relevant to the objective. Preserve source names, page "
            "numbers, and URLs from actual Tool Observations in sources. If no Tool supplied a "
            "source, use an empty sources list. Distinguish source evidence from inference and "
            "never invent citations, measurements, or missing facts. Assign every fact to each "
            "ANSWER PLAN item it actually supports; use an empty plan_item_ids list when none "
            "applies.",
        ]
    if output_role == "review":
        return [
            "You are Qwopus-Agent's evidence reviewer, not the final answer writer.",
            "Inspect the supplied evidence for agreement, conflict, unsupported claims, and "
            "material gaps. Return only one JSON object through final_answer. Do not write the "
            "user-facing answer, use tools, or expose Thought, Observation, and drafts.",
            'Required schema: {"agreements":["..."],"conflicts":["..."],'
            '"unsupported_claims":["..."],"gaps":["..."],"resolution":"...",'
            '"coverage":[{"plan_item_id":"P1","status":"supported|partial|missing|conflicted",'
            '"finding":"..."}]}.',
            "A gap must name specific missing evidence that could change the answer. Keep gaps "
            "empty when the available evidence is sufficient. Treat empirical claims, named "
            "studies, measurements, or percentages without Tool-grounded sources as unsupported; "
            "source-free architectural reasoning is allowed only when identified as inference. "
            "Return exactly one coverage row for every ANSWER PLAN item.",
        ]
    return [
        "You are Qwopus-Agent's final answer synthesizer.",
        "Return only the complete user-facing final answer. Do not expose Tool logs, "
        "Observation, Thought, internal plans, evidence JSON, review JSON, or drafts.",
        "Cite only supplied sources. Facts without sources are unverified reasoning; never invent "
        "citations, percentages, benchmarks, or publications. For each supported ANSWER PLAN "
        "item, give its conclusion and concrete support, explain why it matters, and add a "
        "relevant implication, example, condition, or limitation. Headings and repetition are "
        "not detail.",
    ]


def _role_completion_instruction(output_role: AgentOutputRole) -> str:
    if output_role == "evidence":
        return "convert it into the required evidence JSON and call final_answer"
    if output_role == "review":
        return "return the required review JSON through final_answer"
    return "synthesize it into the complete user-facing answer and call final_answer"


def _role_final_instruction(output_role: AgentOutputRole) -> str:
    if output_role == "evidence":
        return "Now return only the required evidence JSON through final_answer."
    if output_role == "review":
        return "Now return only the required review JSON through final_answer."
    return "Now produce the complete final answer in that same language."


def _response_detail_instruction(
    response_detail: Literal["concise", "balanced", "detailed"],
) -> str:
    """Translate one UI preference into an adaptive, non-numeric answer contract."""
    if response_detail == "concise":
        return (
            "Keep the final answer concise and direct. Include only the conclusion and essential "
            "supporting facts unless the user explicitly requests more detail."
        )
    if response_detail == "balanced":
        return (
            "Provide a balanced answer with a direct conclusion, the main supporting explanation, "
            "and practical caveats when relevant. Match the structure to the question."
        )
    # 原因：硬性最低字数会让简单问题变慢，也会诱导模型重复内容。
    # 作用：详细档要求覆盖关键维度和可执行细节，但仍按问题复杂度自然收束。
    return (
        "Provide a thorough, information-dense final answer unless the user explicitly asks for "
        "brevity. Start with the direct answer, then cover the reasoning or evidence, important "
        "details, practical steps or examples, limitations and caveats, and available sources "
        "when relevant. Use meaningful sections for complex questions and connected prose where "
        "it reads better. Do not reduce the answer to a short bullet list, repeat points, "
        "or pad it to a fixed length."
    )


def _answer_contract_instruction(contract: AnswerContract) -> str:
    """Render typed answer requirements without prescribing a rigid template."""
    facets = ", ".join(contract.required_facets) or "the direct answer"
    return (
        f"Answer contract: task type={contract.task_type}; "
        f"complexity={contract.complexity}; cover these relevant dimensions: "
        f"{facets}. Integrate them naturally, omit only dimensions that truly do not apply, "
        "and do not mention this internal contract."
    )


def _append_knowledge_instructions(
    lines: list[str],
    *,
    enable_local_knowledge: bool,
    include_global_knowledge: bool,
    resolved_knowledge_scope: Literal["private", "global", "none"],
    output_role: AgentOutputRole,
) -> None:
    """Append only the instructions matching the knowledge capability installed this turn."""
    if enable_local_knowledge and resolved_knowledge_scope == "private":
        lines.append(
            "Local knowledge uploaded in this conversation is available. Use rag_search for "
            "semantic document evidence. Use graph_search for named-entity relationships, "
            "cross-document links, or multi-hop paths. Do not call both unless both evidence "
            "types are necessary. After a successful Observation, "
            f"{_role_completion_instruction(output_role)}. Preserve available local source "
            "names or pages and never expose raw Observation text. If the knowledge tools return "
            "no relevant evidence, do not answer from general knowledge."
        )
        if include_global_knowledge:
            lines.append(
                "The user explicitly allowed global knowledge for this turn. Use "
                "global_rag_search or global_graph_search only when the current conversation "
                "does not contain enough evidence, and preserve global source citations."
            )
        else:
            lines.append(
                "Global knowledge is not authorized. Never claim to use files from other "
                "conversations or call a global knowledge tool."
            )
    elif enable_local_knowledge and resolved_knowledge_scope == "global":
        lines.append(
            "The current conversation has no indexed document source. The user explicitly "
            "authorized global knowledge, so rag_search and graph_search are deterministically "
            "bound to the global store for this turn. Use rag_search for semantic evidence or "
            "graph_search for named-entity relationships, then cite the returned global source "
            "names or pages. Do not call global_rag_search or global_graph_search because no "
            "duplicate fallback tools are installed."
        )
    elif enable_local_knowledge:
        lines.append(
            "Local knowledge permission is enabled, but the current conversation has no indexed "
            "document source and global knowledge is not authorized. Do not claim that files "
            "were read or answer document-dependent questions from general knowledge."
        )
    else:
        lines.append(
            "Local knowledge access is disabled. Chat cannot access previously uploaded files or "
            "MiniRAG; ask the user to enable local knowledge or upload files on the document "
            "analysis page when their content is needed."
        )


def has_usable_knowledge_evidence(observations: list[str]) -> bool:
    """Distinguish source-bearing Tool evidence from explicit empty/error observations."""
    for observation in observations:
        normalized = " ".join(observation.casefold().split())
        if normalized and not any(
            marker in normalized for marker in NO_KNOWLEDGE_EVIDENCE_MARKERS
        ):
            return True
    return False


def requires_document_evidence(user_message: str) -> bool:
    """Recognize explicit requests to read user-provided documents conservatively."""
    normalized = " ".join(user_message.split())
    if not normalized or not DOCUMENT_EVIDENCE_ACTION_PATTERN.search(normalized):
        return False
    return any(
        pattern.search(normalized)
        for pattern in DOCUMENT_EVIDENCE_REFERENCE_PATTERNS
    )


def document_evidence_required_answer(user_message: str) -> str:
    """Explain the deterministic preflight failure without invoking a model."""
    if any("\u3400" <= character <= "\u9fff" for character in user_message):
        return (
            "这个请求需要读取你上传或指定的文件，但当前聊天没有可访问的文档证据。"
            "请开启 Knowledge；如果需要使用其他会话中的资料，请同时开启 Global。"
            "也可以在 Document analysis 中重新选择或上传文件后运行。"
            "为避免编造文件内容，本次没有调用模型生成答案。"
        )
    return (
        "This request requires reading uploaded or selected files, but this chat has no "
        "accessible document evidence. Enable Knowledge (and Global if files from other "
        "conversations are needed), or select/upload the files in Document analysis. "
        "The model was not called, so no file content was invented."
    )


def no_knowledge_evidence_answer(user_message: str) -> str:
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


def bounded_chat_history(
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
