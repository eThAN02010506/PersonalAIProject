"""Resolve short or contextual user text into one explicit Agent objective."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from qwopus_agent.services.orchestration_models import (
    AnswerContract,
    ContextReference,
    ContextSnapshot,
    ConversationTaskState,
    InterpretationMode,
    ResolvedIntent,
    TaskType,
)

_CONTINUATION_PATTERNS = (
    r"^\s*(?:继续|接着|往下|再详细|更具体|展开|补充|按之前|照刚才)",
    r"^\s*(?:continue|go on|keep going|more detail|be more specific|expand|elaborate)\b",
)
_PRIOR_REFERENCE_PATTERNS = (
    r"(?:这个|该|上述|前面|刚才|之前)(?:方案|问题|部分|结果|回答|分析|计划|内容)",
    r"\b(?:this|that|the previous|the above)\s+"
    r"(?:approach|issue|part|result|answer|analysis|plan|content)\b",
)
_PLURAL_DOCUMENT_PATTERNS = (
    r"(?:这些|上述|全部|所有)(?:文档|文件|资料)",
    r"\b(?:these|all|the selected)\s+(?:documents|files|sources)\b",
)
_SINGULAR_DOCUMENT_PATTERNS = (
    r"(?:这个|该|这份|上述)(?:文档|文件|资料)",
    r"\b(?:this|that|the uploaded|the selected)\s+(?:document|file|source)\b",
)
_ORDINAL_DOCUMENTS = (
    (re.compile(r"(?:第一份|第一个)(?:文档|文件|资料)?"), 0),
    (re.compile(r"(?:第二份|第二个)(?:文档|文件|资料)?"), 1),
    (re.compile(r"(?:第三份|第三个)(?:文档|文件|资料)?"), 2),
    (re.compile(r"\b(?:first)\s+(?:document|file|source)\b", re.I), 0),
    (re.compile(r"\b(?:second)\s+(?:document|file|source)\b", re.I), 1),
    (re.compile(r"\b(?:third)\s+(?:document|file|source)\b", re.I), 2),
)
_BRIEF_REQUEST_PATTERNS = (
    r"简短|简洁|一句话|只要结论|不要展开",
    r"\b(?:brief|briefly|concise|concisely|short answer|just the answer)\b",
)
_DETAILED_REQUEST_PATTERNS = (
    r"详细|具体|深入|全面|展开|完整",
    r"\b(?:detailed|in detail|specific|in depth|comprehensive|expand|elaborate)\b",
)

_TASK_PATTERNS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
    (
        "compare",
        (
            r"对比|比较|区别|差异|异同|优劣",
            r"\bcompare\b|\bcomparison\b|\bdifference(?:s)?\b|\bversus\b|\bvs\.?\b",
        ),
    ),
    (
        "report",
        (
            r"报告|方案书|研究稿|调研稿",
            r"\breport\b|\bbriefing\b|\bwhite\s*paper\b",
        ),
    ),
    (
        "summarize",
        (
            r"总结|概括|摘要|提炼|综述",
            r"\bsummar(?:y|ize|ise)\b|\boverview\b|\brecap\b",
        ),
    ),
    (
        "how_to",
        (
            r"如何|怎么|怎样|步骤|教程|流程",
            r"\bhow\s+(?:to|should|can|do|would)\b|\bsteps?\b|\btutorial\b|\bprocedure\b",
        ),
    ),
    (
        "explain",
        (
            r"为什么|为何|解释|原理|是什么|什么意思",
            r"\bwhy\b|\bexplain\b|\bwhat\s+is\b|\bhow\s+does\b|\bmeaning\b",
        ),
    ),
    (
        "analyze",
        (
            r"分析|评估|审查|诊断|风险|问题|影响",
            r"\banaly[sz]e\b|\bevaluate\b|\breview\b|\bdiagnos(?:e|is)\b|\brisk\b",
        ),
    ),
)

_FACETS: dict[TaskType, tuple[str, ...]] = {
    "answer": ("direct answer", "supporting explanation", "practical implications"),
    "explain": ("conclusion", "mechanism", "examples", "limitations"),
    "how_to": ("prerequisites", "ordered steps", "verification", "common failures"),
    "compare": (
        "comparison criteria",
        "similarities",
        "differences",
        "trade-offs",
        "conclusion",
    ),
    "summarize": (
        "central theme",
        "key facts",
        "important relationships",
        "implications",
    ),
    "analyze": ("findings", "evidence", "implications", "risks", "limitations"),
    "report": (
        "executive summary",
        "findings",
        "evidence",
        "recommendations",
        "limitations",
    ),
    "continue": ("continued result", "new detail", "remaining limitations"),
}


def build_context_snapshot(
    *,
    conversation_id: str | None,
    task_state: ConversationTaskState | None = None,
    document_sources: Iterable[str] = (),
    active_skill_names: Iterable[str] = (),
) -> ContextSnapshot:
    """Build one bounded snapshot from persisted state and current resources."""
    state = task_state or ConversationTaskState()
    # 原因：旧任务保存的文档和当前知识库清单可能重叠，重复项会误导指代解析。
    # 作用：保留首次出现顺序并合并两类来源，让“第二份文档”具有稳定含义。
    sources = _unique((*state.active_document_sources, *document_sources))
    return ContextSnapshot(
        conversation_id=conversation_id,
        previous_objective=state.last_successful_objective,
        open_tasks=state.open_tasks,
        document_sources=sources,
        active_skill_names=_unique(active_skill_names),
    )


class IntentResolver:
    """Deterministically resolve context before Planner or model execution."""

    def resolve(
        self,
        request: str,
        *,
        snapshot: ContextSnapshot | None = None,
        interpretation_mode: InterpretationMode = "contextual",
        response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    ) -> ResolvedIntent:
        """Return an explicit objective, references, and answer requirements."""
        original = " ".join(request.split())
        if not original:
            raise ValueError("The request must not be blank.")
        context = snapshot or ContextSnapshot()
        references, document_question = _resolve_document_references(
            original,
            context.document_sources,
        )
        inherited = False
        continuation = _matches_any(original, _CONTINUATION_PATTERNS)
        prior_reference = _matches_any(original, _PRIOR_REFERENCE_PATTERNS)
        previous = context.previous_objective

        if continuation or prior_reference:
            if interpretation_mode != "precise" and previous:
                inherited = True
                references.append(
                    ContextReference(
                        kind="task",
                        identifier="previous-objective",
                        label=previous,
                    )
                )
            elif document_question is None:
                # 原因：Precise 模式禁止借用上一任务，孤立的“再详细一点”便没有可执行对象。
                # 作用：所有未绑定上下文的续写或指代请求先澄清，避免模型自行猜测主题。
                document_question = _localized_question(
                    original,
                    chinese=(
                        "请说明你希望继续或展开的具体任务；"
                        "当前对话没有可承接的上一项任务。"
                    ),
                    english=(
                        "Please specify the task you want to continue or expand; "
                        "this conversation has no previous task to resume."
                    ),
                )

        task_basis = previous if inherited and previous else original
        task_type = _classify_task(task_basis)
        effective_detail = _effective_response_detail(original, response_detail)
        objective = _operational_objective(
            original,
            previous=previous if inherited else None,
            document_references=references,
        )
        facets = _answer_facets(
            task_type,
            response_detail=effective_detail,
            interpretation_mode=interpretation_mode,
            has_documents=any(item.kind == "document" for item in references),
        )
        complexity = _task_complexity(
            original,
            task_type=task_type,
            interpretation_mode=interpretation_mode,
            document_count=sum(
                item.kind == "document" for item in references
            ),
        )
        assumptions: list[str] = []
        if inherited:
            assumptions.append("The current message refines or continues the previous task.")
        if interpretation_mode == "exploratory":
            assumptions.append(
                "The answer may cover adjacent implications when they materially help the task."
            )

        return ResolvedIntent(
            original_request=original,
            operational_objective=objective,
            interpretation_mode=interpretation_mode,
            task_type=task_type,
            confidence=_confidence(
                inherited=inherited,
                document_question=document_question,
                references=references,
            ),
            context_references=tuple(references),
            assumptions=tuple(assumptions),
            requires_clarification=document_question is not None,
            clarification_question=document_question,
            answer_contract=AnswerContract(
                task_type=task_type,
                complexity=complexity,
                response_detail=effective_detail,
                required_facets=facets,
            ),
        )


def _resolve_document_references(
    request: str,
    sources: tuple[str, ...],
) -> tuple[list[ContextReference], str | None]:
    references: list[ContextReference] = []
    request_folded = request.casefold()
    for source in sources:
        if source.casefold() in request_folded:
            references.append(_document_reference(source))

    for pattern, index in _ORDINAL_DOCUMENTS:
        if pattern.search(request):
            if index >= len(sources):
                return references, _localized_question(
                    request,
                    chinese=f"当前对话中没有第 {index + 1} 份文档，请重新选择。",
                    english=(
                        f"This conversation does not have document {index + 1}; "
                        "please choose another document."
                    ),
                )
            references.append(_document_reference(sources[index]))

    if _matches_any(request, _PLURAL_DOCUMENT_PATTERNS):
        references.extend(_document_reference(source) for source in sources)
        if not sources:
            return references, _missing_document_question(request)
    elif _matches_any(request, _SINGULAR_DOCUMENT_PATTERNS):
        if len(sources) == 1:
            references.append(_document_reference(sources[0]))
        elif len(sources) > 1 and not references:
            choices = "、".join(sources[:5])
            return references, _localized_question(
                request,
                chinese=f"你指的是哪一份文档？当前可选：{choices}",
                english=f"Which document do you mean? Available documents: {choices}",
            )
        elif not sources:
            return references, _missing_document_question(request)

    return list(_unique_references(references)), None


def _operational_objective(
    original: str,
    *,
    previous: str | None,
    document_references: list[ContextReference],
) -> str:
    parts: list[str] = []
    if previous:
        parts.extend((f"Previous objective: {previous}", f"Current instruction: {original}"))
    else:
        parts.append(original)
    document_labels = [
        reference.label
        for reference in document_references
        if reference.kind == "document"
    ]
    if document_labels:
        parts.append("Authorized document scope: " + ", ".join(document_labels))
    return "\n".join(parts)


def _classify_task(request: str) -> TaskType:
    for task_type, patterns in _TASK_PATTERNS:
        if _matches_any(request, patterns):
            return task_type
    return "answer"


def _answer_facets(
    task_type: TaskType,
    *,
    response_detail: Literal["concise", "balanced", "detailed"],
    interpretation_mode: InterpretationMode,
    has_documents: bool,
) -> tuple[str, ...]:
    facets = list(_FACETS[task_type])
    if response_detail == "concise":
        facets = facets[:2]
    elif response_detail == "balanced":
        facets = facets[: min(3, len(facets))]
    if has_documents:
        facets.append("source-grounded evidence")
    if interpretation_mode == "exploratory":
        facets.extend(("alternatives", "second-order effects"))
    return _unique(facets)


def _effective_response_detail(
    request: str,
    configured: Literal["concise", "balanced", "detailed"],
) -> Literal["concise", "balanced", "detailed"]:
    """Let an explicit current instruction override the persistent UI preference."""
    if _matches_any(request, _BRIEF_REQUEST_PATTERNS):
        return "concise"
    if _matches_any(request, _DETAILED_REQUEST_PATTERNS):
        return "detailed"
    return configured


def _task_complexity(
    request: str,
    *,
    task_type: TaskType,
    interpretation_mode: InterpretationMode,
    document_count: int,
) -> Literal["simple", "standard", "complex"]:
    if interpretation_mode == "exploratory" or document_count > 1:
        return "complex"
    if task_type in {"how_to", "compare", "analyze", "report"}:
        return "complex"
    if document_count == 1 and task_type == "summarize":
        return "complex"
    if _matches_any(request, _DETAILED_REQUEST_PATTERNS):
        return "complex"
    if (
        task_type in {"answer", "explain"}
        and len(request) <= 40
        and re.search(r"是什么|什么意思|\bwhat\s+is\b|\bmeaning\b", request, re.I)
    ):
        return "simple"
    return "standard"


def _confidence(
    *,
    inherited: bool,
    document_question: str | None,
    references: list[ContextReference],
) -> float:
    if document_question is not None:
        return 0.35
    if inherited:
        return 0.95
    if references:
        return 0.9
    return 1.0


def _document_reference(source: str) -> ContextReference:
    return ContextReference(kind="document", identifier=source, label=source)


def _matches_any(value: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, value, re.I) is not None for pattern in patterns)


def _missing_document_question(request: str) -> str:
    return _localized_question(
        request,
        chinese="当前对话还没有可引用的文档，请先上传或选择文档。",
        english=(
            "This conversation has no document to reference yet; "
            "please upload or select one first."
        ),
    )


def _localized_question(request: str, *, chinese: str, english: str) -> str:
    return chinese if re.search(r"[\u3400-\u9fff]", request) else english


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _unique_references(
    references: Iterable[ContextReference],
) -> tuple[ContextReference, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[ContextReference] = []
    for reference in references:
        key = (reference.kind, reference.identifier)
        if key not in seen:
            seen.add(key)
            unique.append(reference)
    return tuple(unique)
