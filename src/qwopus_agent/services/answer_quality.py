"""Adaptive final-answer checks used before an optional single refinement."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from qwopus_agent.services.orchestration_models import (
    AnswerContract,
    AnswerPlan,
    ResolvedIntent,
)
from qwopus_agent.utils.token_budget import estimate_tokens

_COMPLEX_TASKS = {"explain", "how_to", "compare", "summarize", "analyze", "report"}
_DETAIL_TOKEN_TARGETS = {
    "explain": 120,
    "how_to": 140,
    "compare": 140,
    "summarize": 120,
    "analyze": 150,
    "report": 220,
}
_INTERNAL_PROCESS_PATTERN = re.compile(
    r"(?im)^\s*(?:thought|observation|tool output|action)\s*:"
)
_STEP_PATTERN = re.compile(
    r"(?m)^\s*(?:\d+[.)、]|[-*]\s+)|首先|其次|然后|最后|\bfirst\b|\bthen\b|\bfinally\b",
    re.I,
)
_COMPARISON_PATTERN = re.compile(
    r"相比|共同|相同|不同|区别|差异|优点|缺点|权衡|"
    r"\b(?:similar|different|whereas|however|advantage|disadvantage|trade-off)\b",
    re.I,
)
_BULLET_LINE_PATTERN = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)、])\s+")
_SPECIFICITY_PATTERN = re.compile(
    r"因为|由于|因此|所以|表明|显示|根据|例如|比如|具体|意味着|"
    r"前提|条件|适合|风险|限制|验证|来源|"
    r"\b(?:because|therefore|evidence|according to|for example|specifically|"
    r"means|condition|prerequisite|risk|limitation|verify|source)\b",
    re.I,
)
_EMPIRICAL_CLAIM_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*%|百分之|"
    r"(?:研究|实验|调查)(?:报告|表明|显示|发现|结果)|"
    r"\b(?:study|research|experiment|survey|benchmark)\s+"
    r"(?:found|shows?|reports?|results?)\b",
    re.I,
)
_SOURCE_REFERENCE_PATTERN = re.compile(
    r"https?://|\b[^\s/\\]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls)\b",
    re.I,
)
_EVIDENCE_FRAMING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:证据|evidence)(?:\*\*)?\s*:?\s*$|"
    r"(?:经验|典型)(?:案例|观察)|文档中(?:指出|说明)|"
    r"\b(?:case study|document(?:ation)? (?:states|shows))\b",
    re.I,
)


class AnswerQualityReport(BaseModel):
    """Machine-testable result of one final-answer review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    issues: tuple[str, ...] = ()
    estimated_tokens: int = 0
    target_tokens: int = 0
    specificity_signals: int = 0


class AnswerQualityEvaluator:
    """Detect concrete answer defects without another model call."""

    def evaluate(
        self,
        answer: str,
        intent: ResolvedIntent | AnswerContract,
        *,
        has_citations: bool = False,
        answer_plan: AnswerPlan | None = None,
    ) -> AnswerQualityReport:
        """Return only actionable gaps relevant to the requested task."""
        stripped = answer.strip()
        token_count = estimate_tokens(stripped)
        specificity_signals = len(_SPECIFICITY_PATTERN.findall(stripped))
        target_tokens = 0
        issues: list[str] = []
        if not stripped:
            issues.append("empty_answer")
        if _INTERNAL_PROCESS_PATTERN.search(stripped):
            issues.append("internal_process_exposed")
        if _has_repeated_paragraph(stripped):
            issues.append("repeated_content")

        contract = (
            intent.answer_contract
            if isinstance(intent, ResolvedIntent)
            else intent
        )
        if (
            stripped
            and contract.response_detail != "concise"
            and contract.complexity != "simple"
            and contract.task_type in _COMPLEX_TASKS
        ):
            target = _adaptive_token_target(contract, answer_plan)
            if contract.response_detail == "balanced":
                target = max(60, target // 2)
            target_tokens = target
            if token_count < target:
                issues.append("insufficient_depth")
            if (
                contract.response_detail == "detailed"
                and specificity_signals < 2
            ):
                # 原因：空泛回答也能靠重复观点超过长度门槛，字符数量不能代表信息密度。
                # 作用：要求至少出现可观察的因果、证据、实例、条件或限制信号。
                issues.append("insufficient_specificity")
            if (
                contract.task_type == "how_to"
                and not _STEP_PATTERN.search(stripped)
            ):
                issues.append("missing_ordered_steps")
            if (
                contract.task_type == "compare"
                and not _COMPARISON_PATTERN.search(stripped)
            ):
                issues.append("missing_comparison")
            if (
                contract.task_type in {"analyze", "report"}
                and token_count < target * 2
                and _content_block_count(stripped) < 2
            ):
                issues.append("missing_analysis_structure")
            if (
                contract.response_detail == "detailed"
                and contract.task_type
                in {"explain", "compare", "summarize", "analyze", "report"}
                and _is_fragmented_bullet_answer(stripped)
            ):
                # 原因：达到长度门槛的短 bullet 堆叠仍可能没有论证主线，读起来像多份草稿拼接。
                # 作用：只对非步骤型复杂回答触发一次修正，要求模型把事实组织成连贯解释。
                issues.append("fragmented_answer")

        if (
            "source-grounded evidence" in contract.required_facets
            and stripped
            and not has_citations
        ):
            issues.append("missing_source_attribution")
        if (
            stripped
            and not has_citations
            and _SOURCE_REFERENCE_PATTERN.search(stripped)
        ):
            issues.append("ungrounded_source_reference")
        if (
            stripped
            and not has_citations
            and _EVIDENCE_FRAMING_PATTERN.search(stripped)
        ):
            issues.append("unsupported_evidence_framing")
        if (
            stripped
            and not has_citations
            and _EMPIRICAL_CLAIM_PATTERN.search(stripped)
        ):
            # 原因：弱模型会在“请详细说明”时用虚构百分比或研究报告制造具体感。
            # 作用：没有 Tool 来源时触发一次无工具修正，删除伪证据并保留可验证的设计推理。
            issues.append("unsupported_empirical_claims")

        return AnswerQualityReport(
            passed=not issues,
            issues=tuple(dict.fromkeys(issues)),
            estimated_tokens=token_count,
            target_tokens=target_tokens,
            specificity_signals=specificity_signals,
        )


def strip_unsupported_evidence_lines(answer: str) -> str:
    """Remove source-like empirical lines when no trusted source exists."""
    # 原因：一次修正后的弱模型仍可能虚构文件名和实验数据，继续调用模型会形成循环。
    # 作用：只删除包含伪来源或实证声明的整行，保留其余架构分析和验证建议。
    return "\n".join(
        line
        for line in answer.splitlines()
        if not _EMPIRICAL_CLAIM_PATTERN.search(line)
        and not _SOURCE_REFERENCE_PATTERN.search(line)
        and not _EVIDENCE_FRAMING_PATTERN.search(line)
    ).strip()


def _adaptive_token_target(
    contract: AnswerContract,
    answer_plan: AnswerPlan | None,
) -> int:
    """Scale depth with required content units rather than a global word quota."""
    base = _DETAIL_TOKEN_TARGETS[contract.task_type]
    unit_count = (
        len(answer_plan.plan_items)
        if answer_plan is not None and answer_plan.plan_items
        else len(contract.required_facets) + 1
    )
    return base + max(0, unit_count - 4) * 20


def _content_block_count(answer: str) -> int:
    return len(
        [
            block
            for block in re.split(
                r"\n\s*\n|^\s*#{1,6}\s+",
                answer,
                flags=re.MULTILINE,
            )
            if block.strip()
        ]
    )


def _has_repeated_paragraph(answer: str) -> bool:
    paragraphs = [
        " ".join(block.casefold().split())
        for block in re.split(r"\n\s*\n", answer)
        if len(" ".join(block.split())) >= 40
    ]
    return len(paragraphs) != len(set(paragraphs))


def _is_fragmented_bullet_answer(answer: str) -> bool:
    bullet_count = len(_BULLET_LINE_PATTERN.findall(answer))
    if bullet_count < 6:
        return False
    prose_blocks = [
        block
        for block in re.split(r"\n\s*\n", answer)
        if len(" ".join(block.split())) >= 80
        and not _BULLET_LINE_PATTERN.match(block)
    ]
    return len(prose_blocks) < 2
