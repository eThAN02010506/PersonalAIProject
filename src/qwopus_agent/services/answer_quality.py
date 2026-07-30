"""Adaptive final-answer checks used before an optional single refinement."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from qwopus_agent.services.orchestration_models import AnswerContract, ResolvedIntent
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


class AnswerQualityReport(BaseModel):
    """Machine-testable result of one final-answer review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    issues: tuple[str, ...] = ()
    estimated_tokens: int = 0


class AnswerQualityEvaluator:
    """Detect concrete answer defects without another model call."""

    def evaluate(
        self,
        answer: str,
        intent: ResolvedIntent | AnswerContract,
        *,
        has_citations: bool = False,
    ) -> AnswerQualityReport:
        """Return only actionable gaps relevant to the requested task."""
        stripped = answer.strip()
        token_count = estimate_tokens(stripped)
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
            target = _DETAIL_TOKEN_TARGETS[contract.task_type]
            if contract.response_detail == "balanced":
                target = max(60, target // 2)
            if token_count < target:
                issues.append("insufficient_depth")
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

        return AnswerQualityReport(
            passed=not issues,
            issues=tuple(dict.fromkeys(issues)),
            estimated_tokens=token_count,
        )


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
