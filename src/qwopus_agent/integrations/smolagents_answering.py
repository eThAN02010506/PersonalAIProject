"""Final-answer policies used by the smolagents runtime."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from qwopus_agent.services.answer_quality import AnswerQualityEvaluator
from qwopus_agent.services.orchestration_models import (
    AgentOutputRole,
    AnswerContract,
    AnswerPlan,
)

QUALITY_REPAIR_INSTRUCTIONS = {
    "insufficient_depth": (
        "fully develop every relevant answer-plan item with its conclusion, why or how it "
        "follows, concrete support or an example, and a useful implication, condition, or "
        "limitation; do not merely add headings or restate the draft"
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


def role_refinement_prompt(
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
            f"- {issue}: {QUALITY_REPAIR_INSTRUCTIONS.get(issue, 'correct this defect')}"
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


def build_answer_quality_checks(
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
            has_citations=answer_contains_source(answer),
        )
        return report.passed

    return [qwopus_answer_quality]


def answer_quality_issues(
    answer: str,
    contract: AnswerContract | None,
    answer_plan: AnswerPlan | None = None,
    *,
    has_citations: bool | None = None,
) -> tuple[str, ...]:
    """Return actionable final-answer defects for one retry prompt."""
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
            answer_contains_source(answer)
            if has_citations is None
            else has_citations
        ),
        answer_plan=answer_plan,
    ).issues


def answer_contains_source(answer: str) -> bool:
    """Detect whether an answer names a URL or local document source."""
    return bool(
        re.search(r"https?://", answer, re.I)
        or re.search(
            r"\b[^\s/\\]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls)\b",
            answer,
            re.I,
        )
    )


def answer_has_grounded_source(
    answer: str,
    observations: list[str],
) -> bool:
    """Require a source token to appear in both the answer and Tool evidence."""
    answer_sources = source_tokens(answer)
    observation_sources = source_tokens("\n".join(observations))
    return bool(answer_sources.intersection(observation_sources))


def source_tokens(content: str) -> set[str]:
    """Extract comparable source identifiers from prose and Tool evidence."""
    return {
        match.casefold()
        for match in re.findall(
            r"https?://[^\s)\]>`\"']+|"
            r"\b[^\s/\\]+\.(?:pdf|docx|md|txt|png|jpe?g|csv|xlsx?|xls)\b",
            content,
            re.I,
        )
    }
