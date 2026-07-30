"""Reference-concept checks layered on the production answer-quality contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from qwopus_agent.services.answer_quality import (
    AnswerQualityEvaluator,
    AnswerQualityReport,
)
from qwopus_agent.services.orchestration_models import AnswerContract, AnswerPlan


class AnswerBenchmarkCase(BaseModel):
    """One expected final answer with task-specific concepts and safety exclusions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    answer: str
    contract: AnswerContract
    required_concepts: tuple[tuple[str, ...], ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    min_concept_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    has_citations: bool = False


class AnswerBenchmarkReport(BaseModel):
    """Combined production-contract and reference-concept evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    quality: AnswerQualityReport
    concept_recall: float
    missing_concepts: tuple[str, ...] = ()
    forbidden_hits: tuple[str, ...] = ()


def evaluate_answer(
    case: AnswerBenchmarkCase,
    *,
    answer_plan: AnswerPlan | None = None,
) -> AnswerBenchmarkReport:
    """Score a final answer without using the same model as its own judge."""
    quality = AnswerQualityEvaluator().evaluate(
        case.answer,
        case.contract,
        has_citations=case.has_citations,
        answer_plan=answer_plan,
    )
    normalized_answer = _normalize(case.answer)
    missing_concepts = tuple(
        " | ".join(aliases)
        for aliases in case.required_concepts
        if not any(_normalize(alias) in normalized_answer for alias in aliases)
    )
    concept_recall = (
        (len(case.required_concepts) - len(missing_concepts))
        / len(case.required_concepts)
        if case.required_concepts
        else 1.0
    )
    forbidden_hits = tuple(
        phrase
        for phrase in case.forbidden_phrases
        if _normalize(phrase) in normalized_answer
    )
    return AnswerBenchmarkReport(
        passed=(
            quality.passed
            and concept_recall >= case.min_concept_recall
            and not forbidden_hits
        ),
        quality=quality,
        concept_recall=concept_recall,
        missing_concepts=missing_concepts,
        forbidden_hits=forbidden_hits,
    )


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
