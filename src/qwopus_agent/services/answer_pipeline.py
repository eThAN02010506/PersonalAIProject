"""Typed planning, evidence compaction, and review boundaries for final answers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qwopus_agent.services.orchestration_models import (
    AnswerContract,
    AnswerPlan,
    EvidenceFact,
    EvidenceLedger,
    EvidencePacket,
    EvidenceReview,
    SourceCitation,
    TaskType,
)

TModel = TypeVar("TModel", bound=BaseModel)

_DEPTH_QUESTIONS: dict[TaskType, tuple[str, ...]] = {
    "answer": (
        "What context materially changes the direct answer?",
        "What practical implication should the user act on?",
    ),
    "explain": (
        "What mechanism or causal relationship explains the result?",
        "What concrete example makes the mechanism clear?",
        "Where does the explanation stop applying?",
    ),
    "how_to": (
        "What prerequisites must be true before starting?",
        "How can each important step be verified?",
        "What common failure modes and recovery steps matter?",
    ),
    "compare": (
        "Which criteria materially change the choice?",
        "What trade-offs and edge cases distinguish the options?",
        "Under which conditions should each option be preferred?",
    ),
    "summarize": (
        "Which facts are central rather than merely present?",
        "How do the important facts relate to one another?",
        "What implication follows from the source as a whole?",
    ),
    "analyze": (
        "What evidence supports each major finding?",
        "What causes, consequences, risks, and second-order effects matter?",
        "What limitations or alternative explanations remain?",
    ),
    "report": (
        "What findings deserve executive priority?",
        "Which evidence supports each recommendation?",
        "What risks, limitations, and next actions must be explicit?",
    ),
    "continue": (
        "What material detail was missing from the previous answer?",
        "What new implication or limitation should now be added?",
    ),
}


class _EvidenceFactDraft(BaseModel):
    """Strict model-facing shape before trusted task metadata is attached."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=1_000)
    support: str = Field(min_length=1, max_length=4_000)
    sources: tuple[str, ...] = Field(default=(), max_length=12)
    confidence: str = "medium"


class _EvidencePacketDraft(BaseModel):
    """Strict JSON envelope requested from an evidence worker."""

    model_config = ConfigDict(extra="forbid")

    facts: tuple[_EvidenceFactDraft, ...] = Field(default=(), max_length=16)
    limitations: tuple[str, ...] = Field(default=(), max_length=8)


def build_answer_plan(
    objective: str,
    contract: AnswerContract,
) -> AnswerPlan:
    """Create one stable content plan without spending another model call."""
    sections = _unique(("direct answer", *contract.required_facets))
    detailed = contract.response_detail == "detailed"
    # 原因：固定最低字数会制造重复，而用户需要的是更深的事实、机制和边界。
    # 作用：详细档把任务特定的深度问题交给证据收集和最终综合两个阶段。
    depth_questions = _DEPTH_QUESTIONS[contract.task_type] if detailed else ()
    style_rules: tuple[str, ...] = (
        "Lead with the direct conclusion before supporting detail.",
        "Keep every section tied to the objective and avoid repeated claims.",
        "Prefer connected explanation over a list of disconnected bullets.",
    )
    if detailed:
        style_rules += (
            "Use concrete evidence, examples, edge cases, and actionable implications "
            "when they materially help.",
            "Add detail through specificity rather than padding or a fixed word count.",
        )
    return AnswerPlan(
        objective=objective.strip(),
        task_type=contract.task_type,
        response_detail=contract.response_detail,
        central_goal=(
            "Resolve the user's request around one direct conclusion, then support it "
            "with only relevant evidence and implications."
        ),
        required_sections=sections,
        depth_questions=depth_questions,
        style_rules=style_rules,
    )


def parse_evidence_packet(
    content: str,
    *,
    task_id: str,
    agent_name: str,
    citations: Iterable[SourceCitation] = (),
    fallback_confidence: float = 0.5,
    trust_declared_sources: bool = True,
) -> EvidencePacket:
    """Parse strict worker JSON and retain a bounded fallback for weaker models."""
    citation_sources = _unique(_citation_label(item) for item in citations)
    draft = _parse_json_model(content, _EvidencePacketDraft)
    if draft is not None:
        facts = tuple(
            EvidenceFact(
                claim=fact.claim,
                support=fact.support,
                sources=(
                    _unique((*fact.sources, *citation_sources))
                    if trust_declared_sources
                    else citation_sources
                ),
                confidence=_grounded_confidence(
                    _confidence_label(fact.confidence, fallback_confidence),
                    has_runtime_sources=bool(citation_sources),
                    trust_declared_sources=trust_declared_sources,
                ),
            )
            for fact in draft.facts
        )
        limitations = _unique(draft.limitations)
    else:
        compact = " ".join(content.split())[:4_000] or "No usable evidence was returned."
        # 原因：任意兼容模型不一定稳定输出 JSON，格式失败不能抹掉已经取得的 Tool 证据。
        # 作用：把非结构化结果降级为单条有界事实，后续仍可审核且不会直接展示给用户。
        facts = (
            EvidenceFact(
                claim=_first_sentence(compact)[:1_000],
                support=compact,
                sources=citation_sources,
                confidence=_confidence_label("", fallback_confidence),
            ),
        )
        limitations = ("Worker output required deterministic evidence fallback.",)
    if not trust_declared_sources and not citation_sources:
        limitations = _unique(
            (
                *limitations,
                "No tool-grounded source was available; claims are model analysis.",
            )
        )
    return EvidencePacket(
        task_id=task_id,
        agent_name=agent_name,
        facts=facts,
        limitations=limitations,
    )


def build_evidence_ledger(
    packets: Iterable[EvidencePacket],
) -> EvidenceLedger:
    """Merge evidence packets by semantic text identity while preserving sources."""
    merged: dict[str, EvidenceFact] = {}
    limitations: list[str] = []
    for packet in packets:
        limitations.extend(packet.limitations)
        for fact in packet.facts:
            key = _fact_key(fact)
            previous = merged.get(key)
            if previous is None:
                merged[key] = fact
                continue
            merged[key] = previous.model_copy(
                update={
                    "sources": _unique((*previous.sources, *fact.sources)),
                    "confidence": _stronger_confidence(
                        previous.confidence,
                        fact.confidence,
                    ),
                }
            )
    facts = sorted(
        merged.values(),
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}[item.confidence],
            -len(item.sources),
        ),
    )
    return EvidenceLedger(
        facts=tuple(facts[:32]),
        limitations=_unique(limitations)[:16],
    )


def parse_evidence_review(content: str) -> EvidenceReview:
    """Accept one strict review object and degrade to an explicit neutral review."""
    review = _parse_json_model(content, EvidenceReview)
    if review is not None:
        return review
    compact = " ".join(content.split())[:2_000]
    return EvidenceReview(
        resolution=compact,
        unsupported_claims=(
            "Reviewer output was not structured; no automatic gap-fill decision was made.",
        ),
    )


def render_answer_plan(plan: AnswerPlan) -> str:
    """Render an exact bounded plan for a model prompt or Debug Console."""
    return plan.model_dump_json(indent=2)


def render_evidence_ledger(ledger: EvidenceLedger) -> str:
    """Render only deduplicated evidence, never complete worker transcripts."""
    return ledger.model_dump_json(indent=2)


def render_evidence_review(review: EvidenceReview) -> str:
    """Render the structured reviewer decision for final synthesis."""
    return review.model_dump_json(indent=2)


def _parse_json_model(
    content: str,
    model_type: type[TModel],
) -> TModel | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(content[index:])
            return model_type.model_validate(value)
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def _citation_label(citation: SourceCitation) -> str:
    if citation.url:
        return citation.url
    if citation.page:
        return f"{citation.source}, page {citation.page}"
    return citation.source


def _confidence_label(
    value: str,
    fallback: float,
) -> Literal["low", "medium", "high"]:
    normalized = value.casefold().strip()
    if normalized in {"low", "medium", "high"}:
        return cast(Literal["low", "medium", "high"], normalized)
    if fallback >= 0.8:
        return "high"
    if fallback < 0.45:
        return "low"
    return "medium"


def _stronger_confidence(
    left: Literal["low", "medium", "high"],
    right: Literal["low", "medium", "high"],
) -> Literal["low", "medium", "high"]:
    order = {"low": 0, "medium": 1, "high": 2}
    return left if order[left] >= order[right] else right


def _grounded_confidence(
    confidence: Literal["low", "medium", "high"],
    *,
    has_runtime_sources: bool,
    trust_declared_sources: bool,
) -> Literal["low", "medium", "high"]:
    if not trust_declared_sources and not has_runtime_sources and confidence == "high":
        return "medium"
    return confidence


def _fact_key(fact: EvidenceFact) -> str:
    return re.sub(r"\W+", "", f"{fact.claim} {fact.support}".casefold())[:2_000]


def _first_sentence(content: str) -> str:
    match = re.search(r"^.+?(?:[。！？.!?](?:\s|$)|$)", content)
    return match.group(0).strip() if match else content


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
