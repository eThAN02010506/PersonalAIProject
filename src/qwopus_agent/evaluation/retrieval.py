"""Reference-source metrics for MiniRAG retrieval regression cases."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

_SOURCE_PATTERN = re.compile(r"(?:^|\[|\s)Source:\s*([^|\]\r\n]+)", re.IGNORECASE)


class RetrievalBenchmarkCase(BaseModel):
    """Expected source behavior for one fixed query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_sources: tuple[str, ...] = Field(min_length=1)
    forbidden_sources: tuple[str, ...] = ()
    min_source_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_source_precision: float = Field(default=0.5, ge=0.0, le=1.0)
    min_reciprocal_rank: float = Field(default=0.5, ge=0.0, le=1.0)


class RetrievalBenchmarkReport(BaseModel):
    """Source-level retrieval scores that do not require another LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    source_recall: float
    source_precision: float
    reciprocal_rank: float
    retrieved_sources: tuple[str, ...] = ()
    missing_sources: tuple[str, ...] = ()
    forbidden_hits: tuple[str, ...] = ()


def evaluate_retrieval(
    case: RetrievalBenchmarkCase,
    results: list[str],
) -> RetrievalBenchmarkReport:
    """Compare rendered MiniRAG results with an explicit source reference set."""
    ranked_sources = [_extract_sources(result) for result in results]
    retrieved_sources = tuple(
        dict.fromkeys(source for sources in ranked_sources for source in sources)
    )
    expected = {_normalize(source): source for source in case.expected_sources}
    retrieved = {_normalize(source): source for source in retrieved_sources}
    missing = tuple(
        source for key, source in expected.items() if key not in retrieved
    )
    relevant_count = len(expected.keys() & retrieved.keys())
    source_recall = relevant_count / len(expected)
    source_precision = (
        relevant_count / len(retrieved)
        if retrieved
        else 0.0
    )
    first_relevant_rank = next(
        (
            rank
            for rank, sources in enumerate(ranked_sources, start=1)
            if {_normalize(source) for source in sources} & expected.keys()
        ),
        None,
    )
    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )
    forbidden = {_normalize(source): source for source in case.forbidden_sources}
    forbidden_hits = tuple(
        source for key, source in forbidden.items() if key in retrieved
    )
    passed = (
        source_recall >= case.min_source_recall
        and source_precision >= case.min_source_precision
        and reciprocal_rank >= case.min_reciprocal_rank
        and not forbidden_hits
    )
    return RetrievalBenchmarkReport(
        passed=passed,
        source_recall=source_recall,
        source_precision=source_precision,
        reciprocal_rank=reciprocal_rank,
        retrieved_sources=retrieved_sources,
        missing_sources=missing,
        forbidden_hits=forbidden_hits,
    )


def _extract_sources(result: str) -> tuple[str, ...]:
    """Preserve source order from vector chunks and multi-source graph paths."""
    return tuple(
        dict.fromkeys(match.strip() for match in _SOURCE_PATTERN.findall(result))
    )


def _normalize(source: str) -> str:
    return " ".join(source.casefold().split())
