"""Configurable recipe contract for source-grounded report composition.

A :class:`ReportRecipe` makes the grounded multi-document report pipeline work
for arbitrary source collections while allowing a domain-specific recipe to
override parsing labels, item ordering, reference validation, and renderers.

The generic :data:`DEFAULT_RECIPE` treats every parser file as one
independently rendered source slot, orders lesson-named files by lesson number,
and validates scripture references against each source's allowed verse range.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwopus_agent.reports.grounded_facts import SourceGroundingSpec

__all__ = [
    "ComposerThresholds",
    "ReportRecipe",
    "SectionKind",
    "SourceFactLabels",
    "default_recipe",
]


class SectionKind(Enum):
    """One requested report section category resolved by a recipe classifier."""

    SOURCE_UNDERSTANDING = "source_understanding"
    STRATEGY = "strategy"
    OUTLINE = "outline"
    PARAGRAPH = "paragraph"
    FULL_DRAFT = "full_draft"
    EXAMPLES = "examples"
    DRAFT_REVIEW = "draft_review"
    CHECKLIST = "checklist"


@dataclass(frozen=True)
class ComposerThresholds:
    """Gates that decide whether the deterministic grounded composer applies."""

    min_parser_files: int = 2
    min_sections: int = 6


@dataclass(frozen=True)
class SourceFactLabels:
    """Line-prefix alternatives recognized for one SOURCE_FACTS field kind.

    Every tuple holds case-insensitive prefix patterns.  A label kind the
    recipe does not use (for example ``quote_line`` in the generic recipe) is
    an empty tuple.
    """

    document_heading: tuple[str, ...]
    topic_line: tuple[str, ...]
    topic_continuation: tuple[str, ...]
    quote_line: tuple[str, ...]
    opening_line: tuple[str, ...]
    topic_stop_labels: tuple[str, ...] = ()
    # The stable fact key written to SOURCE_FACTS for a matched quote/scripture line.
    quote_fact_key: str = "quote_line"


@dataclass(frozen=True)
class ReportRecipe:
    """One self-contained policy for grounded multi-document report building.

    Data fields are words, patterns, and thresholds.  Strategy fields are
    functions so that domain parsing and rendering stay behind the same object
    instead of leaking back into the shared pipeline.
    """

    source_fact_labels: SourceFactLabels
    rubric_markers: tuple[str, ...]
    invented_score_pattern: re.Pattern[str]
    reference_pattern: re.Pattern[str]
    all_source_request_pattern: re.Pattern[str]
    grounding_rules_text: str
    evidence_section_markers: tuple[str, ...]
    evidence_claim_boost_terms: tuple[str, ...]

    # Strategy fields -------------------------------------------------------
    item_label_from_name: Callable[[str], str | None]
    item_aliases: Callable[[str], tuple[str, ...]]
    render_item_heading: Callable[[SourceGroundingSpec], str]
    reference_key: Callable[[str], tuple[str, tuple[int, ...]] | None]
    reference_is_supported: Callable[
        [tuple[str, tuple[int, ...]] | None, frozenset[tuple[str, tuple[int, ...]]]],
        bool,
    ]
    build_grounding_specs: Callable[[list[str], str], tuple[SourceGroundingSpec, ...]]
    render_fallback: Callable[[SourceGroundingSpec], str]
    section_classifier: Callable[[str], SectionKind | None]
    renderers: Mapping[SectionKind, Callable[..., str]]
    validate_candidate_issues: Callable[..., tuple[str, ...]]
    composer_thresholds: ComposerThresholds = ComposerThresholds()


def default_recipe() -> ReportRecipe:
    """Return the canonical generic recipe, resolving it lazily.

    The generic recipe binds rendering functions defined in
    :mod:`qwopus_agent.reports.grounded`, which itself imports parsing helpers
    from :mod:`qwopus_agent.reports.grounded_facts`.  Resolving lazily here
    keeps that import graph acyclic at module load time.
    """
    from qwopus_agent.reports import grounded as _grounded

    return _grounded.DEFAULT_RECIPE
