"""Validation and deterministic repair for structured grounded reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from qwopus_agent.integrations.smolagents_debug import extract_agent_tool_calls
from qwopus_agent.reports import grounded

_ALL_SOURCE_REQUEST_PATTERN = grounded._ALL_SOURCE_REQUEST_PATTERN
_SCRIPTURE_REFERENCE_PATTERN = grounded._SCRIPTURE_REFERENCE_PATTERN
_LessonGroundingSpec = grounded._LessonGroundingSpec
_canonical_lesson_heading = grounded._canonical_lesson_heading
_lesson_grounding_specs = grounded._lesson_grounding_specs
_lesson_number_from_label = grounded._lesson_number_from_label
_normalized_fact_text = grounded._normalized_fact_text
_render_grounded_lesson_fallback = grounded._render_grounded_lesson_fallback
_render_grounded_source_inventory = grounded._render_grounded_source_inventory
_scripture_reference_is_supported = grounded._scripture_reference_is_supported
_scripture_reference_key = grounded._scripture_reference_key
_source_answer_label = grounded._source_answer_label
_title_is_source_understanding = grounded._title_is_source_understanding
_title_requires_full_draft = grounded._title_requires_full_draft
_extract_agent_tool_calls = extract_agent_tool_calls


@dataclass(frozen=True)
class _NumberedSectionSpan:
    """One top-level requested report section inside a model answer."""

    number: int
    start: int
    end: int
    body: str


@dataclass(frozen=True)
class _LessonSubsection:
    """One Markdown lesson subsection parsed from a full Draft section."""

    number: int
    start: int
    end: int
    heading: str
    body: str


_ANSWER_SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(?:(?P<hashes>#{1,6})\s*|(?P<bold>\*\*))"
    r"(?P<number>\d{1,2})\s*[.、．:：)]\s*[^\r\n]+"
)
_SECTION_PLACEHOLDER_PATTERN = re.compile(
    r"(?is)^\s*(?:[-*+]\s*)?(?:"
    r"略|省略|待补充|待完成|稍后补充|暂无|无内容|"
    r"omitted|todo|tbd|placeholder|to\s+be\s+(?:completed|added)"
    r")[\s。.!！…-]*$"
)
_SECTION_LIST_ITEM_PATTERN = re.compile(
    r"(?m)^\s*(?:[-*+]|\d{1,2}[.、)]|[✅☑✓])\s*\S+"
)
_DRAFT_OMISSION_PATTERN = re.compile(
    r"以下为示例|完整报告请|"
    r"其余.{0,24}(?:同样|同理|类似|按).{0,24}(?:展开|格式|撰写|补充)|"
    r"\b(?:remaining|other)\b.{0,32}\b(?:same|similar)\b.{0,24}"
    r"\b(?:format|way|pattern)\b",
    re.IGNORECASE | re.DOTALL,
)
_DRAFT_META_INSTRUCTION_PATTERN = re.compile(
    r"写作时(?:应|要|先)|这一段(?:应|要)|本段(?:应|要)|"
    r"\b(?:this\s+paragraph\s+should|you\s+should\s+write)\b",
    re.IGNORECASE,
)
_INVENTED_RUBRIC_SCORE_PATTERN = re.compile(
    r"(?:总分|满分)\s*[：:]?\s*\d+\s*分|"
    r"(?:每项|每条|各项).{0,16}\d+\s*分|"
    r"(?:达到|满足)?\s*\d+\s*分标准|"
    r"\b(?:total|worth)\s*(?:of\s*)?\d+\s*(?:points?|marks?)\b",
    re.IGNORECASE,
)
_LESSON_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:(?:#{2,6})\s+|\*\*\s*)"
    r"(?P<label>"
    r"第[一二三四五六七八九十百〇零两\d]+课|"
    r"lesson[\s_-]*\d+"
    r")"
    r"(?P<title>[^\r\n]*?)(?:\*\*)?\s*$"
)
def _numbered_section_spans(
    answer: str,
    requested: dict[int, str],
) -> dict[int, _NumberedSectionSpan]:
    """Locate the dominant top-level heading sequence without cutting at nested headings."""
    if not answer or not requested:
        return {}
    requested_order = {
        number: index for index, number in enumerate(requested)
    }
    grouped: dict[str, list[re.Match[str]]] = {}
    for match in _ANSWER_SECTION_HEADING_PATTERN.finditer(answer):
        hashes = match.group("hashes")
        style = f"h{len(hashes)}" if hashes else "bold"
        grouped.setdefault(style, []).append(match)
    if not grouped:
        return {}

    def ordered_matches(matches: list[re.Match[str]]) -> list[re.Match[str]]:
        selected: list[re.Match[str]] = []
        last_order = -1
        for match in matches:
            number = int(match.group("number"))
            if number not in requested_order:
                continue
            order = requested_order[number]
            if order <= last_order:
                continue
            selected.append(match)
            last_order = order
        return selected

    ordered_by_style = {
        style: ordered_matches(matches) for style, matches in grouped.items()
    }

    def style_score(item: tuple[str, list[re.Match[str]]]) -> tuple[int, int]:
        style, matches = item
        # 原因：第 7 节等正文内部常用 ### 1 / ### 2 组织子项，旧逻辑会把它们误作顶层。
        # 作用：优先选择覆盖最多请求编号的同级序列；数量相同时选择更浅的 Markdown 层级。
        level = int(style[1:]) if style.startswith("h") else 7
        return len(matches), -level

    selected_style, selected = max(ordered_by_style.items(), key=style_score)
    all_same_level_headings = grouped[selected_style]
    spans: dict[int, _NumberedSectionSpan] = {}
    for match in selected:
        number = int(match.group("number"))
        end = next(
            (
                candidate.start()
                for candidate in all_same_level_headings
                if candidate.start() > match.start()
            ),
            len(answer),
        )
        spans[number] = _NumberedSectionSpan(
            number=number,
            start=match.start(),
            end=end,
            body=answer[match.end() : end].strip(),
        )
    return spans


def _section_body_is_sufficient(body: str, title: str) -> bool:
    """Apply language-aware minimum substance checks without treating punctuation as content."""
    if not body or _SECTION_PLACEHOLDER_PATTERN.fullmatch(body):
        return False
    cjk_characters = len(re.findall(r"[\u3400-\u9fff]", body))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", body))
    combined_units = cjk_characters + latin_words
    list_items = len(_SECTION_LIST_ITEM_PATTERN.findall(body))
    normalized_title = title.casefold()

    draft_review_markers = (
        "后分析",
        "分析",
        "检查",
        "评估",
        "复盘",
        "点评",
        "改进",
        "post",
        "after",
        "analysis",
        "review",
        "critique",
        "check",
    )
    draft_delivery_markers = (
        "生成",
        "完整",
        "报告",
        "草稿",
        "撰写",
        "generate",
        "full",
        "complete",
        "report",
        "write",
    )
    is_full_draft = (
        "完整报告" in normalized_title
        or "完整草稿" in normalized_title
        or (
            "draft" in normalized_title
            and not any(marker in normalized_title for marker in draft_review_markers)
            and (
                normalized_title.strip() == "draft"
                or any(
                    marker in normalized_title
                    for marker in draft_delivery_markers
                )
            )
        )
    )
    if is_full_draft:
        # 中文以单字承载的语义密度高于空格分词英文；分别设门槛避免短中文报告被误拒。
        return cjk_characters >= 180 or latin_words >= 300 or combined_units >= 240
    if any(
        marker in normalized_title
        for marker in ("outline", "大纲", "框架", "逐段")
    ):
        return (
            cjk_characters >= 36
            or latin_words >= 60
            or combined_units >= 48
            or list_items >= 4
        )
    # 原因：简洁中文分析可能用十余字完成一个小节，通用 token 下限会把真实内容判成缺失。
    # 作用：普通节只拒绝空白、占位符和极短敷衍文本；结构型清单可由两个实质条目通过。
    return (
        cjk_characters >= 12
        or latin_words >= 18
        or combined_units >= 14
        or list_items >= 2
    )


def _missing_requested_sections(
    answer: str,
    requested: dict[int, str],
) -> dict[int, str]:
    """Return absent or placeholder-sized report sections from an explicit contract."""
    if not requested:
        return {}
    spans = _numbered_section_spans(answer, requested)
    missing: dict[int, str] = {}
    for number, title in requested.items():
        span = spans.get(number)
        if span is None or not _section_body_is_sufficient(span.body, title):
            missing[number] = title
    return missing


def _collection_grounding_evidence(steps: list[dict[str, Any]]) -> str:
    """Keep only evidence returned by an actual collection-summary Tool call."""
    evidence: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "document_collection_summary" not in _extract_agent_tool_calls([step]):
            continue
        observation = step.get("observations")
        if (
            isinstance(observation, str)
            and "QWOPUS_SOURCE_COVERAGE=" in observation
        ):
            evidence.append(observation)
    return "\n\n".join(evidence)


def _lesson_subsections(draft_body: str) -> tuple[_LessonSubsection, ...]:
    """Parse only anchored Markdown lesson headings, never incidental prose mentions."""
    matches = list(_LESSON_HEADING_PATTERN.finditer(draft_body))
    subsections: list[_LessonSubsection] = []
    for index, match in enumerate(matches):
        number = _lesson_number_from_label(match.group("label"))
        if number is None:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(draft_body)
        subsections.append(
            _LessonSubsection(
                number=number,
                start=match.start(),
                end=end,
                heading=match.group(0).strip(),
                body=draft_body[match.end() : end].strip(),
            )
        )
    return tuple(subsections)


def _lesson_candidate_issues(
    subsection: _LessonSubsection,
    spec: _LessonGroundingSpec,
) -> tuple[str, ...]:
    """Validate one generated lesson only against that lesson's own source facts."""
    issues: list[str] = []
    combined = f"{subsection.heading}\n{subsection.body}"
    cjk_characters = len(re.findall(r"[\u3400-\u9fff]", subsection.body))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", subsection.body))
    if cjk_characters + latin_words < 55:
        issues.append("subsection is underdeveloped")
    if _DRAFT_OMISSION_PATTERN.search(subsection.body):
        issues.append("subsection contains an omission placeholder")
    if _DRAFT_META_INSTRUCTION_PATTERN.search(subsection.body):
        issues.append("subsection contains writing instructions instead of Draft prose")

    normalized_candidate = _normalized_fact_text(combined)
    missing_topics = [
        topic
        for topic in spec.topic_lines
        if len(_normalized_fact_text(topic)) >= 3
        and _normalized_fact_text(topic) not in normalized_candidate
    ]
    if missing_topics:
        issues.append("missing exact source topic")

    references = tuple(
        match.group(0).strip()
        for match in _SCRIPTURE_REFERENCE_PATTERN.finditer(combined)
    )
    if spec.allowed_scriptures:
        if not references:
            issues.append("missing source scripture")
        unsupported = [
            reference
            for reference in references
            if not _scripture_reference_is_supported(
                _scripture_reference_key(reference),
                spec.allowed_scriptures,
            )
        ]
        if unsupported:
            issues.append(
                "scripture belongs outside this lesson: " + ", ".join(unsupported)
            )
    elif references:
        issues.append("scripture was added without a source fact")
    return tuple(issues)


def _lesson_slot_manifest(specs: tuple[_LessonGroundingSpec, ...]) -> str:
    """Render the exact ordered lesson contract used by the model repair prompt."""
    if not specs:
        return ""
    lines = ["MANDATORY_LESSON_SLOTS (write each exactly once, in this order):"]
    for spec in specs:
        facts = [
            f"source={spec.file_name}",
            *spec.topic_lines,
            *spec.scripture_lines,
        ]
        lines.append(f"- {spec.canonical_label} | " + " | ".join(facts))
    return "\n".join(lines)


def _report_quality_issues(
    *,
    answer: str,
    requested: dict[int, str],
    file_names: list[str],
    user_question: str,
    collection_evidence: str,
) -> dict[int, list[str]]:
    """Return source-grounding and exhaustive-deliverable defects by report section."""
    if not requested or not collection_evidence:
        return {}
    spans = _numbered_section_spans(answer, requested)
    if not spans:
        return {}
    issues: dict[int, list[str]] = {}

    def add(number: int, message: str) -> None:
        issues.setdefault(number, []).append(message)

    if _ALL_SOURCE_REQUEST_PATTERN.search(user_question):
        understanding_number = next(
            (
                number
                for number, title in requested.items()
                if any(
                    marker in title.casefold()
                    for marker in (
                        "文档理解",
                        "文件理解",
                        "材料理解",
                        "document understanding",
                        "source understanding",
                    )
                )
            ),
            None,
        )
        understanding = (
            spans.get(understanding_number)
            if understanding_number is not None
            else None
        )
        if understanding_number is not None and understanding is not None:
            normalized_body = understanding.body.casefold()
            missing_labels = [
                _source_answer_label(file_name)
                for file_name in file_names
                if _source_answer_label(file_name).casefold() not in normalized_body
            ]
            if missing_labels:
                add(
                    understanding_number,
                    "summarize every selected source explicitly; missing source labels: "
                    + ", ".join(missing_labels),
                )

    lesson_specs = _lesson_grounding_specs(file_names, collection_evidence)
    for number, title in requested.items():
        if not _title_requires_full_draft(title):
            continue
        span = spans.get(number)
        if span is None:
            continue
        subsections = _lesson_subsections(span.body)
        expected_numbers = [spec.number for spec in lesson_specs]
        observed_numbers = [subsection.number for subsection in subsections]
        subsection_counts = {
            lesson_number: observed_numbers.count(lesson_number)
            for lesson_number in set(observed_numbers)
        }
        missing_lessons = [
            spec.canonical_label
            for spec in lesson_specs
            if subsection_counts.get(spec.number, 0) == 0
        ]
        if missing_lessons:
            add(
                number,
                "the complete Draft must contain a substantive subsection for every lesson; "
                "missing lesson labels: "
                + ", ".join(missing_lessons),
            )
        duplicate_lessons = [
            spec.canonical_label
            for spec in lesson_specs
            if subsection_counts.get(spec.number, 0) > 1
        ]
        if duplicate_lessons:
            add(
                number,
                "each lesson must appear exactly once; duplicate lesson headings: "
                + ", ".join(duplicate_lessons),
            )
        unexpected_lessons = sorted(
            set(observed_numbers).difference(expected_numbers)
        )
        if unexpected_lessons:
            add(
                number,
                "remove lesson headings that are not backed by selected sources: "
                + ", ".join(str(value) for value in unexpected_lessons),
            )
        observed_expected = [
            lesson_number
            for lesson_number in observed_numbers
            if lesson_number in expected_numbers
        ]
        if (
            len(observed_expected) == len(expected_numbers)
            and observed_expected != expected_numbers
        ):
            add(
                number,
                "lesson subsections must follow source lesson order; "
                f"expected={expected_numbers}, observed={observed_expected}",
            )

        subsections_by_number = {
            spec.number: [
                subsection
                for subsection in subsections
                if subsection.number == spec.number
            ]
            for spec in lesson_specs
        }
        for spec in lesson_specs:
            candidates = subsections_by_number[spec.number]
            if len(candidates) != 1:
                continue
            candidate_issues = _lesson_candidate_issues(candidates[0], spec)
            if candidate_issues:
                add(
                    number,
                    f"{spec.canonical_label} is not a grounded substantive subsection: "
                    + ", ".join(candidate_issues),
                )
        if _DRAFT_OMISSION_PATTERN.search(span.body):
            add(
                number,
                "remove example-only/remaining-sections placeholders and write the full Draft",
            )
        repeated_sentences: dict[str, int] = {}
        for subsection in subsections:
            seen_in_subsection = {
                sentence.strip()
                for sentence in re.findall(
                    r"[^。！？!?]+[。！？!?]",
                    subsection.body,
                )
                if len(sentence.strip()) >= 30
            }
            for sentence in seen_in_subsection:
                repeated_sentences[sentence] = (
                    repeated_sentences.get(sentence, 0) + 1
                )
        boilerplate = [
            sentence
            for sentence, count in repeated_sentences.items()
            if count >= 3
        ]
        if boilerplate:
            add(
                number,
                "replace repeated cross-lesson boilerplate with source-specific analysis",
            )
        cjk_characters = len(re.findall(r"[\u3400-\u9fff]", span.body))
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", span.body))
        minimum_units = max(160, len(lesson_specs) * 70)
        if cjk_characters + latin_words < minimum_units:
            add(
                number,
                f"the complete Draft is underdeveloped; provide at least {minimum_units} "
                "substantive Chinese characters or words across the required lessons",
            )

    allowed_scriptures = {
        key
        for line in re.findall(
            r"(?m)^-\s*scripture_line:\s*(.+)$",
            collection_evidence,
        )
        if (key := _scripture_reference_key(line)) is not None
    }
    grounding_markers = (
        "文档理解",
        "文件理解",
        "材料理解",
        "outline",
        "大纲",
        "框架",
        "逐段",
        "完整报告",
        "完整草稿",
        "生成",
    )
    if allowed_scriptures:
        for number, span in spans.items():
            title = requested[number]
            # 完整 Draft 已按课次使用各自来源的许可集检查，不能再退回全集 allowlist。
            if _title_requires_full_draft(title):
                continue
            if not (
                any(marker in title.casefold() for marker in grounding_markers)
            ):
                continue
            unsupported = sorted(
                {
                    match.group(0).strip()
                    for match in _SCRIPTURE_REFERENCE_PATTERN.finditer(span.body)
                    if not _scripture_reference_is_supported(
                        _scripture_reference_key(match.group(0)),
                        allowed_scriptures,
                    )
                }
            )
            if unsupported:
                add(
                    number,
                    "remove or correct scripture references absent from SOURCE_FACTS: "
                    + ", ".join(unsupported),
                )

    if "QWOPUS_EXPLICIT_RUBRIC_FOUND=false" in collection_evidence:
        for number, span in spans.items():
            invented_scores = sorted(
                {
                    match.group(0).strip()
                    for match in _INVENTED_RUBRIC_SCORE_PATTERN.finditer(span.body)
                }
            )
            if invented_scores:
                add(
                    number,
                    "the sources contain no explicit rubric; say so and remove invented "
                    "scores/weights: "
                    + ", ".join(invented_scores),
                )
    return issues


def _best_grounded_lesson_candidate(
    *,
    refinement_body: str,
    original_body: str,
    spec: _LessonGroundingSpec,
) -> _LessonSubsection | None:
    """Prefer a valid repaired candidate, then a valid original candidate."""
    for draft_body in (refinement_body, original_body):
        candidates = [
            subsection
            for subsection in _lesson_subsections(draft_body)
            if subsection.number == spec.number
            and not _lesson_candidate_issues(subsection, spec)
        ]
        if candidates:
            return max(
                candidates,
                key=lambda subsection: len(subsection.body),
            )
    return None


def _merge_full_draft_lesson_slots(
    *,
    original_body: str,
    refinement_body: str,
    specs: tuple[_LessonGroundingSpec, ...],
) -> str:
    """Rebuild a Draft from one canonical, ordered, source-grounded slot per lesson."""
    rendered: list[str] = []
    for spec in specs:
        candidate = _best_grounded_lesson_candidate(
            refinement_body=refinement_body,
            original_body=original_body,
            spec=spec,
        )
        if candidate is None:
            rendered.append(_render_grounded_lesson_fallback(spec))
            continue
        rendered.append(
            f"{_canonical_lesson_heading(spec)}\n\n{candidate.body.strip()}"
        )
    return "\n\n".join(rendered)


def _is_model_generation_failure_output(answer: str) -> bool:
    """Recognize smolagents' final-generation error placeholder."""
    normalized = " ".join(answer.casefold().split())
    return any(
        marker in normalized
        for marker in (
            "error in generating final llm output:",
            "error in generating final model output:",
        )
    )


def _apply_grounded_report_fallbacks(
    *,
    answer: str,
    refinement: str,
    requested: dict[int, str],
    target_sections: dict[int, str],
    quality_issues: dict[int, list[str]],
    file_names: list[str],
    collection_evidence: str,
    lesson_specs: tuple[_LessonGroundingSpec, ...],
) -> str:
    """Deterministically repair source inventory and lesson slots after model formatting drift."""
    spans = _numbered_section_spans(answer, requested)
    edits: list[tuple[int, int, str]] = []
    for number, title in target_sections.items():
        span = spans.get(number)
        if span is None or number not in quality_issues:
            continue
        body = span.body
        if _title_requires_full_draft(title) and lesson_specs:
            body = _merge_full_draft_lesson_slots(
                original_body=span.body,
                refinement_body=refinement,
                specs=lesson_specs,
            )
        elif _title_is_source_understanding(title):
            inventory = _render_grounded_source_inventory(
                file_names=file_names,
                collection_evidence=collection_evidence,
                existing_body=body,
            )
            if inventory:
                body = f"{body.rstrip()}\n\n{inventory}"
        else:
            continue

        original_segment = answer[span.start : span.end]
        heading = next(
            line.strip()
            for line in original_segment.splitlines()
            if line.strip()
        )
        trailing_whitespace = re.search(r"\s*$", original_segment)
        separator = (
            trailing_whitespace.group(0)
            if trailing_whitespace and trailing_whitespace.group(0)
            else ("\n\n" if span.end < len(answer) else "")
        )
        edits.append(
            (
                span.start,
                span.end,
                f"{heading}\n\n{body.strip()}{separator}",
            )
        )

    repaired = answer
    for start, end, replacement in sorted(edits, reverse=True):
        repaired = f"{repaired[:start]}{replacement}{repaired[end:]}"
    return repaired.strip()


def _merge_numbered_section_refinement(
    original: str,
    refinement: str,
    requested: dict[int, str],
    target_sections: dict[int, str],
    lesson_specs: tuple[_LessonGroundingSpec, ...] = (),
) -> str:
    """Replace only requested deficient sections while retaining accepted answer sections."""
    original_spans = _numbered_section_spans(original, requested)
    refinement_spans = _numbered_section_spans(refinement, target_sections)
    if not refinement_spans:
        return original

    # A response with no contract headings has no reliable boundary for a safe merge.
    if not original_spans:
        return "\n\n".join(
            refinement[refinement_spans[number].start : refinement_spans[number].end].strip()
            for number in requested
            if number in refinement_spans
        )

    request_order = {number: index for index, number in enumerate(requested)}
    edits: list[tuple[int, int, str]] = []
    insertions: dict[int, list[str]] = {}
    for number in requested:
        if number not in target_sections or number not in refinement_spans:
            continue
        refinement_span = refinement_spans[number]
        refinement_segment = refinement[
            refinement_span.start : refinement_span.end
        ].strip()
        original_span = original_spans.get(number)
        if _title_requires_full_draft(requested[number]) and lesson_specs:
            original_body = original_span.body if original_span is not None else ""
            lesson_body = _merge_full_draft_lesson_slots(
                original_body=original_body,
                refinement_body=refinement_span.body,
                specs=lesson_specs,
            )
            heading_source = (
                original[original_span.start : original_span.end].strip()
                if original_span is not None
                else refinement_segment
            )
            heading = next(
                line.strip()
                for line in heading_source.splitlines()
                if line.strip()
            )
            replacement = f"{heading}\n\n{lesson_body}"
        else:
            replacement = refinement_segment
        if original_span is not None:
            original_segment = original[original_span.start : original_span.end]
            trailing_whitespace = re.search(r"\s*$", original_segment)
            separator = (
                trailing_whitespace.group(0)
                if trailing_whitespace and trailing_whitespace.group(0)
                else ("\n\n" if original_span.end < len(original) else "")
            )
            replacement = f"{replacement}{separator}"
            edits.append((original_span.start, original_span.end, replacement))
            continue
        later_starts = [
            span.start
            for other_number, span in original_spans.items()
            if request_order[other_number] > request_order[number]
        ]
        insert_at = min(later_starts, default=len(original))
        insertions.setdefault(insert_at, []).append(replacement)

    for insert_at, sections in insertions.items():
        insertion = "\n\n".join(sections).strip()
        insertion = (
            f"\n\n{insertion}"
            if insert_at == len(original)
            else f"{insertion}\n\n"
        )
        edits.append((insert_at, insert_at, insertion))

    merged = original
    # 同一位置先替换既有不足节，再插入它前面的缺失节，避免偏移或次序反转。
    for start, end, replacement in sorted(
        edits,
        key=lambda edit: (edit[0], edit[1]),
        reverse=True,
    ):
        merged = f"{merged[:start]}{replacement}{merged[end:]}"
    return merged.strip()
