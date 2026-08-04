"""Fact parsing and evidence normalization for grounded report composition.

This module intentionally has no dependency on report rendering or smolagents,
so source parsing and evidence validation remain independently testable.

The parsing helpers here are generic: a recipe supplies the item label
extractor, reference validator, and spec builder so that domain-specific
concepts (for example Bible lessons in
:mod:`qwopus_agent.reports.bible_recipe`) never leak into this module.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from qwopus_agent.utils.token_budget import truncate_to_tokens


@dataclass(frozen=True)
class SourceGroundingSpec:
    """Machine-checkable facts for one source slot in a collection report.

    ``number`` is the deterministic source order (generic recipe) or a
    domain item number (Bible recipe uses the lesson number).  ``passage_lines``
    carries quoted reference lines (scripture in the Bible recipe);
    ``allowed_references`` restricts which references that source may cite.
    """

    number: int
    canonical_label: str
    file_name: str
    document_heading: str | None
    topic_lines: tuple[str, ...]
    passage_lines: tuple[str, ...]
    allowed_references: frozenset[tuple[str, tuple[int, ...]]]
    evidence_excerpt: str
    application_excerpt: str


_ALL_SOURCE_REQUEST_PATTERN = re.compile(
    r"(?:所有|全部|逐一).{0,24}(?:文件|文档|资料|材料)|"
    r"(?:文件|文档|资料|材料).{0,24}(?:所有|全部|逐一)|"
    r"\b(?:all|every)\b.{0,32}\b(?:files?|documents?|sources?|materials?)\b|"
    r"\beach\b(?:\s+of\s+the)?(?:\s+(?:selected|uploaded|provided)){0,2}"
    r"\s+(?:file|document|source|material)s?\b",
    re.IGNORECASE | re.DOTALL,
)


def _requested_numbered_sections(user_question: str) -> dict[int, str]:
    """Extract an explicit Markdown deliverable contract from the user's request."""
    sections: dict[int, str] = {}
    pattern = re.compile(
        r"(?m)^\s*#{1,6}\s*(?P<number>\d{1,2})\s*[.、．:：)]\s*(?P<title>[^\r\n]+)"
    )
    for match in pattern.finditer(user_question):
        number = int(match.group("number"))
        title = match.group("title").strip().strip("*").strip()
        if title:
            sections.setdefault(number, title)
    return sections


def _title_requires_full_draft(title: str) -> bool:
    """Return true only for the deliverable draft, never its later review."""
    normalized_title = title.casefold()
    review_markers = (
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
    delivery_markers = (
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
    return (
        "完整报告" in normalized_title
        or "完整草稿" in normalized_title
        or (
            "draft" in normalized_title
            and not any(marker in normalized_title for marker in review_markers)
            and (
                normalized_title.strip() == "draft"
                or any(marker in normalized_title for marker in delivery_markers)
            )
        )
    )


def _collection_manifest_sources(collection_evidence: str) -> tuple[str, ...]:
    """Parse one exact, duplicate-free source manifest from collection evidence."""
    matches = re.findall(
        r"(?m)^QWOPUS_SOURCE_COVERAGE=(\[[^\r\n]*\])\s*$",
        collection_evidence,
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Collection evidence must contain exactly one source coverage manifest."
        )
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Collection source coverage manifest is invalid JSON.") from exc
    if (
        not isinstance(payload, list)
        or any(not isinstance(source, str) or not source.strip() for source in payload)
    ):
        raise RuntimeError("Collection source coverage manifest must be a list of names.")
    sources = tuple(source.strip() for source in payload)
    if len(sources) != len(set(sources)):
        raise RuntimeError("Collection source coverage manifest contains duplicates.")
    return sources


def _source_answer_label(file_name: str) -> str:
    """Return the stable human-visible source label expected in an all-files summary."""
    stem = Path(file_name).stem.strip()
    return re.sub(r"\(\d+\)$", "", stem).strip()


def _collection_source_blocks(collection_evidence: str) -> dict[str, str]:
    """Split the collection Tool result into exact, source-isolated file blocks."""
    matches = list(
        re.finditer(
            r"(?m)^# File:\s*(?P<name>[^\r\n]+)\s*$",
            collection_evidence,
        )
    )
    return {
        match.group("name").strip(): collection_evidence[
            match.start() : (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(collection_evidence)
            )
        ].strip()
        for index, match in enumerate(matches)
    }


def _source_fact_values(block: str, label: str) -> tuple[str, ...]:
    """Read exact SOURCE_FACTS values without interpreting prose elsewhere in a block."""
    return tuple(
        match.group("value").strip()
        for match in re.finditer(
            rf"(?m)^-\s*{re.escape(label)}:\s*(?P<value>[^\r\n]+)$",
            block,
        )
        if match.group("value").strip()
    )


def _source_tagged_excerpt(block: str, label: str) -> str:
    """Return one exact, source-isolated evidence field from a collection block."""
    match = re.search(
        rf"(?s){re.escape(label)}[^\r\n]*:\s*\n"
        r"(?P<body>.*?)(?=\n(?:CORE_INTERPRETATION_EVIDENCE|"
        r"QUERY_RELEVANT_EVIDENCE|"
        r"SECONDARY_INTERPRETATION_EVIDENCE|"
        r"APPLICATION_EVIDENCE|BALANCED_OVERVIEW)\b|\Z)",
        block,
    )
    if match is None:
        return ""
    return match.group("body").strip()


def _source_evidence_excerpt(block: str) -> str:
    """Return only the source's cited interpretation excerpt."""
    return "\n\n".join(
        excerpt
        for excerpt in (
            _source_tagged_excerpt(
                block,
                "CORE_INTERPRETATION_EVIDENCE",
            ),
            _source_tagged_excerpt(block, "QUERY_RELEVANT_EVIDENCE"),
            _source_tagged_excerpt(
                block,
                "SECONDARY_INTERPRETATION_EVIDENCE",
            ),
        )
        if excerpt
    )


def _source_application_excerpt(block: str) -> str:
    """Return only the source's cited application excerpt."""
    return _source_tagged_excerpt(block, "APPLICATION_EVIDENCE")


def _normalized_fact_text(value: str) -> str:
    """Normalize harmless typography while preserving fact-bearing letters and digits."""
    payload = re.sub(
        r"^(?:题目|題目|主题|主題|title)\s*[：:]\s*",
        "",
        unicodedata.normalize("NFKC", value).strip(),
        flags=re.IGNORECASE,
    )
    return "".join(
        character.casefold()
        for character in payload
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )


def _topic_payload(value: str) -> str:
    """Remove only a SOURCE_FACTS topic label for canonical Markdown rendering."""
    return re.sub(
        r"^(?:题目|題目|主题|主題|title)\s*[：:]\s*",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).strip()


def _canonical_source_heading(spec: SourceGroundingSpec) -> str:
    topic = " ".join(
        payload
        for value in spec.topic_lines
        if (payload := _topic_payload(value))
    )
    suffix = f"：{topic}" if topic else ""
    return f"### {spec.canonical_label}{suffix}"


def _grounded_evidence_claim(
    spec: SourceGroundingSpec,
    *,
    boost_terms: tuple[str, ...] = ("区别", "关系", "核心"),
) -> tuple[str, str]:
    """Select one readable source sentence and describe its visible reasoning form."""
    evidence = re.sub(r"\s+", " ", spec.evidence_excerpt).strip()
    sentences = [
        sentence.strip(" \t\r\n“”\"'")
        for sentence in re.findall(r"[^。！？!?]+[。！？!?]", evidence)
        if (
            12 <= len(sentence.strip()) <= 320
            and re.match(r"^[，。；：、”’）)\]]", sentence.strip()) is None
            and not sentence.strip().startswith(("点，而", "决定”，而", "决定\",而"))
        )
    ]
    pairs = [
        f"{first} {second}"
        for first, second in zip(sentences, sentences[1:], strict=False)
        if len(first) + len(second) <= 320
    ]
    candidates = [*sentences, *pairs]
    topic_cjk = "".join(
        re.findall(
            r"[\u3400-\u9fff]",
            " ".join(spec.topic_lines),
        )
    )
    ignored_grams = {
        "题目",
        "系列",
        "什么",
        "一个",
        "不是",
        "怎样",
        "我们",
        "自己",
        "生命",
        "的人",
        "的事",
    }
    topic_grams = {
        topic_cjk[index : index + width]
        for width in (2, 3)
        for index in range(max(0, len(topic_cjk) - width + 1))
        if topic_cjk[index : index + width] not in ignored_grams
    }

    def evidence_score(candidate: str) -> tuple[int, int]:
        normalized = _normalized_fact_text(candidate)
        topic_score = sum(
            2 if len(gram) == 3 else 1
            for gram in topic_grams
            if gram in normalized
        )
        reasoning_score = 0
        if "不是" in candidate and any(
            marker in candidate for marker in ("而是", "但", "相反")
        ):
            reasoning_score += 7
        if any(marker in candidate for marker in ("因为", "所以", "因此")):
            reasoning_score += 5
        reasoning_score += sum(
            1
            for marker in boost_terms
            if marker in candidate
        )
        return topic_score + reasoning_score, min(len(candidate), 180)

    preferred = (
        max(candidates, key=evidence_score)
        if candidates
        else truncate_to_tokens(evidence, 90).strip()
    )
    if "不是" in preferred and "而是" in preferred:
        reasoning = "使用“不是……而是……”划定概念边界"
    elif any(marker in preferred for marker in ("因为", "所以", "因此")):
        reasoning = "以明确的因果关系推进论证"
    elif any(marker in preferred for marker in ("为什么", "？", "?")):
        reasoning = "先提出问题，再要求读者检查原有假设"
    else:
        reasoning = "从一个具体判断或处境推进主题"
    return preferred, reasoning


def _grounded_application_claim(spec: SourceGroundingSpec) -> str:
    """Select one complete, source-authored application sentence or question."""
    application = re.sub(r"\s+", " ", spec.application_excerpt).strip()
    candidates = [
        sentence.strip(" \t\r\n“”\"'")
        for sentence in re.findall(r"[^。！？!?]+[。！？!?]", application)
        if 10 <= len(sentence.strip()) <= 360
    ]
    return next(
        (
            sentence
            for sentence in candidates
            if any(
                marker in sentence
                for marker in ("你", "我们", "分享", "如何", "什么", "哪", "？")
            )
        ),
        candidates[0] if candidates else truncate_to_tokens(application, 100).strip(),
    )


def _render_generic_source_fallback(spec: SourceGroundingSpec) -> str:
    """Render a safe source card when the model omitted or corrupted one source."""
    if not (
        spec.document_heading
        or spec.topic_lines
        or spec.passage_lines
        or spec.evidence_excerpt
    ):
        raise RuntimeError(
            f"No grounded source facts are available for {spec.canonical_label} "
            f"from {spec.file_name}."
        )
    topic = "；".join(
        payload
        for value in spec.topic_lines
        if (payload := _topic_payload(value))
    )
    references = "；".join(spec.passage_lines)
    evidence, reasoning = _grounded_evidence_claim(spec)
    application = _grounded_application_claim(spec)
    lines = [
        _canonical_source_heading(spec),
        (
            f"**引文与主题：** {references}。该来源围绕“{topic}”展开。"
            if references and topic
            else (
                f"**引文与主题：** {references}。"
                if references
                else f"**引文与主题：** 材料未单列引用；该来源围绕“{topic}”展开。"
            )
        ),
        (
            f"**核心论点：** {spec.canonical_label}的材料把中心界定为“{topic}”。"
            f"{spec.canonical_label}的论点来自本来源 SOURCE_FACTS，不是从相邻来源"
            "推演出来的概括。"
            if topic
            else "**核心论点：** 该来源按对应源文件的标题和正文证据展开。"
        ),
    ]
    if evidence:
        lines.append(
            f"**材料论证：** 材料写道：“{evidence}”"
            f"{spec.canonical_label}的证据{reasoning}。{spec.canonical_label}必须把"
            f"本来源主题从标题推进到可辨认的判断；{spec.canonical_label}对引文的理解"
            "须服从原句显示的区别、原因或问题。"
        )
    if application:
        lines.append(
            f"**具体展开：** 材料把应用落在这个具体处境：“{application}”"
            f"{spec.canonical_label}由此不只停在观念认同，而要在真实关系、决定或"
            f"习惯中检验“{topic or spec.canonical_label}”所涉及的动机和回应。"
        )
    else:
        lines.append(
            "**具体展开：** 当前证据包没有抽取出独立的展开段，因此这里只保留"
            "引文与论点，不虚构材料未提供的案例。"
        )
    lines.extend(
        [
            (
                f"**本来源结论：** {topic or spec.canonical_label}必须由本来源的引用、"
                f"上述材料论证和具体处境共同界定。{spec.canonical_label}可以与相邻来源"
                "形成推进，却不能被"
                f"压缩成适用于所有来源的同一句“{topic or spec.canonical_label}”口号。"
            ),
            f"**来源：** `{spec.file_name}`",
        ]
    )
    return "\n\n".join(lines)


def _source_topic(spec: SourceGroundingSpec) -> str:
    return "；".join(
        payload
        for value in spec.topic_lines
        if (payload := _topic_payload(value))
    ) or spec.canonical_label


def _source_reference(spec: SourceGroundingSpec) -> str:
    return "；".join(spec.passage_lines) or "材料未单列引用"


def _source_evidence(spec: SourceGroundingSpec, *, max_tokens: int = 55) -> str:
    return re.sub(
        r"\s+",
        " ",
        truncate_to_tokens(spec.evidence_excerpt, max_tokens),
    ).strip()


def build_generic_grounding_specs(
    file_names: list[str],
    collection_evidence: str,
) -> tuple[SourceGroundingSpec, ...]:
    """Build one spec per parser file in deterministic file order.

    This is the generic recipe's spec builder: every file is one independent
    source slot, and any domain item that the recipe does not recognize is
    simply treated as another source in the selected order.
    """
    source_blocks = _collection_source_blocks(collection_evidence)
    specs: list[SourceGroundingSpec] = []
    for index, file_name in enumerate(file_names):
        block = source_blocks.get(file_name, "")
        # 原因：没有 collection 证据的文件没有可渲染的来源槽位，也不能做 fallback。
        # 作用：只对实际出现在证据中的文件构建 spec，避免对空块执行渲染校验。
        if not block:
            continue
        passage_lines = _source_fact_values(block, "quote_line")
        specs.append(
            SourceGroundingSpec(
                number=index,
                canonical_label=_source_answer_label(file_name),
                file_name=file_name,
                document_heading=next(
                    iter(_source_fact_values(block, "document_heading")),
                    None,
                ),
                topic_lines=(
                    *_source_fact_values(block, "topic_line"),
                    *_source_fact_values(block, "topic_continuation"),
                ),
                passage_lines=passage_lines,
                allowed_references=frozenset(),
                evidence_excerpt=_source_evidence_excerpt(block),
                application_excerpt=_source_application_excerpt(block),
            )
        )
    return tuple(specs)
