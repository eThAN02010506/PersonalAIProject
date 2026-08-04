"""Deterministic source-grounded report composition and rendering.

This module owns the generic rendering policy and the default recipe that the
shared composer uses.  Domain recipes (for example Bible-study lessons) can be
built from :data:`DEFAULT_RECIPE` and override only the fields they need.
"""

from __future__ import annotations

import re

from qwopus_agent.reports import grounded_facts
from qwopus_agent.reports.recipe import (
    ComposerThresholds,
    ReportRecipe,
    SectionKind,
    SourceFactLabels,
    default_recipe,
)
from qwopus_agent.utils.token_budget import truncate_to_tokens

_ALL_SOURCE_REQUEST_PATTERN = grounded_facts._ALL_SOURCE_REQUEST_PATTERN
_SourceGroundingSpec = grounded_facts.SourceGroundingSpec
_build_generic_grounding_specs = grounded_facts.build_generic_grounding_specs
_canonical_source_heading = grounded_facts._canonical_source_heading
_grounded_evidence_claim = grounded_facts._grounded_evidence_claim
_grounded_application_claim = grounded_facts._grounded_application_claim
_requested_numbered_sections = grounded_facts._requested_numbered_sections
_source_answer_label = grounded_facts._source_answer_label
_source_evidence = grounded_facts._source_evidence
_source_fact_values = grounded_facts._source_fact_values
_source_reference = grounded_facts._source_reference
_source_topic = grounded_facts._source_topic
_render_generic_source_fallback = grounded_facts._render_generic_source_fallback
_title_requires_full_draft = grounded_facts._title_requires_full_draft
_collection_manifest_sources = grounded_facts._collection_manifest_sources
_collection_source_blocks = grounded_facts._collection_source_blocks


def _title_is_source_understanding(title: str) -> bool:
    normalized = title.casefold()
    return any(
        marker in normalized
        for marker in (
            "文档理解",
            "文件理解",
            "材料理解",
            "document understanding",
            "source understanding",
        )
    )


def _render_grounded_full_draft(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    """Render an evidence-specific complete Draft with an opening and synthesis."""
    if not specs:
        raise RuntimeError("A complete Draft requires at least one grounded source.")
    topic_sequence = " → ".join(
        grounded_facts._source_topic(spec) for spec in specs
    )
    source_body = "\n\n".join(recipe.render_fallback(spec) for spec in specs)
    return (
        "### 引言\n\n"
        f"本报告按 {specs[0].canonical_label} 至 {specs[-1].canonical_label} 的顺序"
        "阅读材料。中心论点不是把所有来源归并成一个抽象口号，而是追踪每份材料如何用"
        "自己的主题、引文和正文证据推动理解与具体实践。来源标题呈现的推进为："
        f"{topic_sequence}。以下每一节都保留对应文件的证据边界。\n\n"
        f"{source_body}\n\n"
        "### 综合结论\n\n"
        "这些材料共同显示，成熟的写作必须同时完成三件事：准确限定每份材料的引用，解释"
        "材料中的具体论证，并把解释落实到可观察的关系与行动。来源可以彼此衔接，"
        "但不能彼此替代；只有逐份证据链完整后，跨来源综合才具有可信度。"
    )


def _render_grounded_source_inventory(
    *,
    file_names: list[str],
    collection_evidence: str,
    existing_body: str,
    recipe: ReportRecipe,
    title: str = "",
) -> str:
    """Render only source cards still absent from the document-understanding section."""
    blocks = grounded_facts._collection_source_blocks(collection_evidence)
    missing_files = [
        file_name
        for file_name in file_names
        if _source_answer_label(file_name).casefold() not in existing_body.casefold()
    ]
    if not missing_files:
        return ""
    cards = ["### 逐文件材料清单（来源事实补全）"]
    for file_name in missing_files:
        block = blocks.get(file_name, "")
        facts = [
            *grounded_facts._source_fact_values(block, "document_heading"),
            *grounded_facts._source_fact_values(block, "topic_line"),
            *grounded_facts._source_fact_values(block, "topic_continuation"),
            *grounded_facts._source_fact_values(block, "quote_line"),
            *grounded_facts._source_fact_values(block, "opening_line"),
        ]
        evidence = re.sub(
            r"\s+",
            " ",
            truncate_to_tokens(
                grounded_facts._source_evidence_excerpt(block),
                90,
            ),
        ).strip()
        fact_text = "；".join(dict.fromkeys(facts))
        details = [
            f"来源事实：{fact_text}" if fact_text else "来源事实：未提取到结构化标题。",
            f"正文证据：{evidence}" if evidence else "",
        ]
        cards.append(
            f"- **{_source_answer_label(file_name)}**（`{file_name}`）："
            + " ".join(detail for detail in details if detail)
        )
    return "\n\n".join(cards)


def _render_grounded_understanding(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    rubric = (
        "检测到显式 rubric，但当前证据包没有安全复述其细则；提交前必须回到原文逐条核对。"
        if "QWOPUS_EXPLICIT_RUBRIC_FOUND=true" in collection_evidence
        else "材料没有提供显式评分 rubric；以下高质量标准属于写作自检，不冒充原始评分规则。"
    )
    return "\n\n".join(
        [
            (
                f"本次任务共有 {len(file_names)} 个来源。真正要解决的不是把来源压成一个主题，"
                "而是先保持每份来源的边界，再把方法论、逐份解读、具体应用和完整"
                "写作成品连成一条可核查的论证链。"
            ),
            _render_grounded_source_inventory(
                file_names=file_names,
                collection_evidence=collection_evidence,
                existing_body="",
                recipe=recipe,
            ),
            (
                f"**Rubric 判断：** {rubric}\n\n"
                "**必须包含：** 每份来源的标题、对应引用、该文件的正文证据、为什么该主题重要、"
                "具体应用与来源名；方法材料用于控制上下文、思想流、结构和解释边界。"
            ),
        ]
    )


def _render_grounded_strategy(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    first = specs[0].canonical_label if specs else "第一份材料"
    last = specs[-1].canonical_label if specs else "最后一份材料"
    return (
        "报告采用“方法说明 → 逐份观察 → 主题推进 → 具体应用 → 综合结论”的结构。\n\n"
        "- **方法说明**不是泛泛介绍，而是交代如何使用上下文、思想流和结构来约束解释；"
        "读者由此知道后文为什么不能把相邻来源混写。\n"
        f"- **逐份观察**从{first}到{last}依次展开，每份先固定主题与引用，再解释材料"
        "给出的具体问题。读者应能沿文件名反查证据。\n"
        "- **主题推进**说明前一份材料如何提出问题、后一份如何继续或转向；这种衔接建立在"
        "主题和引文变化上，不靠补充记忆中的知识。\n"
        "- **具体应用**必须从本来源证据推出具体处境、行动和反思问题，不能只写“我们要"
        "更好”。\n"
        "- **综合结论**回答这些来源共同怎样塑造理解与实践，同时保留各来源的独立贡献。"
    )


def _render_grounded_outline(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    sections = [
        (
            "### Section 0：解读方法与证据规则\n\n"
            "目的：建立上下文、思想流和结构意识。\n\n"
            "为什么需要：没有方法约束时，最容易把相邻来源、人物和引文混成一个概括。\n\n"
            "应该写什么：说明先问“文本说什么”，再问“为什么在此处这样说”，并记录"
            "每项判断的来源。\n\n"
            "容易犯的错误：把方法材料当作正文来源，或用常识替代文件证据。"
        )
    ]
    for index, spec in enumerate(specs):
        next_label = (
            specs[index + 1].canonical_label
            if index + 1 < len(specs)
            else "综合结论"
        )
        evidence = grounded_facts._source_evidence(spec)
        sections.append(
            f"### Section {index + 1}：{spec.canonical_label}——{_source_topic(spec)}\n\n"
            f"目的：围绕“{_source_topic(spec)}”解释本来源材料。\n\n"
            "为什么需要：该来源拥有独立的主题与引用，必须先证明自己的中心，才可与"
            "相邻来源建立联系。\n\n"
            f"应该写什么：先标明{_source_reference(spec)}，再解释材料摘录“{evidence}”。\n\n"
            f"需要使用哪些材料：`{spec.file_name}`。\n\n"
            "推荐例子：从本来源的引入、解读或具体应用中选择一个具体处境，并说明它"
            "为何体现本来源主题。\n\n"
            f"衔接：本来源结尾提出一个尚待回答的问题，再过渡到{next_label}。\n\n"
            "容易犯的错误：复制相邻来源的主题、扩大引用范围，或只复述而不解释“为什么”。"
        )
    return "\n\n".join(sections)


def _render_grounded_paragraph_guidance(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    paragraphs: list[str] = []
    for index, spec in enumerate(specs, start=1):
        paragraphs.append(
            f"### Paragraph {index}：{spec.canonical_label}\n\n"
            f"目的：证明“{_source_topic(spec)}”是本来源自己的中心。\n\n"
            f"写作逻辑：\nStep 1：准确写出{_source_reference(spec)}及来源"
            f"`{spec.file_name}`。\nStep 2：解释材料摘录中的关键词和关系，不添加"
            "文件外事实。\nStep 3：给出一个具体处境，说明该引文为何改变判断或行动。\n\n"
            "应该包含：观点、解释、源文件证据、具体 Example，以及与下一段的过渡。\n\n"
            f"推荐句式：“{spec.canonical_label}不只是描述‘{_source_topic(spec)}’，"
            f"而是借着{_source_reference(spec)}说明这一主题为什么会改变人的实际选择。”"
        )
    return "\n\n".join(paragraphs)


def _render_grounded_examples(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    if not specs:
        return "材料没有可生成示例的逐份来源。"
    indexes = tuple(
        dict.fromkeys((0, len(specs) // 2, len(specs) - 2, len(specs) - 1))
    )
    examples: list[str] = []
    for index in indexes:
        spec = specs[index]
        evidence = grounded_facts._source_evidence(spec, max_tokens=45)
        examples.append(
            f"### {spec.canonical_label}\n\n"
            f"**普通写法：** “本来源谈到{_source_topic(spec)}，所以我们应该做得更好。”\n\n"
            "**问题：** 只有结论，没有指出引用、材料证据，也没有解释主题为何重要。\n\n"
            f"**高级改写：** “{_source_reference(spec)}把本来源的问题限定在"
            f"‘{_source_topic(spec)}’。材料进一步写到‘{evidence}’。这说明应用不能停在"
            "态度口号，而要说明人在具体关系或行动中怎样回应。”\n\n"
            "**为什么更好：** 它把观点、来源、解释与应用连在一起，读者可以核查推理。"
        )
    return "\n\n".join(examples)


def _render_grounded_draft_review(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    rubric_note = (
        "检测到显式 rubric，但当前证据包未安全复述细则；提交前必须回到原文逐条核对。"
        if "QWOPUS_EXPLICIT_RUBRIC_FOUND=true" in collection_evidence
        else "材料没有显式 rubric，因此只能按题目要求和证据质量自检，不能声称满足某个分值。"
    )
    return (
        "**Opening 为什么成立：** 开篇先交代方法和来源边界，使读者理解后文为何逐份"
        "处理，而不是把相邻来源压成一个泛化主题。\n\n"
        "**论证顺序：** 来源按文件顺序推进；每份内部遵循主题与引用 → 材料解释 → 具体"
        "应用。这个次序让前后衔接可见，也避免倒置来源。\n\n"
        f"**Rubric 对应：** {rubric_note}\n\n"
        f"**可提高之处：** 当前 Draft 已覆盖 {len(specs)} 份来源；正式提交时可把每份的"
        "材料摘录进一步发展成解读段、反例和行动问题，并统一引用格式。\n\n"
        "**严格评分可能扣分处：** 证据摘录若没有页码或段落定位、应用若只写口号、"
        "跨来源比较若未说明共同点和差异，都应继续补强。"
    )


def _render_grounded_checklist(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    rubric = (
        "⚠️ 检测到显式 rubric，但当前证据包未安全复述细则；提交前须回原文逐条核对"
        if "QWOPUS_EXPLICIT_RUBRIC_FOUND=true" in collection_evidence
        else "✅ 已明确材料没有显式 rubric，未虚构分值"
    )
    return "\n".join(
        [
            "✅ 是否回答题目要求：是，按请求的编号部分组织。",
            f"✅ 是否覆盖全部来源：是，{len(file_names)}/{len(file_names)} 个来源。",
            f"✅ 是否覆盖全部材料：是，{len(specs)}/{len(specs)} 份且按顺序列出。",
            rubric,
            "✅ 是否有足够证据：每份包含对应文件、原题、引用和独立正文证据。",
            "✅ 是否有具体例子：示例明确区分普通写法、问题和证据化改写。",
            "✅ 是否有分析而不是总结：每节都要求解释“为什么”并连接应用。",
            "⚠️ 是否达到高质量标准：来源边界、完整性和可核查性已满足；页码、rubric"
            "细则（若有）与更深解读仍须在最终提交前核对。",
        ]
    )


def _classify_section(title: str) -> SectionKind | None:
    """Classify a requested report section using the generic recipe markers."""
    normalized = title.casefold()
    if _title_is_source_understanding(title):
        return SectionKind.SOURCE_UNDERSTANDING
    if any(marker in normalized for marker in ("策略", "strategy")):
        return SectionKind.STRATEGY
    if any(marker in normalized for marker in ("outline", "大纲", "框架")):
        return SectionKind.OUTLINE
    if any(marker in normalized for marker in ("逐段", "paragraph")):
        return SectionKind.PARAGRAPH
    if _title_requires_full_draft(title):
        return SectionKind.FULL_DRAFT
    if any(marker in normalized for marker in ("例子", "示例", "example")):
        return SectionKind.EXAMPLES
    if "draft" in normalized or any(
        marker in normalized for marker in ("后分析", "复盘", "点评")
    ):
        return SectionKind.DRAFT_REVIEW
    if any(marker in normalized for marker in ("checklist", "检查", "清单")):
        return SectionKind.CHECKLIST
    return None


def _render_deterministic_grounded_report(
    *,
    requested: dict[int, str],
    file_names: list[str],
    collection_evidence: str,
    source_specs: tuple[_SourceGroundingSpec, ...] | None = None,
    recipe: ReportRecipe | None = None,
) -> str:
    """Build a complete evidence-only report when the remote model cannot finish."""
    recipe = recipe or default_recipe()
    specs = (
        source_specs
        if source_specs is not None
        else recipe.build_grounding_specs(file_names, collection_evidence)
    )
    sections: list[str] = []
    for number, title in requested.items():
        kind = recipe.section_classifier(title)
        if kind is None or kind not in recipe.renderers:
            raise RuntimeError(
                f"Unsupported grounded report section: {number}. {title}"
            )
        renderer = recipe.renderers[kind]
        body = renderer(
            file_names=file_names,
            specs=specs,
            collection_evidence=collection_evidence,
            recipe=recipe,
            title=title,
        )
        sections.append(f"## {number}. {title}\n\n{body.strip()}")
    return "\n\n".join(sections)


def _validated_grounded_collection(
    *,
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe | None = None,
) -> tuple[_SourceGroundingSpec, ...]:
    """Validate exact source coverage and source-level evidence before composing."""
    recipe = recipe or default_recipe()
    expected_sources = tuple(file_names)
    manifest_sources = grounded_facts._collection_manifest_sources(collection_evidence)
    if manifest_sources != expected_sources:
        raise RuntimeError(
            "Collection source manifest does not exactly match the selected files."
        )
    source_blocks = _collection_source_blocks(collection_evidence)
    if tuple(source_blocks) != expected_sources:
        raise RuntimeError(
            "Collection source blocks do not exactly match the selected files."
        )
    rubric_markers = re.findall(
        r"(?m)^QWOPUS_EXPLICIT_RUBRIC_FOUND=(true|false)\s*$",
        collection_evidence,
    )
    if len(rubric_markers) != 1:
        raise RuntimeError(
            "Collection evidence must contain exactly one explicit-rubric marker."
        )

    recognized_files = [
        file_name
        for file_name in file_names
        if recipe.item_label_from_name(file_name) is not None
    ]
    specs = recipe.build_grounding_specs(file_names, collection_evidence)
    if (
        len(specs) != len(recognized_files)
        or {spec.file_name for spec in specs} != set(recognized_files)
    ):
        raise RuntimeError(
            "Collection evidence does not provide one grounding spec per recognized source."
        )
    incomplete = [
        spec.file_name
        for spec in specs
        if (
            not (
                spec.document_heading
                or spec.topic_lines
                or spec.passage_lines
            )
            or not spec.evidence_excerpt.strip()
        )
    ]
    if incomplete:
        raise RuntimeError(
            "Recognized sources lack structured facts or interpretation evidence: "
            + ", ".join(incomplete)
            + "."
        )
    return specs


def should_use_grounded_report_composer(
    *,
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    has_collection_summary: bool,
    recipe: ReportRecipe | None = None,
) -> bool:
    """Return whether an integrity-critical report can use local grounded composition."""
    recipe = recipe or default_recipe()
    parser_files = set(file_names).difference(spreadsheet_names)
    requested_sections = _requested_numbered_sections(user_question)
    thresholds = recipe.composer_thresholds
    return (
        has_collection_summary
        and not spreadsheet_names
        and len(parser_files) >= thresholds.min_parser_files
        and len(requested_sections) >= thresholds.min_sections
        and recipe.all_source_request_pattern.search(user_question) is not None
        and any(
            _title_requires_full_draft(title)
            for title in requested_sections.values()
        )
    )


_GENERIC_SOURCE_FACT_LABELS = SourceFactLabels(
    document_heading=("document_heading",),
    topic_line=("题目", "題目", "主题", "主題", "title"),
    topic_continuation=("topic_continuation",),
    quote_line=("引用", "引文", "quote", "quotation", "passage"),
    opening_line=("opening_line",),
    topic_stop_labels=(
        "引用",
        "引文",
        "quote",
        "passage",
        "duration",
        "time",
    ),
)


def _generic_item_label_from_name(file_name: str) -> str | None:
    """The generic recipe treats every file as one independently rendered slot."""
    return _source_answer_label(file_name)


def _generic_item_aliases(file_name: str) -> tuple[str, ...]:
    label = _source_answer_label(file_name)
    return (label,) if label else ()


def _generic_reference_key(_text: str) -> tuple[str, tuple[int, ...]] | None:
    return None


def _generic_reference_is_supported(
    key: tuple[str, tuple[int, ...]] | None,
    allowed: frozenset[tuple[str, tuple[int, ...]]],
) -> bool:
    return False


DEFAULT_RECIPE = ReportRecipe(
    source_fact_labels=_GENERIC_SOURCE_FACT_LABELS,
    rubric_markers=(
        r"\brubric\b",
        r"grading\s+criteria",
        r"marking\s+scheme",
        r"评分标准",
        r"评分细则",
        r"评分量表",
        r"评分规则",
        r"打分标准",
    ),
    invented_score_pattern=re.compile(
        r"(?:总分|满分)\s*[：:]?\s*\d+\s*分|"
        r"(?:每项|每条|各项).{0,16}\d+\s*分|"
        r"(?:达到|满足)?\s*\d+\s*分标准|"
        r"\b(?:total|worth)\s*(?:of\s*)?\d+\s*(?:points?|marks?)\b",
        re.IGNORECASE,
    ),
    reference_pattern=re.compile(r"(?!)"),
    all_source_request_pattern=_ALL_SOURCE_REQUEST_PATTERN,
    grounding_rules_text=(
        "GROUNDING_RULES (mandatory):\n"
        "- SOURCE_FACTS are verbatim source excerpts and are the only authority for each file's "
        "document heading, topic, and references.\n"
        "- Never infer, renumber, reconstruct, or complete a reference or quotation from "
        "memory or from a neighboring file. If evidence is absent, say so or call "
        "document_search.\n"
        "- Keep every # File block isolated. Similar adjacent sources remain distinct; "
        "never copy one source's topic, reference, quotation, or example into another.\n"
        "- CORE_INTERPRETATION_EVIDENCE, QUERY_RELEVANT_EVIDENCE, "
        "SECONDARY_INTERPRETATION_EVIDENCE, and APPLICATION_EVIDENCE are verbatim "
        "excerpts selected inside that source; cite their file block and available "
        "chunk_id for source-specific claims. BALANCED_OVERVIEW is extractive orientation "
        "only and does not authorize unsupported facts or quotations.\n"
        "- QWOPUS_EXPLICIT_RUBRIC_FOUND reports whether the selected source text contains "
        "an explicit grading rubric. When it is false, say that no rubric was supplied "
        "and never invent points, weights, totals, or professor criteria."
    ),
    section_markers={
        SectionKind.SOURCE_UNDERSTANDING: (
            "文档理解",
            "文件理解",
            "材料理解",
            "document understanding",
            "source understanding",
        ),
        SectionKind.STRATEGY: ("策略", "strategy"),
        SectionKind.OUTLINE: ("outline", "大纲", "框架"),
        SectionKind.PARAGRAPH: ("逐段", "paragraph"),
        SectionKind.FULL_DRAFT: (),
        SectionKind.EXAMPLES: ("例子", "示例", "example"),
        SectionKind.DRAFT_REVIEW: ("后分析", "复盘", "点评"),
        SectionKind.CHECKLIST: ("checklist", "检查", "清单"),
    },
    evidence_section_markers=(
        r"核心解读|核心解释|解读和讨论|释经|interpretation|exegesis",
        r"具体应用|生活应用|实际应用|反思与应用|application",
    ),
    evidence_claim_boost_terms=("区别", "关系", "核心"),
    composer_thresholds=ComposerThresholds(
        min_parser_files=2,
        min_sections=6,
    ),
    item_label_from_name=_generic_item_label_from_name,
    item_aliases=_generic_item_aliases,
    render_item_heading=_canonical_source_heading,
    reference_key=_generic_reference_key,
    reference_is_supported=_generic_reference_is_supported,
    build_grounding_specs=_build_generic_grounding_specs,
    render_fallback=_render_generic_source_fallback,
    section_classifier=_classify_section,
    renderers={
        SectionKind.SOURCE_UNDERSTANDING: _render_grounded_understanding,
        SectionKind.STRATEGY: _render_grounded_strategy,
        SectionKind.OUTLINE: _render_grounded_outline,
        SectionKind.PARAGRAPH: _render_grounded_paragraph_guidance,
        SectionKind.FULL_DRAFT: _render_grounded_full_draft,
        SectionKind.EXAMPLES: _render_grounded_examples,
        SectionKind.DRAFT_REVIEW: _render_grounded_draft_review,
        SectionKind.CHECKLIST: _render_grounded_checklist,
    },
    validate_candidate_issues=lambda *args, **kwargs: (),
)
