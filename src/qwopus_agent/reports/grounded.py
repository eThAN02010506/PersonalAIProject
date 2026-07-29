"""Rendering and entry policy for deterministic source-grounded reports."""

from __future__ import annotations

import re

from qwopus_agent.reports import grounded_facts
from qwopus_agent.utils.token_budget import truncate_to_tokens

# 原因：报告事实解析和渲染已经分层，但现有 runtime 仍通过 grounded 门面使用稳定名称。
# 作用：集中保留兼容 API；新代码可以直接依赖 grounded_facts 或本渲染模块。
_ALL_SOURCE_REQUEST_PATTERN = grounded_facts._ALL_SOURCE_REQUEST_PATTERN
_SCRIPTURE_REFERENCE_PATTERN = grounded_facts._SCRIPTURE_REFERENCE_PATTERN
_LessonGroundingSpec = grounded_facts._LessonGroundingSpec
_canonical_lesson_heading = grounded_facts._canonical_lesson_heading
_chinese_integer = grounded_facts._chinese_integer
_collection_manifest_sources = grounded_facts._collection_manifest_sources
_collection_source_blocks = grounded_facts._collection_source_blocks
_grounded_application_claim = grounded_facts._grounded_application_claim
_grounded_evidence_claim = grounded_facts._grounded_evidence_claim
_lesson_answer_aliases = grounded_facts._lesson_answer_aliases
_lesson_answer_label = grounded_facts._lesson_answer_label
_lesson_evidence = grounded_facts._lesson_evidence
_lesson_grounding_specs = grounded_facts._lesson_grounding_specs
_lesson_number_from_label = grounded_facts._lesson_number_from_label
_lesson_scripture = grounded_facts._lesson_scripture
_lesson_topic = grounded_facts._lesson_topic
_normalized_fact_text = grounded_facts._normalized_fact_text
_render_grounded_lesson_fallback = grounded_facts._render_grounded_lesson_fallback
_requested_numbered_sections = grounded_facts._requested_numbered_sections
_scripture_reference_is_supported = grounded_facts._scripture_reference_is_supported
_scripture_reference_key = grounded_facts._scripture_reference_key
_source_answer_label = grounded_facts._source_answer_label
_source_application_excerpt = grounded_facts._source_application_excerpt
_source_evidence_excerpt = grounded_facts._source_evidence_excerpt
_source_fact_values = grounded_facts._source_fact_values
_source_tagged_excerpt = grounded_facts._source_tagged_excerpt
_title_requires_full_draft = grounded_facts._title_requires_full_draft
_topic_payload = grounded_facts._topic_payload


def _render_grounded_full_draft(
    specs: tuple[_LessonGroundingSpec, ...],
) -> str:
    """Render an evidence-specific complete Draft with an opening and synthesis."""
    if not specs:
        raise RuntimeError("A complete Draft requires at least one grounded lesson.")
    topic_sequence = " → ".join(_lesson_topic(spec) for spec in specs)
    lesson_body = "\n\n".join(
        _render_grounded_lesson_fallback(spec) for spec in specs
    )
    return (
        "### 引言\n\n"
        f"本报告按 {specs[0].canonical_label} 至 {specs[-1].canonical_label} 的顺序"
        "阅读材料。中心论点不是把所有课程归并成一个抽象口号，而是追踪每课如何用"
        "自己的题目、经文和正文证据推动信仰理解与生活实践。课程标题呈现的推进为："
        f"{topic_sequence}。以下每一节都保留对应文件的证据边界。\n\n"
        f"{lesson_body}\n\n"
        "### 综合结论\n\n"
        "这些材料共同显示，成熟的写作必须同时完成三件事：准确限定每课的经文，解释"
        "材料中的具体论证，并把解释落实到可观察的关系与行动。课程可以彼此衔接，"
        "但不能彼此替代；只有逐课证据链完整后，跨课综合才具有可信度。"
    )


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


def _render_grounded_source_inventory(
    *,
    file_names: list[str],
    collection_evidence: str,
    existing_body: str,
) -> str:
    """Render only source cards still absent from the document-understanding section."""
    blocks = _collection_source_blocks(collection_evidence)
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
            *_source_fact_values(block, "document_heading"),
            *_source_fact_values(block, "topic_line"),
            *_source_fact_values(block, "topic_continuation"),
            *_source_fact_values(block, "scripture_line"),
            *_source_fact_values(block, "opening_line"),
        ]
        evidence = re.sub(
            r"\s+",
            " ",
            truncate_to_tokens(_source_evidence_excerpt(block), 90),
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
    file_names: list[str],
    collection_evidence: str,
    specs: tuple[_LessonGroundingSpec, ...],
) -> str:
    method_count = sum(_lesson_answer_label(name) is None for name in file_names)
    rubric = (
        "检测到显式 rubric，但当前证据包没有安全复述其细则；提交前必须回到原文逐条核对。"
        if "QWOPUS_EXPLICIT_RUBRIC_FOUND=true" in collection_evidence
        else "材料没有提供显式评分 rubric；以下高质量标准属于写作自检，不冒充原始评分规则。"
    )
    return "\n\n".join(
        [
            (
                f"本次任务共有 {len(file_names)} 个来源：{method_count} 份查经方法材料，"
                f"以及 {len(specs)} 份逐课材料。真正要解决的不是把课程压成一个主题，"
                "而是先保持每份来源的边界，再把方法论、逐课经文解释、生活应用和完整"
                "写作成品连成一条可核查的论证链。"
            ),
            _render_grounded_source_inventory(
                file_names=file_names,
                collection_evidence=collection_evidence,
                existing_body="",
            ),
            (
                f"**Rubric 判断：** {rubric}\n\n"
                "**必须包含：** 每课原题、对应经文、该文件的正文证据、为什么该主题重要、"
                "具体应用与来源名；方法材料用于控制上下文、思想流、结构和解释边界。"
            ),
        ]
    )


def _render_grounded_strategy(specs: tuple[_LessonGroundingSpec, ...]) -> str:
    first = specs[0].canonical_label if specs else "第一份材料"
    last = specs[-1].canonical_label if specs else "最后一份材料"
    return (
        "报告采用“方法说明 → 逐课观察 → 主题推进 → 生活应用 → 综合结论”的结构。\n\n"
        "- **方法说明**不是泛泛介绍，而是交代如何使用上下文、思想流和结构来约束解释；"
        "读者由此知道后文为什么不能把相邻课程混写。\n"
        f"- **逐课观察**从{first}到{last}依次展开，每课先固定题目与经文，再解释材料"
        "给出的具体问题。读者应能沿文件名反查证据。\n"
        "- **主题推进**说明前一课如何提出问题、后一课如何继续或转向；这种衔接建立在"
        "题目和经文变化上，不靠补充记忆中的圣经知识。\n"
        "- **生活应用**必须从本课证据推出具体处境、行动和反思问题，不能只写“我们要"
        "更好”。\n"
        "- **综合结论**回答这些课程共同怎样塑造信仰实践，同时保留各课的独立贡献。"
    )


def _render_grounded_outline(specs: tuple[_LessonGroundingSpec, ...]) -> str:
    sections = [
        (
            "### Section 0：查经方法与证据规则\n\n"
            "目的：建立上下文、思想流和结构意识。\n\n"
            "为什么需要：没有方法约束时，最容易把相邻课程、人物和经文混成一个概括。\n\n"
            "应该写什么：说明先问“文本说什么”，再问“为什么在此处这样说”，并记录"
            "每项判断的来源。\n\n"
            "容易犯的错误：把方法材料当作腓立比书课件，或用常识替代文件证据。"
        )
    ]
    for index, spec in enumerate(specs):
        next_label = (
            specs[index + 1].canonical_label
            if index + 1 < len(specs)
            else "综合结论"
        )
        evidence = _lesson_evidence(spec)
        sections.append(
            f"### Section {index + 1}：{spec.canonical_label}——{_lesson_topic(spec)}\n\n"
            f"目的：围绕“{_lesson_topic(spec)}”解释本课材料。\n\n"
            "为什么需要：本课拥有独立的题目与经文，必须先证明自己的中心，才可与"
            "前后课建立联系。\n\n"
            f"应该写什么：先标明{_lesson_scripture(spec)}，再解释材料摘录“{evidence}”。\n\n"
            f"需要使用哪些材料：`{spec.file_name}`。\n\n"
            "推荐例子：从本课破冰、经文讨论或生活应用中选择一个具体处境，并说明它"
            "为何体现本课主题。\n\n"
            f"衔接：本课结尾提出一个尚待回答的问题，再过渡到{next_label}。\n\n"
            "容易犯的错误：复制相邻课的题目、扩大经文范围，或只复述而不解释“为什么”。"
        )
    return "\n\n".join(sections)


def _render_grounded_paragraph_guidance(
    specs: tuple[_LessonGroundingSpec, ...],
) -> str:
    paragraphs: list[str] = []
    for index, spec in enumerate(specs, start=1):
        paragraphs.append(
            f"### Paragraph {index}：{spec.canonical_label}\n\n"
            f"目的：证明“{_lesson_topic(spec)}”是本课自己的中心。\n\n"
            f"写作逻辑：\nStep 1：准确写出{_lesson_scripture(spec)}及来源"
            f"`{spec.file_name}`。\nStep 2：解释材料摘录中的关键词和关系，不添加"
            "文件外事实。\nStep 3：给出一个具体生活处境，说明该经文为何改变判断或行动。\n\n"
            "应该包含：观点、解释、源文件证据、具体 Example，以及与下一段的过渡。\n\n"
            f"推荐句式：“{spec.canonical_label}不只是描述‘{_lesson_topic(spec)}’，"
            f"而是借着{_lesson_scripture(spec)}说明这一主题为什么会改变人的实际选择。”"
        )
    return "\n\n".join(paragraphs)


def _render_grounded_examples(specs: tuple[_LessonGroundingSpec, ...]) -> str:
    if not specs:
        return "材料没有可生成示例的逐课来源。"
    indexes = tuple(
        dict.fromkeys((0, len(specs) // 2, len(specs) - 2, len(specs) - 1))
    )
    examples: list[str] = []
    for index in indexes:
        spec = specs[index]
        evidence = _lesson_evidence(spec, max_tokens=45)
        examples.append(
            f"### {spec.canonical_label}\n\n"
            f"**普通写法：** “本课谈到{_lesson_topic(spec)}，所以我们应该做得更好。”\n\n"
            "**问题：** 只有结论，没有指出经文、材料证据，也没有解释主题为何重要。\n\n"
            f"**高级改写：** “{_lesson_scripture(spec)}把本课的问题限定在"
            f"‘{_lesson_topic(spec)}’。材料进一步写到‘{evidence}’。这说明应用不能停在"
            "态度口号，而要说明人在具体关系或行动中怎样回应。”\n\n"
            "**为什么更好：** 它把观点、来源、解释与应用连在一起，读者可以核查推理。"
        )
    return "\n\n".join(examples)


def _render_grounded_draft_review(
    specs: tuple[_LessonGroundingSpec, ...],
    collection_evidence: str,
) -> str:
    rubric_note = (
        "检测到显式 rubric，但当前证据包未安全复述细则；提交前必须回到原文逐条核对。"
        if "QWOPUS_EXPLICIT_RUBRIC_FOUND=true" in collection_evidence
        else "材料没有显式 rubric，因此只能按题目要求和证据质量自检，不能声称满足某个分值。"
    )
    return (
        "**Opening 为什么成立：** 开篇先交代方法和来源边界，使读者理解后文为何逐课"
        "处理，而不是把 21～31 课压成一个泛化主题。\n\n"
        "**论证顺序：** 课程按课号推进；每课内部遵循题目与经文 → 材料解释 → 生活"
        "应用。这个次序让前后衔接可见，也避免倒置来源。\n\n"
        f"**Rubric 对应：** {rubric_note}\n\n"
        f"**可提高之处：** 当前 Draft 已覆盖 {len(specs)} 课；正式提交时可把每课的"
        "材料摘录进一步发展成释经段、反例和行动问题，并统一引用格式。\n\n"
        "**严格评分可能扣分处：** 证据摘录若没有页码或段落定位、应用若只写口号、"
        "跨课比较若未说明共同点和差异，都应继续补强。"
    )


def _render_grounded_checklist(
    *,
    file_names: list[str],
    specs: tuple[_LessonGroundingSpec, ...],
    collection_evidence: str,
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
            f"✅ 是否覆盖全部课程：是，{len(specs)}/{len(specs)} 课且按课号排序。",
            rubric,
            "✅ 是否有足够证据：每课包含对应文件、原题、经文和独立正文证据。",
            "✅ 是否有具体例子：示例明确区分普通写法、问题和证据化改写。",
            "✅ 是否有分析而不是总结：每节都要求解释“为什么”并连接应用。",
            "⚠️ 是否达到高质量标准：来源边界、完整性和可核查性已满足；页码、rubric"
            "细则（若有）与更深释经仍须在最终提交前核对。",
        ]
    )


def _validated_grounded_collection(
    *,
    file_names: list[str],
    collection_evidence: str,
) -> tuple[_LessonGroundingSpec, ...]:
    """Validate exact source coverage and lesson-level evidence before composing."""
    expected_sources = tuple(file_names)
    manifest_sources = _collection_manifest_sources(collection_evidence)
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

    lesson_files = [
        file_name
        for file_name in file_names
        if _lesson_answer_label(file_name) is not None
    ]
    lesson_numbers = [
        number
        for file_name in lesson_files
        if (
            number := _lesson_number_from_label(
                _lesson_answer_label(file_name) or ""
            )
        )
        is not None
    ]
    if (
        len(lesson_numbers) != len(lesson_files)
        or len(lesson_numbers) != len(set(lesson_numbers))
    ):
        raise RuntimeError(
            "Selected lesson sources contain an unparseable or duplicate lesson number."
        )

    specs = _lesson_grounding_specs(file_names, collection_evidence)
    if (
        len(specs) != len(lesson_files)
        or {spec.file_name for spec in specs} != set(lesson_files)
    ):
        raise RuntimeError(
            "Collection evidence does not provide one grounding spec per lesson source."
        )
    incomplete = [
        spec.file_name
        for spec in specs
        if (
            not (
                spec.document_heading
                or spec.topic_lines
                or spec.scripture_lines
            )
            or not spec.evidence_excerpt.strip()
        )
    ]
    if incomplete:
        raise RuntimeError(
            "Lesson sources lack structured facts or interpretation evidence: "
            + ", ".join(incomplete)
            + "."
        )
    return specs


def _render_deterministic_grounded_report(
    *,
    requested: dict[int, str],
    file_names: list[str],
    collection_evidence: str,
    lesson_specs: tuple[_LessonGroundingSpec, ...] | None = None,
) -> str:
    """Build a complete evidence-only report when the remote model cannot finish."""
    specs = (
        lesson_specs
        if lesson_specs is not None
        else _lesson_grounding_specs(file_names, collection_evidence)
    )
    sections: list[str] = []
    for number, title in requested.items():
        normalized = title.casefold()
        if _title_is_source_understanding(title):
            body = _render_grounded_understanding(
                file_names=file_names,
                collection_evidence=collection_evidence,
                specs=specs,
            )
        elif any(marker in normalized for marker in ("策略", "strategy")):
            body = _render_grounded_strategy(specs)
        elif any(marker in normalized for marker in ("outline", "大纲", "框架")):
            body = _render_grounded_outline(specs)
        elif any(marker in normalized for marker in ("逐段", "paragraph")):
            body = _render_grounded_paragraph_guidance(specs)
        elif _title_requires_full_draft(title):
            body = _render_grounded_full_draft(specs)
        elif any(marker in normalized for marker in ("例子", "示例", "example")):
            body = _render_grounded_examples(specs)
        elif "draft" in normalized or any(
            marker in normalized for marker in ("后分析", "复盘", "点评")
        ):
            body = _render_grounded_draft_review(specs, collection_evidence)
        elif any(marker in normalized for marker in ("checklist", "检查", "清单")):
            body = _render_grounded_checklist(
                file_names=file_names,
                specs=specs,
                collection_evidence=collection_evidence,
            )
        else:
            raise RuntimeError(
                f"Unsupported grounded report section: {number}. {title}"
            )
        sections.append(f"## {number}. {title}\n\n{body.strip()}")
    return "\n\n".join(sections)


def should_use_grounded_report_composer(
    *,
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    has_collection_summary: bool,
) -> bool:
    """Return whether an integrity-critical report can use local grounded composition."""
    parser_files = set(file_names).difference(spreadsheet_names)
    requested_sections = _requested_numbered_sections(user_question)
    return (
        has_collection_summary
        and not spreadsheet_names
        and len(parser_files) > 1
        and len(requested_sections) >= 6
        and _ALL_SOURCE_REQUEST_PATTERN.search(user_question) is not None
        and any(
            _title_requires_full_draft(title)
            for title in requested_sections.values()
        )
        and sum(
            _lesson_answer_label(file_name) is not None
            for file_name in file_names
        )
        >= 2
    )
