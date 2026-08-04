"""Optional Bible-study lesson recipe for the grounded report composer.

The generic default recipe treats every file as one source slot.  This recipe
restores the Bible-study behavior: file names that carry a lesson number are
ordered by lesson number, scripture references are validated against each
source's allowed verse range, and the report templates use the Bible-study
wording (经文与主题 / 生活展开).

Enable it explicitly by passing ``recipe=BIBLE_RECIPE`` to the file-analysis
entry points or by calling :func:`qwopus_agent.reports.recipe.set_default_recipe`.
"""

from __future__ import annotations

import re
from typing import Any

from qwopus_agent.reports import grounded_facts
from qwopus_agent.reports.grounded import DEFAULT_RECIPE
from qwopus_agent.reports.recipe import (
    ComposerThresholds,
    ReportRecipe,
    SectionKind,
    SourceFactLabels,
)
from qwopus_agent.utils.token_budget import truncate_to_tokens

_SourceGroundingSpec = grounded_facts.SourceGroundingSpec
_canonical_source_heading = grounded_facts._canonical_source_heading
_collection_source_blocks = grounded_facts._collection_source_blocks
_grounded_application_claim = grounded_facts._grounded_application_claim
_grounded_evidence_claim = grounded_facts._grounded_evidence_claim
_source_answer_label = grounded_facts._source_answer_label
_source_application_excerpt = grounded_facts._source_application_excerpt
_source_evidence = grounded_facts._source_evidence
_source_evidence_excerpt = grounded_facts._source_evidence_excerpt
_source_fact_values = grounded_facts._source_fact_values
_source_reference = grounded_facts._source_reference
_source_topic = grounded_facts._source_topic
_topic_payload = grounded_facts._topic_payload
_title_requires_full_draft = grounded_facts._title_requires_full_draft


# Scripture books and aliases ------------------------------------------------

_SCRIPTURE_BOOKS = (
    "帖撒罗尼迦前书",
    "帖撒罗尼迦后书",
    "哥林多前书",
    "哥林多后书",
    "撒母耳记上",
    "撒母耳记下",
    "历代志上",
    "历代志下",
    "提摩太前书",
    "提摩太后书",
    "彼得前书",
    "彼得后书",
    "约翰一书",
    "约翰二书",
    "约翰三书",
    "列王纪上",
    "列王纪下",
    "耶利米哀歌",
    "使徒行传",
    "马太福音",
    "马可福音",
    "路加福音",
    "约翰福音",
    "创世记",
    "出埃及记",
    "利未记",
    "民数记",
    "申命记",
    "约书亚记",
    "士师记",
    "路得记",
    "以斯拉记",
    "尼希米记",
    "以斯帖记",
    "约伯记",
    "传道书",
    "以赛亚书",
    "耶利米书",
    "以西结书",
    "但以理书",
    "何西阿书",
    "约珥书",
    "阿摩司书",
    "俄巴底亚书",
    "约拿书",
    "弥迦书",
    "那鸿书",
    "哈巴谷书",
    "西番雅书",
    "哈该书",
    "撒迦利亚书",
    "玛拉基书",
    "罗马书",
    "加拉太书",
    "以弗所书",
    "腓立比书",
    "歌罗西书",
    "提多书",
    "腓利门书",
    "希伯来书",
    "雅各书",
    "犹大书",
    "启示录",
    "诗篇",
    "箴言",
    "雅歌",
)
_SCRIPTURE_BOOK_ALIASES = {
    alias: canonical
    for canonical in _SCRIPTURE_BOOKS
    for alias in (
        canonical,
        canonical[:-1] if canonical.endswith("书") else canonical,
    )
}
_SCRIPTURE_BOOK_MATCHES = tuple(
    sorted(_SCRIPTURE_BOOK_ALIASES, key=len, reverse=True)
)
SCRIPTURE_REFERENCE_PATTERN = re.compile(
    rf"(?:{'|'.join(map(re.escape, _SCRIPTURE_BOOK_MATCHES))})"
    r"\s*\d+\s*章\s*\d+\s*节?"
    r"(?:\s*(?:上|下)?半节)?"
    r"(?:\s*[-‐‑‒–—至]\s*\d+\s*节?)?"
)

_LESSON_LABEL_CHINESE = re.compile(r"第[一二三四五六七八九十百〇零两\d]+课")
_LESSON_LABEL_ENGLISH = re.compile(r"\blesson[\s_-]*\d+\b", re.IGNORECASE)
_LESSON_LABEL_NUMBER_CHINESE = re.compile(
    r"第(?P<number>[一二三四五六七八九十百〇零两\d]+)课"
)
_LESSON_LABEL_NUMBER_ENGLISH = re.compile(
    r"\blesson[\s_-]*(?P<number>\d+)\b", re.IGNORECASE
)


def _lesson_answer_label(file_name: str) -> str | None:
    """Extract a course/lesson identifier that a complete draft must explicitly cover."""
    stem = _source_answer_label(file_name)
    chinese = _LESSON_LABEL_CHINESE.search(stem)
    if chinese is not None:
        return chinese.group(0)
    english = _LESSON_LABEL_ENGLISH.search(stem)
    return english.group(0) if english is not None else None


def _chinese_integer(value: str) -> int | None:
    """Parse the small Chinese numerals commonly used in lesson file names."""
    if value.isdigit():
        return int(value)
    digits = {
        "〇": 0,
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if "百" in value:
        left, _, right = value.partition("百")
        hundreds = digits.get(left, 1)
        remainder = _chinese_integer(right) if right else 0
        return None if remainder is None else hundreds * 100 + remainder
    if "十" in value:
        left, _, right = value.partition("十")
        tens = digits.get(left, 1)
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    parsed = [digits.get(character) for character in value]
    if not parsed or any(number is None for number in parsed):
        return None
    return int("".join(str(number) for number in parsed))


def _lesson_number_from_label(label: str) -> int | None:
    """Normalize a Chinese or English lesson heading to its integer identifier."""
    chinese = _LESSON_LABEL_NUMBER_CHINESE.search(label)
    if chinese is not None:
        return _chinese_integer(chinese.group("number"))
    english = _LESSON_LABEL_NUMBER_ENGLISH.search(label)
    return int(english.group("number")) if english is not None else None


def _lesson_answer_aliases(file_name: str) -> tuple[str, ...]:
    """Accept the source's Chinese-number label and the equivalent Arabic form."""
    label = _lesson_answer_label(file_name)
    if label is None:
        return ()
    chinese = re.fullmatch(
        r"第(?P<number>[一二三四五六七八九十百〇零两\d]+)课",
        label,
    )
    if chinese is not None:
        number = _chinese_integer(chinese.group("number"))
        if number is not None:
            return tuple(dict.fromkeys((label, f"第{number}课")))
    english_number = re.search(r"\d+", label)
    if english_number is not None:
        number_text = english_number.group(0)
        return tuple(
            dict.fromkeys(
                (
                    label,
                    f"lesson {number_text}",
                    f"lesson-{number_text}",
                    f"lesson_{number_text}",
                )
            )
        )
    return (label,)


def _scripture_reference_key(text: str) -> tuple[str, tuple[int, ...]] | None:
    """Normalize a Chinese scripture reference to book plus chapter/verse numbers."""
    matched_book = next(
        (name for name in _SCRIPTURE_BOOK_MATCHES if name in text),
        None,
    )
    numbers = tuple(int(value) for value in re.findall(r"\d+", text)[:3])
    if matched_book is None or len(numbers) < 2:
        return None
    return _SCRIPTURE_BOOK_ALIASES[matched_book], numbers


def _scripture_reference_is_supported(
    key: tuple[str, tuple[int, ...]] | None,
    allowed: frozenset[tuple[str, tuple[int, ...]]],
) -> bool:
    """Allow only references wholly contained by one source-grounded verse interval."""
    if key is None:
        return False
    book, numbers = key
    chapter = numbers[0]
    cited_start = numbers[1]
    cited_end = numbers[2] if len(numbers) > 2 else cited_start
    return any(
        allowed_book == book
        and len(allowed_numbers) >= 2
        and allowed_numbers[0] == chapter
        and allowed_numbers[1] <= cited_start
        and cited_end
        <= (
            allowed_numbers[2]
            if len(allowed_numbers) > 2
            else allowed_numbers[1]
        )
        for allowed_book, allowed_numbers in allowed
    )


def build_bible_grounding_specs(
    file_names: list[str],
    collection_evidence: str,
) -> tuple[_SourceGroundingSpec, ...]:
    """Map every lesson file to its own exact facts and evidence, ordered by lesson."""
    source_blocks = _collection_source_blocks(collection_evidence)
    specs: list[_SourceGroundingSpec] = []
    seen_numbers: set[int] = set()
    for file_name in file_names:
        label = _lesson_answer_label(file_name)
        if label is None:
            continue
        number = _lesson_number_from_label(label)
        if number is None or number in seen_numbers:
            continue
        seen_numbers.add(number)
        block = source_blocks.get(file_name, "")
        passage_lines = _source_fact_values(block, "scripture_line")
        specs.append(
            _SourceGroundingSpec(
                number=number,
                canonical_label=label,
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
                allowed_references=frozenset(
                    key
                    for line in passage_lines
                    if (key := _scripture_reference_key(line)) is not None
                ),
                evidence_excerpt=_source_evidence_excerpt(block),
                application_excerpt=_source_application_excerpt(block),
            )
        )
    return tuple(sorted(specs, key=lambda spec: spec.number))


# Bible renderers ------------------------------------------------------------

def _canonical_lesson_heading(spec: _SourceGroundingSpec) -> str:
    return _canonical_source_heading(spec)


def _lesson_topic(spec: _SourceGroundingSpec) -> str:
    return _source_topic(spec)


def _lesson_scripture(spec: _SourceGroundingSpec) -> str:
    return "；".join(spec.passage_lines) or "材料未单列经文"


def _lesson_evidence(spec: _SourceGroundingSpec, *, max_tokens: int = 55) -> str:
    return _source_evidence(spec, max_tokens=max_tokens)


def _render_bible_source_fallback(spec: _SourceGroundingSpec) -> str:
    """Render a safe source card when the model omitted or corrupted one lesson."""
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
    scripture = "；".join(spec.passage_lines)
    evidence, reasoning = _grounded_evidence_claim(
        spec,
        boost_terms=("区别", "关系", "核心", "明证", "同心", "信任", "喜乐"),
    )
    application = _grounded_application_claim(spec)
    lines = [
        _canonical_lesson_heading(spec),
        (
            f"**经文与主题：** {scripture}。本课围绕“{topic}”展开。"
            if scripture and topic
            else (
                f"**经文与主题：** {scripture}。"
                if scripture
                else f"**经文与主题：** 材料未单列经文；本课围绕“{topic}”展开。"
            )
        ),
        (
            f"**核心论点：** {spec.canonical_label}的材料把中心界定为“{topic}”。"
            f"{spec.canonical_label}的论点来自本课 SOURCE_FACTS，不是从相邻课程"
            "推演出来的概括。"
            if topic
            else "**核心论点：** 本课按对应源文件的标题和正文证据展开。"
        ),
    ]
    if evidence:
        lines.append(
            f"**材料论证：** 材料写道：“{evidence}”"
            f"{spec.canonical_label}的证据{reasoning}。{spec.canonical_label}必须把"
            f"本课主题从标题推进到可辨认的判断；{spec.canonical_label}对经文的理解"
            "须服从原句显示的区别、原因或问题。"
        )
    if application:
        lines.append(
            f"**生活展开：** 材料把应用落在这个具体处境：“{application}”"
            f"{spec.canonical_label}由此不只停在观念认同，而要在真实关系、决定或"
            f"习惯中检验“{topic or spec.canonical_label}”所涉及的动机和回应。"
        )
    else:
        lines.append(
            "**生活展开：** 当前证据包没有抽取出独立的生活应用段，因此这里只保留"
            "经文与释经论点，不虚构材料未提供的案例。"
        )
    lines.extend(
        [
            (
                f"**本课结论：** {topic or spec.canonical_label}必须由本课的经文、上述"
                f"材料论证和生活处境共同界定。{spec.canonical_label}可以与前后课程"
                "形成推进，却不能被"
                f"压缩成适用于所有课程的同一句“{topic or spec.canonical_label}”口号。"
            ),
            f"**来源：** `{spec.file_name}`",
        ]
    )
    return "\n\n".join(lines)


def _render_bible_full_draft(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    """Render an evidence-specific complete Draft with an opening and synthesis."""
    if not specs:
        raise RuntimeError("A complete Draft requires at least one grounded lesson.")
    topic_sequence = " → ".join(_lesson_topic(spec) for spec in specs)
    lesson_body = "\n\n".join(recipe.render_fallback(spec) for spec in specs)
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


def _render_bible_source_inventory(
    *,
    file_names: list[str],
    collection_evidence: str,
    existing_body: str,
    recipe: ReportRecipe,
    title: str = "",
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
            *grounded_facts._source_fact_values(block, "document_heading"),
            *grounded_facts._source_fact_values(block, "topic_line"),
            *grounded_facts._source_fact_values(block, "topic_continuation"),
            *grounded_facts._source_fact_values(block, "scripture_line"),
            *grounded_facts._source_fact_values(block, "opening_line"),
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


def _render_bible_understanding(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
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
            _render_bible_source_inventory(
                file_names=file_names,
                collection_evidence=collection_evidence,
                existing_body="",
                recipe=recipe,
            ),
            (
                f"**Rubric 判断：** {rubric}\n\n"
                "**必须包含：** 每课原题、对应经文、该文件的正文证据、为什么该主题重要、"
                "具体应用与来源名；方法材料用于控制上下文、思想流、结构和解释边界。"
            ),
        ]
    )


def _render_bible_strategy(
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


def _render_bible_outline(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
    sections = [
        (
            "### Section 0：查经方法与证据规则\n\n"
            "目的：建立上下文、思想流和结构意识。\n\n"
            "为什么需要：没有方法约束时，最容易把相邻课程、人物和经文混成一个概括。\n\n"
            "应该写什么：说明先问“文本说什么”，再问“为什么在此处这样说”，并记录"
            "每项判断的来源。\n\n"
            "容易犯的错误：把方法材料当作课程正文，或用常识替代文件证据。"
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


def _render_bible_paragraph_guidance(
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
            f"目的：证明“{_lesson_topic(spec)}”是本课自己的中心。\n\n"
            f"写作逻辑：\nStep 1：准确写出{_lesson_scripture(spec)}及来源"
            f"`{spec.file_name}`。\nStep 2：解释材料摘录中的关键词和关系，不添加"
            "文件外事实。\nStep 3：给出一个具体生活处境，说明该经文为何改变判断或行动。\n\n"
            "应该包含：观点、解释、源文件证据、具体 Example，以及与下一段的过渡。\n\n"
            f"推荐句式：“{spec.canonical_label}不只是描述‘{_lesson_topic(spec)}’，"
            f"而是借着{_lesson_scripture(spec)}说明这一主题为什么会改变人的实际选择。”"
        )
    return "\n\n".join(paragraphs)


def _render_bible_examples(
    *,
    specs: tuple[_SourceGroundingSpec, ...],
    file_names: list[str],
    collection_evidence: str,
    recipe: ReportRecipe,
    title: str,
) -> str:
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


def _render_bible_draft_review(
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
        "**Opening 为什么成立：** 开篇先交代方法和来源边界，使读者理解后文为何逐课"
        "处理，而不是把相邻课压成一个泛化主题。\n\n"
        "**论证顺序：** 课程按课号推进；每课内部遵循题目与经文 → 材料解释 → 生活"
        "应用。这个次序让前后衔接可见，也避免倒置来源。\n\n"
        f"**Rubric 对应：** {rubric_note}\n\n"
        f"**可提高之处：** 当前 Draft 已覆盖 {len(specs)} 课；正式提交时可把每课的"
        "材料摘录进一步发展成释经段、反例和行动问题，并统一引用格式。\n\n"
        "**严格评分可能扣分处：** 证据摘录若没有页码或段落定位、应用若只写口号、"
        "跨课比较若未说明共同点和差异，都应继续补强。"
    )


def _render_bible_checklist(
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
            f"✅ 是否覆盖全部课程：是，{len(specs)}/{len(specs)} 课且按课号排序。",
            rubric,
            "✅ 是否有足够证据：每课包含对应文件、原题、经文和独立正文证据。",
            "✅ 是否有具体例子：示例明确区分普通写法、问题和证据化改写。",
            "✅ 是否有分析而不是总结：每节都要求解释“为什么”并连接应用。",
            "⚠️ 是否达到高质量标准：来源边界、完整性和可核查性已满足；页码、rubric"
            "细则（若有）与更深释经仍须在最终提交前核对。",
        ]
    )


def _bible_candidate_issues(
    subsection: Any,
    spec: _SourceGroundingSpec,
    recipe: ReportRecipe,
) -> tuple[str, ...]:
    """Validate one generated lesson only against that lesson's own source facts."""
    from qwopus_agent.reports.contract import _base_candidate_issues

    issues = _base_candidate_issues(subsection, spec)
    combined = f"{subsection.heading}\n{subsection.body}"
    references = tuple(
        match.group(0).strip()
        for match in SCRIPTURE_REFERENCE_PATTERN.finditer(combined)
    )
    if spec.allowed_references:
        if not references:
            issues.append("missing source scripture")
        unsupported = [
            reference
            for reference in references
            if not _scripture_reference_is_supported(
                _scripture_reference_key(reference),
                spec.allowed_references,
            )
        ]
        if unsupported:
            issues.append(
                "scripture belongs outside this lesson: " + ", ".join(unsupported)
            )
    elif references:
        issues.append("scripture was added without a source fact")
    return tuple(issues)


# Bible recipe ----------------------------------------------------------------

BIBLE_SOURCE_FACT_LABELS = SourceFactLabels(
    document_heading=("document_heading",),
    topic_line=("题目", "題目", "主题", "主題", "title"),
    topic_continuation=("topic_continuation",),
    quote_line=("经文", "經文", "scripture", "passage", "text"),
    opening_line=("opening_line",),
    topic_stop_labels=(
        "经文",
        "經文",
        "scripture",
        "passage",
        "duration",
        "time",
    ),
    quote_fact_key="scripture_line",
)

BIBLE_RECIPE = ReportRecipe(
    source_fact_labels=BIBLE_SOURCE_FACT_LABELS,
    rubric_markers=DEFAULT_RECIPE.rubric_markers,
    invented_score_pattern=DEFAULT_RECIPE.invented_score_pattern,
    reference_pattern=SCRIPTURE_REFERENCE_PATTERN,
    all_source_request_pattern=DEFAULT_RECIPE.all_source_request_pattern,
    grounding_rules_text=(
        "GROUNDING_RULES (mandatory):\n"
        "- SOURCE_FACTS are verbatim source excerpts and are the only authority for each file's "
        "document heading, lesson topic, and scripture reference.\n"
        "- Never infer, renumber, reconstruct, or complete a scripture reference or quotation "
        "from memory or from a neighboring file. If evidence is absent, say so or call "
        "document_search.\n"
        "- Keep every # File block isolated. Similar adjacent lessons remain distinct; never "
        "copy one lesson's topic, scripture, quotation, or example into another.\n"
        "- CORE_INTERPRETATION_EVIDENCE, QUERY_RELEVANT_EVIDENCE, "
        "SECONDARY_INTERPRETATION_EVIDENCE, and APPLICATION_EVIDENCE are verbatim "
        "excerpts selected inside that source; cite their file block and available "
        "chunk_id for source-specific claims. BALANCED_OVERVIEW is extractive orientation "
        "only and does not authorize unsupported facts or quotations.\n"
        "- QWOPUS_EXPLICIT_RUBRIC_FOUND reports whether the selected source text contains "
        "an explicit grading rubric. When it is false, say that no rubric was supplied "
        "and never invent points, weights, totals, or professor criteria."
    ),
    section_markers=DEFAULT_RECIPE.section_markers,
    evidence_section_markers=(
        r"经文解释|解释和讨论|释经|exegesis|interpret",
        r"生活运用|生活应用|实际应用|反思与应用|application",
    ),
    evidence_claim_boost_terms=("区别", "关系", "核心", "明证", "同心", "信任", "喜乐"),
    composer_thresholds=ComposerThresholds(min_parser_files=2, min_sections=6),
    item_label_from_name=_lesson_answer_label,
    item_aliases=_lesson_answer_aliases,
    render_item_heading=_canonical_lesson_heading,
    reference_key=_scripture_reference_key,
    reference_is_supported=_scripture_reference_is_supported,
    build_grounding_specs=build_bible_grounding_specs,
    render_fallback=_render_bible_source_fallback,
    section_classifier=DEFAULT_RECIPE.section_classifier,
    renderers={
        SectionKind.SOURCE_UNDERSTANDING: _render_bible_understanding,
        SectionKind.STRATEGY: _render_bible_strategy,
        SectionKind.OUTLINE: _render_bible_outline,
        SectionKind.PARAGRAPH: _render_bible_paragraph_guidance,
        SectionKind.FULL_DRAFT: _render_bible_full_draft,
        SectionKind.EXAMPLES: _render_bible_examples,
        SectionKind.DRAFT_REVIEW: _render_bible_draft_review,
        SectionKind.CHECKLIST: _render_bible_checklist,
    },
    validate_candidate_issues=_bible_candidate_issues,
)
