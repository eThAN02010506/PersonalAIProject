"""smolagents Tool adapters for Qwopus-Agent capabilities."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from qwopus_agent.documents import DocumentStructure, HierarchicalDocumentSummary
from qwopus_agent.integrations.smolagents_data_tools import (
    build_excel_analysis_tool,
    build_excel_schema_tool,
)
from qwopus_agent.integrations.smolagents_knowledge_tools import (
    build_graph_search_tool,
    build_minirag_search_tool,
)
from qwopus_agent.integrations.smolagents_web_tools import build_tavily_search_tool
from qwopus_agent.integrations.tavily import TavilySearchConfig
from qwopus_agent.memory.knowledge_store import KnowledgeStore
from qwopus_agent.utils.token_budget import (
    TokenBudgetManager,
    estimate_tokens,
    truncate_to_tokens,
)

__all__ = [
    "TavilySearchConfig",
    "build_document_collection_summary_tool",
    "build_direct_document_search_tool",
    "build_document_outline_tool",
    "build_document_search_tool",
    "build_document_section_tool",
    "build_document_summary_tool",
    "build_excel_analysis_tool",
    "build_excel_schema_tool",
    "build_graph_search_tool",
    "build_minirag_search_tool",
    "build_tavily_search_tool",
]


def build_document_outline_tool(
    documents: Mapping[str, DocumentStructure],
    *,
    budget_manager: TokenBudgetManager | None = None,
) -> Any:
    """Expose heading hierarchy without returning the full document body."""
    Tool = _smolagents_tool_class()
    if not documents:
        raise ValueError("documents must contain at least one parsed document.")
    budget = budget_manager or TokenBudgetManager()
    available_files = ", ".join(documents)

    class DocumentOutlineTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_outline"
        description = (
            "List the heading hierarchy, section ids, and page ranges for one uploaded "
            f"document. Use this before selecting a section. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded file name.",
            }
        }
        output_type = "string"

        def forward(self, file_name: str) -> str:
            structure = _lookup_file_value(documents, file_name)
            lines = [f"# Outline: {file_name}"]
            for section in structure.sections:
                indent = "  " * max(section.level - 1, 0)
                pages = _page_label(section.page_start, section.page_end)
                lines.append(
                    f"{indent}- {section.title} [section_id={section.id}{pages}]"
                )
            return truncate_to_tokens(
                "\n".join(lines),
                budget.observation_budget,
            )

    return DocumentOutlineTool()


def build_document_search_tool(
    minirag: KnowledgeStore,
    documents: Mapping[str, DocumentStructure],
    *,
    min_relevance: float = 0.55,
    selected_section_ids: Mapping[str, tuple[str, ...]] | None = None,
    budget_manager: TokenBudgetManager | None = None,
) -> Any:
    """Search only the requested current document and optional selected sections."""
    Tool = _smolagents_tool_class()
    if not documents:
        raise ValueError("documents must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    available_files = ", ".join(documents)

    class DocumentSearchTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_search"
        description = (
            "Search relevant evidence chunks inside one current uploaded document. "
            "Use this for specific questions instead of reading the document from the start. "
            f"Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded file name.",
            },
            "query": {
                "type": "string",
                "description": "Question or evidence to find in this file.",
            },
        }
        output_type = "string"

        def forward(self, file_name: str, query: str) -> str:
            structure = _lookup_file_value(documents, file_name)
            allowed_sections = (selected_section_ids or {}).get(file_name)
            results = minirag.search(
                query,
                min_relevance=min_relevance,
                document_ids=(structure.document_id,),
                section_ids=allowed_sections,
            )
            if not results:
                allowed = set(allowed_sections or ())
                ranked = sorted(
                    (
                        (_direct_match_score(query, chunk), chunk)
                        for chunk in structure.chunks
                        if not allowed or chunk.section_id in allowed
                    ),
                    key=lambda item: (-item[0], item[1].position),
                )
                matches = [chunk for score, chunk in ranked if score > 0][:8]
                if not matches:
                    return f"No relevant evidence found in {file_name}."
                # 原因：短文档的正确原文可能低于 embedding 阈值，即使查询含完全相同的关键词。
                # 作用：只在当前指定文件内进行字面回退，补召回原文而不引入其他来源。
                return _render_section_chunks(
                    file_name,
                    matches,
                    max_tokens=budget.observation_budget,
                )
            # 原因：检索已经限定当前文件，仍需服从当前模型可用的证据 token。
            # 作用：只压缩 Tool Observation，不截断持久化原文和章节 Chunk。
            return truncate_to_tokens(
                "\n\n".join(results),
                budget.observation_budget,
            )

    return DocumentSearchTool()


def build_direct_document_search_tool(
    documents: Mapping[str, DocumentStructure],
    *,
    selected_section_ids: Mapping[str, tuple[str, ...]] | None = None,
    budget_manager: TokenBudgetManager | None = None,
    max_chunks: int = 8,
) -> Any:
    """Search parsed current documents locally without a vector index."""
    Tool = _smolagents_tool_class()
    if not documents:
        raise ValueError("documents must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    available_files = ", ".join(documents)

    class DirectDocumentSearchTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_search"
        description = (
            "Search parsed chunks inside one selected local file without MiniRAG or a vector "
            f"index. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact selected file path shown in the file list.",
            },
            "query": {
                "type": "string",
                "description": "Words or evidence to find in this file.",
            },
        }
        output_type = "string"

        def forward(self, file_name: str, query: str) -> str:
            structure = _lookup_file_value(documents, file_name)
            allowed_sections = set((selected_section_ids or {}).get(file_name, ()))
            ranked = sorted(
                (
                    (_direct_match_score(query, chunk), chunk)
                    for chunk in structure.chunks
                    if not allowed_sections or chunk.section_id in allowed_sections
                ),
                key=lambda item: (-item[0], item[1].position),
            )
            matches = [chunk for score, chunk in ranked if score > 0][:max_chunks]
            if not matches:
                return f"No directly matching evidence found in {file_name}."
            # 原因：目录模式应直接检查内存中的解析结果，不能为一次分析建立持久向量索引。
            # 作用：按字面相关性选择当前文件 chunk，并沿用统一引用和 token 限制。
            return _render_section_chunks(
                file_name,
                matches,
                max_tokens=budget.observation_budget,
            )

    return DirectDocumentSearchTool()


def build_document_collection_summary_tool(
    summaries: Mapping[str, HierarchicalDocumentSummary],
    *,
    documents: Mapping[str, DocumentStructure] | None = None,
    query: str = "",
    budget_manager: TokenBudgetManager | None = None,
) -> Any:
    """Expose balanced, source-labelled evidence for every selected document."""
    Tool = _smolagents_tool_class()
    if not summaries:
        raise ValueError("summaries must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    ordered_sources = tuple(summaries)

    class DocumentCollectionSummaryTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_collection_summary"
        description = (
            "Read verbatim source facts, a source-balanced overview, and cited evidence for "
            "every selected document in one call. This tool returns a machine-verifiable "
            "coverage manifest and must be used before multi-document synthesis."
        )
        inputs: dict[str, dict[str, str]] = {}
        output_type = "string"

        def forward(self) -> str:
            # 原因：逐文件 Tool 调用会受 Agent 步数限制，较大文件夹可能只分析前几份。
            # 作用：先为每个来源保留不可截断的标题和 manifest，再平均分配正文预算；
            # 若连最小证据槽都放不下则明确失败，绝不以前缀截断伪装成全覆盖。
            payloads = {
                file_name: _collection_source_payload(
                    file_name=file_name,
                    summary=summary,
                    structure=(documents or {}).get(file_name),
                    query=query,
                )
                for file_name, summary in summaries.items()
            }
            covered_sources = tuple(
                file_name
                for file_name in ordered_sources
                if any(part.strip() for part in payloads[file_name])
            )
            explicit_rubric_found = _documents_contain_explicit_rubric(
                documents or {}
            )
            return _pack_collection_evidence(
                payloads=payloads,
                covered_sources=covered_sources,
                explicit_rubric_found=explicit_rubric_found,
                # 原因：这是一次覆盖整个来源集合的合成证据，不是普通单文件 Tool 结果；
                # 6000-token observation cap 会让 13 个来源只剩约 400 token/份。
                # 作用：沿用模型窗口扣除输出/system/history 后的 synthesis 安全预算，
                # 默认最多 12000 tokens，让逐源事实、正文证据和摘要能同时保留。
                max_tokens=budget.synthesis_budget,
            )

    return DocumentCollectionSummaryTool()


def build_document_section_tool(
    documents: Mapping[str, DocumentStructure],
    *,
    selected_section_ids: Mapping[str, tuple[str, ...]] | None = None,
    budget_manager: TokenBudgetManager | None = None,
) -> Any:
    """Read one section and descendants with explicit truncation metadata."""
    Tool = _smolagents_tool_class()
    if not documents:
        raise ValueError("documents must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    available_files = ", ".join(documents)

    class DocumentSectionTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_read_section"
        description = (
            "Read evidence from one section and its child sections. Call document_outline "
            "to obtain section_id. Use section_id='__all__' only for whole-document analysis. "
            f"Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded file name.",
            },
            "section_id": {
                "type": "string",
                "description": "Section id from document_outline, or __all__.",
            },
        }
        output_type = "string"

        def forward(self, file_name: str, section_id: str) -> str:
            structure = _lookup_file_value(documents, file_name)
            allowed_sections = set((selected_section_ids or {}).get(file_name, ()))
            if section_id == "__all__":
                chunks = [
                    chunk
                    for chunk in structure.chunks
                    if not allowed_sections or chunk.section_id in allowed_sections
                ]
            else:
                section = next(
                    (item for item in structure.sections if item.id == section_id),
                    None,
                )
                if section is None:
                    raise ValueError(f"Unknown section_id: {section_id}.")
                chunks = [
                    chunk
                    for chunk in structure.chunks
                    if chunk.section_path[: len(section.section_path)] == section.section_path
                    and (not allowed_sections or chunk.section_id in allowed_sections)
                ]
            return _render_section_chunks(
                file_name,
                chunks,
                max_tokens=budget.observation_budget,
            )

    return DocumentSectionTool()


def build_document_summary_tool(
    summaries: Mapping[str, HierarchicalDocumentSummary],
    *,
    budget_manager: TokenBudgetManager | None = None,
) -> Any:
    """Expose a whole-document hierarchical reduction instead of a prefix."""
    Tool = _smolagents_tool_class()
    if not summaries:
        raise ValueError("summaries must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    available_files = ", ".join(summaries)

    class DocumentSummaryTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_summary"
        description = (
            "Read a hierarchical summary that covers every chapter of one current document. "
            "Use this for whole-document summaries, then use document_search when exact evidence "
            f"is needed. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded file name.",
            }
        }
        output_type = "string"

        def forward(self, file_name: str) -> str:
            summary = _lookup_file_value(summaries, file_name)
            return truncate_to_tokens(
                summary.document_summary,
                budget.observation_budget,
            )

    return DocumentSummaryTool()


def _smolagents_tool_class() -> Any:
    """Load the Tool base lazily so non-Agent modules remain importable without smolagents."""
    try:
        from smolagents import Tool
    except ModuleNotFoundError as exc:
        raise RuntimeError("smolagents is required to build Agent tools.") from exc
    return Tool


def _lookup_file_value(values: Mapping[str, Any], file_name: str) -> Any:
    """Resolve an exact Agent-provided file name without allowing arbitrary paths."""
    normalized_name = file_name.strip()
    if normalized_name not in values:
        available_files = ", ".join(values)
        raise ValueError(
            f"Unknown file_name: {normalized_name}. Available files: {available_files}."
        )
    return values[normalized_name]


def _leading_chunks_for_best_sections(
    chunks: Sequence[Any],
    *,
    query: str,
    section_pattern: re.Pattern[str],
    excluded_ids: set[str],
    limit: int,
) -> tuple[Any, ...]:
    """Choose leading chunks from the best-matching distinct semantic sections."""
    grouped: dict[tuple[str, ...], list[Any]] = {}
    root_chunk = min(chunks, key=lambda chunk: chunk.position) if chunks else None
    root_component = (
        root_chunk.section_path[0]
        if root_chunk is not None and root_chunk.section_path
        else ""
    )
    for chunk in chunks:
        if chunk.id in excluded_ids:
            continue
        scoped_path = (
            chunk.section_path[1:]
            if (
                len(chunk.section_path) > 1
                and chunk.section_path[0] == root_component
            )
            else chunk.section_path
        )
        section_text = " / ".join(scoped_path)
        if section_pattern.search(section_text) is None:
            continue
        grouped.setdefault(tuple(chunk.section_path), []).append(chunk)
    if not grouped:
        return ()
    ranked_sections = sorted(
        grouped.items(),
        key=lambda item: (
            max(_direct_match_score(query, chunk) for chunk in item[1]),
            -min(chunk.position for chunk in item[1]),
            " / ".join(item[0]),
        ),
        reverse=True,
    )
    return tuple(
        min(section_chunks, key=lambda chunk: chunk.position)
        for _, section_chunks in ranked_sections[:limit]
    )


def _sentence_safe_excerpt(text: str, *, max_tokens: int) -> str:
    """Avoid ending a reserved evidence excerpt in the middle of a sentence."""
    normalized = text.strip()
    truncated = truncate_to_tokens(normalized, max_tokens)
    if truncated == normalized:
        return truncated
    sentence_end = max(
        truncated.rfind(marker)
        for marker in ("。", "！", "？", ".", "!", "?")
    )
    if sentence_end >= len(truncated) // 2:
        return truncated[: sentence_end + 1].strip()
    return truncated.strip()


def _chunk_evidence_text(chunk: Any) -> str:
    """Include a semantic leaf heading when it carries the start of the evidence."""
    content = str(chunk.content).strip()
    heading = str(chunk.section_path[-1]).strip() if chunk.section_path else ""
    if not heading or heading.casefold() in content[: len(heading) + 24].casefold():
        return content
    return f"{heading} {content}".strip()


def _focused_evidence_excerpt(chunks: tuple[Any, ...], *, query: str) -> str:
    """Reserve one complete source sentence most connected to this file's own facts."""
    evidence = re.sub(
        r"\s+",
        " ",
        " ".join(_chunk_evidence_text(chunk) for chunk in chunks),
    ).strip()
    sentences = [
        sentence.strip()
        for sentence in re.findall(r"[^。！？!?]+[。！？!?]", evidence)
        if 12 <= len(sentence.strip()) <= 320
    ]
    if not sentences:
        return _sentence_safe_excerpt(evidence, max_tokens=70)
    terms = _meaningful_query_terms(query)

    def score(sentence: str) -> tuple[int, int]:
        normalized = sentence.casefold()
        relevance = sum(normalized.count(term) for term in terms)
        if "不是" in sentence and any(
            marker in sentence for marker in ("而是", "但", "相反")
        ):
            relevance += 8
        if any(marker in sentence for marker in ("因为", "所以", "因此")):
            relevance += 4
        return relevance, min(len(sentence), 180)

    return _sentence_safe_excerpt(max(sentences, key=score), max_tokens=80)


def _collection_source_payload(
    *,
    file_name: str,
    summary: HierarchicalDocumentSummary,
    structure: DocumentStructure | None,
    query: str,
) -> tuple[str, str]:
    """Build reserved source facts plus optional balanced/relevant evidence."""
    critical_parts: list[str] = []
    supplemental_parts: list[str] = []
    if structure is not None and structure.chunks:
        anchor = min(structure.chunks, key=lambda chunk: chunk.position)
        fact_lines = _extract_source_fact_lines(anchor.content)
        citation = f"{file_name} / {' / '.join(anchor.section_path)}"
        citation += _page_label(anchor.page_start, anchor.page_end)
        if fact_lines:
            rendered_facts = "\n".join(
                f"- {label}: {truncate_to_tokens(line, 80)}"
                for label, line in fact_lines
            )
            critical_parts.append(
                "SOURCE_FACTS "
                f"[verbatim excerpts; chunk_id={anchor.id}; source={citation}]:\n"
                f"{rendered_facts}"
            )

        lesson_fact_query = " ".join(
            line
            for label, line in fact_lines
            if label in {"topic_line", "topic_continuation", "scripture_line"}
        ).strip()
        source_query = lesson_fact_query or query
        explanation_pattern = re.compile(
            r"经文解释|解释和讨论|释经|exegesis|interpret",
            re.IGNORECASE,
        )
        application_pattern = re.compile(
            r"生活运用|生活应用|实际应用|反思与应用|application",
            re.IGNORECASE,
        )
        explanation_chunks = _leading_chunks_for_best_sections(
            structure.chunks,
            query=source_query,
            section_pattern=explanation_pattern,
            excluded_ids={anchor.id},
            limit=2,
        )
        evidence = explanation_chunks[0] if explanation_chunks else None
        if evidence is None:
            evidence = next(
                (
                    chunk
                    for chunk in sorted(
                        structure.chunks,
                        key=lambda chunk: (
                            -_direct_match_score(source_query, chunk),
                            chunk.position,
                        ),
                    )
                    if chunk.id != anchor.id
                ),
                anchor,
            )
        evidence_citation = f"{file_name} / {' / '.join(evidence.section_path)}"
        evidence_citation += _page_label(evidence.page_start, evidence.page_end)
        critical_parts.append(
            "CORE_INTERPRETATION_EVIDENCE "
            "[verbatim excerpt reserved for this source]:\n"
            f"{_focused_evidence_excerpt((evidence, *explanation_chunks[1:]), query=source_query)}"
        )
        supplemental_parts.append(
            "QUERY_RELEVANT_EVIDENCE "
            f"[verbatim excerpt; chunk_id={evidence.id}; source={evidence_citation}]:\n"
            f"{_sentence_safe_excerpt(_chunk_evidence_text(evidence), max_tokens=240)}"
        )
        if len(explanation_chunks) > 1:
            secondary = explanation_chunks[1]
            secondary_citation = (
                f"{file_name} / {' / '.join(secondary.section_path)}"
            )
            secondary_citation += _page_label(
                secondary.page_start,
                secondary.page_end,
            )
            critical_parts.append(
                "SECONDARY_INTERPRETATION_EVIDENCE "
                "[verbatim excerpt; "
                f"chunk_id={secondary.id}; source={secondary_citation}]:\n"
                f"{_sentence_safe_excerpt(_chunk_evidence_text(secondary), max_tokens=120)}"
            )
        application_chunks = _leading_chunks_for_best_sections(
            structure.chunks,
            query=source_query,
            section_pattern=application_pattern,
            excluded_ids={
                anchor.id,
                evidence.id,
                *(chunk.id for chunk in explanation_chunks),
            },
            limit=1,
        )
        application = application_chunks[0] if application_chunks else None
        if application is not None:
            application_citation = (
                f"{file_name} / {' / '.join(application.section_path)}"
            )
            application_citation += _page_label(
                application.page_start,
                application.page_end,
            )
            # 原因：应用证据若和长 overview 共用剩余预算，后排来源会只保留释经而丢掉应用。
            # 作用：每份有生活运用章节的课程都获得一个不可被 overview 挤掉的独立证据槽。
            critical_parts.append(
                "APPLICATION_EVIDENCE "
                "[verbatim excerpt; "
                f"chunk_id={application.id}; source={application_citation}]:\n"
                f"{_sentence_safe_excerpt(_chunk_evidence_text(application), max_tokens=110)}"
            )
    if summary.document_summary.strip():
        supplemental_parts.append(
            "BALANCED_OVERVIEW "
            "(extractive orientation; paraphrase it, do not present it as a quotation):\n"
            f"{summary.document_summary.strip()}"
        )
    return (
        "\n\n".join(part for part in critical_parts if part.strip()),
        "\n\n".join(part for part in supplemental_parts if part.strip()),
    )


def _extract_source_fact_lines(text: str) -> tuple[tuple[str, str], ...]:
    """Extract exact opening identity/topic/scripture lines without interpreting them."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("![")
    ]
    if not lines:
        return ()

    facts: list[tuple[str, str]] = [("document_heading", lines[0])]
    topic_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^(?:题目|題目|主题|主題|title)\s*[：:]", line, re.IGNORECASE)
        ),
        None,
    )
    if topic_index is not None:
        facts.append(("topic_line", lines[topic_index]))
        # 原因：系列名和本课题目在真实 DOCX 中经常被拆成相邻两行。
        # 作用：保留原行而不自行拼写题目，同时在“经文/时长”等元数据前停止。
        for continuation in lines[topic_index + 1 : topic_index + 3]:
            if re.match(
                (
                    r"^(?:经文|經文|时长|時長|时间|時間|"
                    r"scripture|passage|duration|time)\s*[：:]"
                ),
                continuation,
                re.IGNORECASE,
            ):
                break
            facts.append(("topic_continuation", continuation))

    scripture_line = next(
        (
            line
            for line in lines
            if re.match(
                r"^(?:经文|經文|scripture|passage|text)\s*[：:]",
                line,
                re.IGNORECASE,
            )
        ),
        None,
    )
    if scripture_line is not None:
        facts.append(("scripture_line", scripture_line))
    elif topic_index is None and len(lines) > 1:
        facts.append(("opening_line", lines[1]))
    return tuple(facts)


_COLLECTION_GROUNDING_RULES = """GROUNDING_RULES (mandatory):
- SOURCE_FACTS are verbatim source excerpts and are the only authority for each file's document
  heading, lesson topic, and scripture reference.
- Never infer, renumber, reconstruct, or complete a scripture reference or quotation from
  memory or from a neighboring file. If evidence is absent, say so or call document_search.
- Keep every # File block isolated. Similar adjacent lessons remain distinct; never copy one
  lesson's topic, scripture, quotation, or example into another.
- CORE_INTERPRETATION_EVIDENCE, QUERY_RELEVANT_EVIDENCE,
  SECONDARY_INTERPRETATION_EVIDENCE, and APPLICATION_EVIDENCE are verbatim excerpts selected
  inside that source; cite their file block and available chunk_id for source-specific claims.
  BALANCED_OVERVIEW is extractive orientation only and does not authorize unsupported facts or
  quotations.
- QWOPUS_EXPLICIT_RUBRIC_FOUND reports whether the selected source text contains an explicit
  grading rubric. When it is false, say that no rubric was supplied and never invent points,
  weights, totals, or professor criteria."""


def _documents_contain_explicit_rubric(
    documents: Mapping[str, DocumentStructure],
) -> bool:
    """Detect only explicit grading language in source text, never in the user prompt."""
    rubric_pattern = re.compile(
        r"\brubric\b|grading\s+criteria|marking\s+scheme|"
        r"评分标准|评分细则|评分量表|评分规则|打分标准",
        re.IGNORECASE,
    )
    return any(
        rubric_pattern.search(chunk.content)
        for structure in documents.values()
        for chunk in structure.chunks
    )


def _pack_collection_evidence(
    *,
    payloads: Mapping[str, tuple[str, str]],
    covered_sources: tuple[str, ...],
    explicit_rubric_found: bool,
    max_tokens: int,
) -> str:
    """Reserve exact facts for every source, then fairly divide supplemental evidence."""
    source_names = tuple(payloads)
    manifest = (
        "QWOPUS_SOURCE_COVERAGE="
        + json.dumps(list(covered_sources), ensure_ascii=False, separators=(",", ":"))
    )
    rubric_marker = (
        "QWOPUS_EXPLICIT_RUBRIC_FOUND="
        + str(explicit_rubric_found).lower()
    )
    headers = {source: f"# File: {source}" for source in source_names}
    base_blocks = [
        "\n".join(
            part
            for part in (
                headers[source],
                payloads[source][0].strip()
                or "[No structured source facts were extractable.]",
            )
            if part
        )
        for source in source_names
    ]
    # 原因：旧打包只为文件标题保留预算，长 summary 会把后置题目、经文或应用截掉。
    # 作用：manifest、规则、每份文件标题、SOURCE_FACTS、核心释经句和应用证据先固定，
    # 剩余预算再用于较长的 query evidence 与 overview。
    fixed_render = "\n\n".join(
        [manifest, rubric_marker, _COLLECTION_GROUNDING_RULES, *base_blocks]
    )
    fixed_tokens = estimate_tokens(fixed_render)
    evidence_sources = tuple(
        source for source in source_names if payloads[source][1].strip()
    )
    minimum_payload_tokens = 24
    required_tokens = fixed_tokens + minimum_payload_tokens * len(evidence_sources)
    if required_tokens > max_tokens:
        raise RuntimeError(
            "The selected document manifest and grounded source facts do not fit the model "
            "observation budget; "
            "select fewer documents or use a larger context window."
        )

    per_source_tokens = (
        (max_tokens - fixed_tokens) // len(evidence_sources)
        if evidence_sources
        else 0
    )
    while True:
        blocks = [
            "\n".join(
                part
                for part in (
                    base_blocks[index],
                    truncate_to_tokens(payloads[source][1], per_source_tokens)
                    if payloads[source][1].strip()
                    else "",
                )
                if part
            )
            for index, source in enumerate(source_names)
        ]
        rendered = "\n\n".join(
            [manifest, rubric_marker, _COLLECTION_GROUNDING_RULES, *blocks]
        )
        rendered_tokens = estimate_tokens(rendered)
        if rendered_tokens <= max_tokens:
            return rendered
        if per_source_tokens <= minimum_payload_tokens:
            raise RuntimeError(
                "The selected document evidence does not fit the model observation budget; "
                "select fewer documents or use a larger context window."
            )
        overflow_per_source = max(
            1,
            (rendered_tokens - max_tokens + len(evidence_sources) - 1)
            // len(evidence_sources),
        )
        per_source_tokens = max(
            minimum_payload_tokens,
            per_source_tokens - overflow_per_source,
        )


def _direct_match_score(query: str, chunk: Any) -> int:
    """Score one parsed chunk using deterministic local text matching."""
    normalized_query = query.casefold().strip()
    if not normalized_query:
        return 0
    searchable = " ".join(
        (
            chunk.source,
            " ".join(chunk.section_path),
            chunk.content,
        )
    ).casefold()
    terms = _meaningful_query_terms(normalized_query)
    phrase_score = searchable.count(normalized_query) * 8
    return phrase_score + sum(searchable.count(term) for term in terms)


def _meaningful_query_terms(query: str) -> set[str]:
    """Extract useful words and CJK n-grams while dropping common instruction noise."""
    ignored_words = {
        "about",
        "all",
        "analysis",
        "answer",
        "document",
        "documents",
        "example",
        "examples",
        "file",
        "files",
        "please",
        "report",
        "write",
        "writing",
    }
    terms = {
        word
        for word in re.findall(r"[a-z0-9_]+", query)
        if len(word) > 1 and word not in ignored_words
    }
    for run in re.findall(r"[\u3400-\u9fff]+", query):
        for width in (2, 3, 4):
            terms.update(
                run[index : index + width]
                for index in range(max(0, len(run) - width + 1))
            )
    return terms


def _page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return ""
    if page_end is not None and page_end != page_start:
        return f", pages={page_start}-{page_end}"
    return f", page={page_start}"


def _render_section_chunks(
    file_name: str,
    chunks: list[Any],
    *,
    max_tokens: int,
) -> str:
    if not chunks:
        return f"No readable chunks found in the selected section of {file_name}."
    rendered: list[str] = []
    used_tokens = 0
    for chunk in chunks:
        citation = f"{file_name} / {' / '.join(chunk.section_path)}"
        citation += _page_label(chunk.page_start, chunk.page_end)
        block = f"[{citation}]\n{chunk.content}"
        block_tokens = estimate_tokens(block)
        if rendered and used_tokens + block_tokens > max_tokens:
            break
        rendered.append(
            block if block_tokens <= max_tokens else truncate_to_tokens(block, max_tokens)
        )
        used_tokens += min(block_tokens, max_tokens)
    remaining = len(chunks) - len(rendered)
    return (
        "\n\n".join(rendered)
        + f"\n\n[section_read truncated={str(remaining > 0).lower()} "
        f"remaining_chunks={remaining}]"
    )
