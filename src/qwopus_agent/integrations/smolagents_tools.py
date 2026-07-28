"""smolagents Tool adapters for Qwopus-Agent capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.analysis.pandas_sandbox import execute_pandas_code
from qwopus_agent.documents import DocumentStructure, HierarchicalDocumentSummary
from qwopus_agent.integrations.skill_tools import build_skill_tool
from qwopus_agent.integrations.tavily import TavilySearchConfig, TavilySearchProvider
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.graph_search import GraphSearchSkill
from qwopus_agent.skills.rag_search import RagSearchSkill
from qwopus_agent.skills.web_search import WebSearchSkill
from qwopus_agent.utils.token_budget import (
    TokenBudgetManager,
    estimate_tokens,
    truncate_to_tokens,
)

if TYPE_CHECKING:
    from qwopus_agent.memory import MiniRAG
    from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex


def build_tavily_search_tool(
    config: TavilySearchConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    """Build a smolagents Tool that searches Tavily."""
    resolved_config = config or TavilySearchConfig()
    skill = WebSearchSkill(
        provider=TavilySearchProvider(
            config=resolved_config,
            progress_callback=progress_callback,
        )
    )
    # 原因：联网业务只能存在于 WebSearchSkill，Tool 不应再复制 HTTP 和格式化逻辑。
    # 作用：正式 Agent、Planner/Executor 和测试共享同一个 Tavily Provider。
    return build_skill_tool(
        skill,
        tool_name="tavily_search",
        description=(
            "Search the live web with Tavily when current external information is needed. "
            "Output is concise Markdown evidence."
        ),
        inputs={"query": {"type": "string", "description": "The web search query."}},
    )

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
    minirag: MiniRAG,
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
                return f"No relevant evidence found in {file_name}."
            # 原因：检索已经限定当前文件，仍需服从当前模型可用的证据 token。
            # 作用：只压缩 Tool Observation，不截断持久化原文和章节 Chunk。
            return truncate_to_tokens(
                "\n\n".join(results),
                budget.observation_budget,
            )

    return DocumentSearchTool()


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


def build_excel_schema_tool(
    spreadsheet_contexts: Mapping[str, str],
    *,
    budget_manager: TokenBudgetManager | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Expose spreadsheet schema, samples, and local summaries only."""
    Tool = _smolagents_tool_class()
    contexts = {
        str(file_name): str(context)
        for file_name, context in spreadsheet_contexts.items()
        if str(context).strip()
    }
    if not contexts:
        raise ValueError("spreadsheet_contexts must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    output_budget = max_tokens or budget.observation_budget
    available_files = ", ".join(contexts)

    class ExcelSchemaTool(Tool):  # type: ignore[misc, valid-type]
        name = "excel_schema"
        description = (
            "Inspect spreadsheet sheet names, columns, data types, sample rows, "
            "and local summaries. "
            "Call this before excel_analysis. It never returns the full spreadsheet. "
            f"Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            }
        }
        output_type = "string"

        def forward(self, file_name: str) -> str:
            context = str(_lookup_file_value(contexts, file_name))
            # 原因：LLM 只需要 schema、样本和本地统计来设计分析代码。
            # 作用：严格阻止整份 Excel 数据通过 Tool 进入模型上下文。
            if estimate_tokens(context) <= output_budget:
                return context
            return (
                f"{truncate_to_tokens(context, output_budget)}\n\n"
                "[Spreadsheet schema context truncated by the tool.]"
            )

    return ExcelSchemaTool()


def build_excel_analysis_tool(spreadsheets: Mapping[str, str | Path]) -> Any:
    """Execute Agent-generated pandas code in the existing local sandbox."""
    Tool = _smolagents_tool_class()
    spreadsheet_paths = {
        str(file_name): Path(file_path) for file_name, file_path in spreadsheets.items()
    }
    if not spreadsheet_paths:
        raise ValueError("spreadsheets must not be empty.")
    available_files = ", ".join(spreadsheet_paths)

    class ExcelAnalysisTool(Tool):  # type: ignore[misc, valid-type]
        name = "excel_analysis"
        description = (
            "Execute restricted pandas code against an uploaded spreadsheet locally. "
            "Call excel_schema first. The code may use only dfs and pd and must "
            "assign its final value "
            f"to result. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            },
            "code": {
                "type": "string",
                "description": "Restricted pandas code that assigns the computed answer to result.",
            },
        }
        output_type = "string"

        def forward(self, file_name: str, code: str) -> str:
            path = _lookup_file_value(spreadsheet_paths, file_name)
            if not path.exists():
                raise FileNotFoundError(f"Spreadsheet does not exist: {path}")
            spreadsheet = read_spreadsheet(path)
            # 原因：模型负责提出分析代码，但不能直接运行任意 Python 或读取本机文件。
            # 作用：在 AST 受限沙箱内针对本地 DataFrame 执行，只把计算结果返回 Agent。
            execution = execute_pandas_code(code, spreadsheet.sheets)
            return str(execution.markdown)

    return ExcelAnalysisTool()


def build_minirag_search_tool(
    minirag: MiniRAG,
    min_relevance: float = 0.55,
    max_results: int = 3,
    budget_manager: TokenBudgetManager | None = None,
    progress_callback: Callable[[str], None] | None = None,
    tool_name: str = "rag_search",
    description: str | None = None,
) -> Any:
    """Expose MiniRAG.search(query) as a bounded smolagents Tool."""
    budget = budget_manager or TokenBudgetManager()
    return build_skill_tool(
        RagSearchSkill(minirag=minirag),
        tool_name=tool_name,
        inputs={
            "query": {
                "type": "string",
                "description": "Semantic search query for the local knowledge base.",
            }
        },
        request_factory=lambda values: SkillRequest(
            query=str(values["query"]),
            arguments={
                "min_relevance": min_relevance,
                "max_results": max_results,
            },
        ),
        description=description
        or (
            "Search documents uploaded in the current conversation through MiniRAG. "
            "Use this only when this conversation's prior files may help answer the question."
        ),
        max_output_tokens=budget.observation_budget,
        progress_callback=progress_callback,
        start_phase="retrieving",
    )


def build_graph_search_tool(
    index: KnowledgeGraphIndex,
    max_hops: int = 4,
    max_results: int = 5,
    budget_manager: TokenBudgetManager | None = None,
    progress_callback: Callable[[str], None] | None = None,
    tool_name: str = "graph_search",
    description: str | None = None,
) -> Any:
    """Expose bounded persistent graph traversal as a smolagents Tool."""
    budget = budget_manager or TokenBudgetManager()
    return build_skill_tool(
        GraphSearchSkill(index=index),
        tool_name=tool_name,
        inputs={
            "query": {
                "type": "string",
                "description": "A relationship or graph-path question containing entity names.",
            }
        },
        request_factory=lambda values: SkillRequest(
            query=str(values["query"]),
            arguments={"max_hops": max_hops, "limit": max_results},
        ),
        description=description
        or (
            "Search explicit entity relationships, cross-document evidence, and multi-hop "
            "paths in the current conversation's knowledge graph. Use this instead of "
            "rag_search when the question asks how named entities are related."
        ),
        max_output_tokens=budget.observation_budget,
        progress_callback=progress_callback,
        start_phase="retrieving",
    )


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
