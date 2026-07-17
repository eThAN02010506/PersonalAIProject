"""smolagents Tool adapters for Qwopus-Agent capabilities."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.analysis.pandas_sandbox import execute_pandas_code
from qwopus_agent.memory import MiniRAG


@dataclass
class TavilySearchConfig:
    """Runtime configuration for Tavily search."""

    api_key: str | None = None

    endpoint: str = "https://api.tavily.com/search"

    max_results: int = 5

    timeout_seconds: int = 20


def build_tavily_search_tool(
    config: TavilySearchConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    """Build a smolagents Tool that searches Tavily."""
    Tool = _smolagents_tool_class()

    resolved_config = config or TavilySearchConfig()
    completed_queries: set[str] = set()

    class TavilySearchTool(Tool):  # type: ignore[misc, valid-type]
        name = "tavily_search"
        description = (
            "Search the live web with Tavily. Use this when current external "
            "information is needed. "
            "Input is a search query. Output is concise markdown search evidence."
        )
        inputs = {"query": {"type": "string", "description": "The web search query."}}
        output_type = "string"

        def forward(self, query: str) -> str:
            normalized_query = " ".join(query.split()).casefold()
            if normalized_query in completed_queries:
                # 原因：部分模型会在下一步原样重复同一个 Tool Call，浪费配额和上下文。
                # 作用：相同查询只访问 Tavily 一次，并明确要求 Agent 使用已有证据收尾。
                return (
                    "This exact Tavily query was already completed. Use the previous "
                    "Observation and call final_answer now; do not search it again."
                )

            api_key = _resolve_tavily_api_key(resolved_config.api_key)
            if not api_key:
                raise RuntimeError("TAVILY_API_KEY is not configured.")
            if progress_callback is not None:
                progress_callback("searching")

            payload = json.dumps(
                {
                    "query": query,
                    "search_depth": "basic",
                    "max_results": resolved_config.max_results,
                    "topic": "general",
                    "include_answer": True,
                    "include_raw_content": False,
                    "include_images": False,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                resolved_config.endpoint,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Qwopus-Agent/0.1",
                },
                method="POST",
            )
            with urllib.request.urlopen(
                request,
                timeout=resolved_config.timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))

            # 原因：Tavily 是搜索服务，smolagents 是调度入口；Tool 只返回证据文本。
            # 作用：让 Agent 基于搜索证据生成最终回答，而不是把原始 JSON 暴露给 UI。
            formatted_result = _format_tavily_results(
                data,
                max_results=resolved_config.max_results,
            )
            completed_queries.add(normalized_query)
            if progress_callback is not None:
                progress_callback("generating")
            return formatted_result

    return TavilySearchTool()


def build_document_parser_tool(
    documents: Mapping[str, str],
    max_chars: int = 16000,
) -> Any:
    """Expose locally parsed Markdown to the smolagents runtime."""
    Tool = _smolagents_tool_class()
    parsed_documents = {
        str(file_name): str(markdown)
        for file_name, markdown in documents.items()
        if str(markdown).strip()
    }
    if not parsed_documents:
        raise ValueError("documents must contain at least one parsed document.")
    available_files = ", ".join(parsed_documents)

    class DocumentParserTool(Tool):  # type: ignore[misc, valid-type]
        name = "document_parser"
        description = (
            "Read the locally parsed Markdown for one current uploaded document. "
            f"Use the exact file_name. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded file name to read.",
            }
        }
        output_type = "string"

        def forward(self, file_name: str) -> str:
            markdown = _lookup_file_value(parsed_documents, file_name)
            # 原因：完整长文档会无界占用 Agent 上下文，甚至让模型停在 Observation 阶段。
            # 作用：Tool 只返回有上限的 Markdown；原文件仍完整保存在本地和 MiniRAG 中。
            return _bounded_text(
                markdown,
                max_chars=max_chars,
                truncation_message="[Document content truncated by the tool.]",
            )

    return DocumentParserTool()


def build_excel_schema_tool(
    spreadsheet_contexts: Mapping[str, str],
    max_chars: int = 10000,
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
            context = _lookup_file_value(contexts, file_name)
            # 原因：LLM 只需要 schema、样本和本地统计来设计分析代码。
            # 作用：严格阻止整份 Excel 数据通过 Tool 进入模型上下文。
            return _bounded_text(
                context,
                max_chars=max_chars,
                truncation_message="[Spreadsheet schema context truncated by the tool.]",
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
    max_results: int = 3,
    max_chars: int = 6000,
) -> Any:
    """Expose MiniRAG.search(query) as a bounded smolagents Tool."""
    Tool = _smolagents_tool_class()

    class MiniRAGSearchTool(Tool):  # type: ignore[misc, valid-type]
        name = "rag_search"
        description = (
            "Search previously indexed local documents through MiniRAG. "
            "Use this only when prior uploaded knowledge may help answer the question."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "Semantic search query for the local knowledge base.",
            }
        }
        output_type = "string"

        def forward(self, query: str) -> str:
            results = minirag.search(query)[:max_results]
            if not results:
                return "No relevant MiniRAG results."
            sections = [
                f"## MiniRAG Result {index}\n\n{document}"
                for index, document in enumerate(results, start=1)
            ]
            # 原因：知识库可能含有多份长文档，检索 Tool 不能把所有原文灌入一次推理。
            # 作用：限制结果数量和总长度，同时保持 MiniRAG 对外仍只有 search(query)。
            return _bounded_text(
                "\n\n".join(sections),
                max_chars=max_chars,
                truncation_message="[MiniRAG results truncated by the tool.]",
            )

    return MiniRAGSearchTool()


def _format_tavily_results(payload: dict[str, Any], max_results: int) -> str:
    """Format Tavily response JSON as bounded markdown evidence."""
    sections: list[str] = []
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        sections.append(f"## Tavily Answer\n\n{answer.strip()}")

    result_lines: list[str] = []
    for item in payload.get("results", [])[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title and not content:
            continue
        heading = f"### {title}" if title else "### Search Result"
        if url:
            heading = f"{heading}\n{url}"
        result_lines.append(f"{heading}\n\n{content}".strip())

    if result_lines:
        sections.append("## Tavily Search Results\n\n" + "\n\n".join(result_lines))
    return "\n\n".join(sections) if sections else "No Tavily search results."


def _resolve_tavily_api_key(explicit_api_key: str | None) -> str:
    """Resolve Tavily credentials without changing the tracked .env file."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    # 原因：项目的 .env 已被 Git 跟踪，联网密钥不能继续写入该文件。
    # 作用：优先读取被 Git 忽略的本地配置，同时保留进程环境变量部署方式。
    local_api_key = dotenv_values(".env.local").get("TAVILY_API_KEY")
    if isinstance(local_api_key, str) and local_api_key.strip():
        return local_api_key.strip()
    return (os.getenv("TAVILY_API_KEY") or "").strip()


def _smolagents_tool_class() -> Any:
    """Load the Tool base lazily so non-Agent modules remain importable without smolagents."""
    try:
        from smolagents import Tool  # type: ignore[import-untyped]
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


def _bounded_text(text: str, max_chars: int, truncation_message: str) -> str:
    """Bound Tool observations before they enter the Agent context."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n{truncation_message}"
