"""Upload analysis service.

This module owns the file-analysis business flow so UI layers only collect inputs and render
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qwopus_agent.analysis import AnalysisResult, analyze_uploaded_file
from qwopus_agent.documents import (
    DocumentStore,
    DocumentStructure,
    HierarchicalDocumentSummary,
    build_document_structure,
    chunk_document_structure,
    save_uploaded_bytes,
    summarize_document,
)
from qwopus_agent.integrations.smolagents_runtime import (
    AgentDebugRun,
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
    run_smolagents_file_analysis_with_debug,
)
from qwopus_agent.integrations.smolagents_tools import (
    build_document_outline_tool,
    build_document_search_tool,
    build_document_section_tool,
    build_document_summary_tool,
    build_excel_analysis_tool,
    build_excel_schema_tool,
    build_minirag_search_tool,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.utils.conversation_log import append_conversation_event
from qwopus_agent.utils.logging_config import get_logger
from qwopus_agent.utils.token_budget import TokenBudgetManager

logger = get_logger("services.analysis_service")


@dataclass(frozen=True)
class UploadedFileInput:
    """Uploaded file payload independent from Streamlit."""

    name: str

    content: bytes


@dataclass(frozen=True)
class UploadAnalysisOutcome:
    """Result returned to UI after analyzing uploaded files."""

    result: AnalysisResult

    debug_steps: list[str]

    analyzed_file_names: list[str] = field(default_factory=list)

    debug_runs: tuple[AgentDebugRun, ...] = ()


def analyze_uploaded_files(
    uploaded_files: list[UploadedFileInput],
    user_question: str,
    settings: SmolagentsModelSettings,
    minirag: MiniRAG,
    min_source_relevance: float = 0.55,
    selected_sections: dict[str, tuple[str, ...]] | None = None,
    document_store: DocumentStore | None = None,
    analysis_mode: str = "question",
) -> UploadAnalysisOutcome:
    """Analyze uploaded files, update MiniRAG, and optionally call the LLM."""
    # 原因：服务端模型可能在两次上传分析之间发生切换。
    # 作用：每次分析开始时刷新模型 id，避免继续使用 .env 中的旧名称。
    settings = resolve_model_settings(settings)
    effective_question = user_question.strip()
    if not effective_question and analysis_mode == "full":
        effective_question = "Summarize the complete uploaded document."
    elif not effective_question and analysis_mode == "section":
        effective_question = "Summarize the selected document sections."
    debug_steps: list[str] = []
    analyzed_results: list[tuple[str, AnalysisResult]] = []
    document_structures: dict[str, DocumentStructure] = {}
    document_summaries: dict[str, HierarchicalDocumentSummary] = {}
    spreadsheet_contexts: dict[str, str] = {}
    spreadsheet_paths: dict[str, Path] = {}
    debug_runs: list[AgentDebugRun] = []

    for uploaded_file in uploaded_files:
        logger.info(
            "upload_received filename=%s size=%s",
            uploaded_file.name,
            len(uploaded_file.content),
        )
        stored = save_uploaded_bytes(uploaded_file.name, uploaded_file.content)
        logger.info("upload_saved filename=%s path=%s", stored.original_name, stored.path)
        debug_steps.extend(
            [
                f"文件已保存：{stored.original_name}",
                f"保存路径：{stored.path}",
            ]
        )
        # 原因：文件解析是确定性的输入预处理，不应该再启动另一套 Planner/Executor。
        # 作用：只生成 UI、MiniRAG 和 Tool 共用的安全上下文；Agent 决策留给 smolagents。
        result = analyze_uploaded_file(
            stored.path,
            user_question=effective_question,
            source_name=stored.original_name,
        )
        if result.markdown_document:
            if result.metadata.get("source_type") == "spreadsheet":
                spreadsheet_contexts[stored.original_name] = result.markdown_document
                spreadsheet_paths[stored.original_name] = stored.path
            else:
                structure = (
                    result.document_structures[0]
                    if result.document_structures
                    else chunk_document_structure(
                        build_document_structure(
                            result.markdown_document,
                            source=stored.original_name,
                        )
                    )
                )
                document_structures[stored.original_name] = structure
                summary = summarize_document(structure)
                document_summaries[stored.original_name] = summary
                if stored.path.exists():
                    resolved_store = document_store or DocumentStore()
                    resolved_store.persist(
                        original_path=stored.path,
                        markdown=result.markdown_document,
                        structure=structure,
                        metadata=result.metadata,
                    )
                    resolved_store.persist_summary(summary)
        logger.info(
            "upload_analyzed filename=%s metadata=%s",
            stored.original_name,
            result.metadata,
        )
        debug_steps.append(f"本地预处理完成：{stored.original_name}: {result.metadata}")
        analyzed_results.append((stored.original_name, result))

    result = combine_analysis_results(analyzed_results)
    scoped_sections = _scope_sections_by_file(
        document_structures,
        selected_sections or {},
    )
    if analysis_mode == "section" and not scoped_sections:
        raise ValueError("Section analysis requires at least one valid selected section.")
    memory_hit_count = _index_uploaded_results(
        result=result,
        analyzed_results=analyzed_results,
        document_structures=document_structures,
        question=effective_question,
        minirag=minirag,
        min_source_relevance=min_source_relevance,
        debug_steps=debug_steps,
    )
    result, model_debug_runs = _run_model_analysis(
        result=result,
        analyzed_results=analyzed_results,
        document_structures=document_structures,
        document_summaries=document_summaries,
        spreadsheet_contexts=spreadsheet_contexts,
        spreadsheet_paths=spreadsheet_paths,
        scoped_sections=scoped_sections,
        question=effective_question,
        analysis_mode=analysis_mode,
        settings=settings,
        minirag=minirag,
        min_source_relevance=min_source_relevance,
        memory_hit_count=memory_hit_count,
        debug_steps=debug_steps,
    )
    debug_runs.extend(model_debug_runs)

    logger.info("analysis_completed file_count=%s", len(analyzed_results))
    return UploadAnalysisOutcome(
        result=result,
        debug_steps=debug_steps,
        analyzed_file_names=[file_name for file_name, _ in analyzed_results],
        debug_runs=tuple(debug_runs),
    )


def _index_uploaded_results(
    *,
    result: AnalysisResult,
    analyzed_results: list[tuple[str, AnalysisResult]],
    document_structures: dict[str, DocumentStructure],
    question: str,
    minirag: MiniRAG,
    min_source_relevance: float,
    debug_steps: list[str],
) -> int:
    """Search previous knowledge, then index each current file independently."""
    if not result.markdown_document:
        return 0
    result.metadata["minirag_inserted"] = False
    result.metadata["minirag_search_hits"] = 0
    memory_hit_count = 0
    if question:
        # 原因：当前文件尚未入库时检索，命中数才只代表先前已有知识。
        # 作用：避免把本次上传内容误报为历史 MiniRAG 上下文。
        memory_results = minirag.search(
            question,
            min_relevance=min_source_relevance,
        )
        memory_hit_count = len(memory_results)
        result.metadata["minirag_search_hits"] = memory_hit_count
        logger.info(
            "minirag_search query_length=%s hits=%s",
            len(question),
            memory_hit_count,
        )
        debug_steps.append(f"MiniRAG 检索完成：命中 {memory_hit_count} 条已有知识。")

    for file_name, analysis_result in analyzed_results:
        if not analysis_result.markdown_document:
            continue
        indexed_structure = document_structures.get(file_name)
        minirag.insert(
            f"# File: {file_name}\n\n{analysis_result.markdown_document}",
            document_id=(
                indexed_structure.document_id
                if indexed_structure is not None
                else None
            ),
        )
    result.metadata["minirag_inserted"] = True
    logger.info(
        "minirag_inserted file_count=%s context_length=%s",
        len(analyzed_results),
        len(result.markdown_document),
    )
    debug_steps.append("MiniRAG 入库完成：已插入当前文件的 Markdown/安全摘要。")
    return memory_hit_count


def _run_model_analysis(
    *,
    result: AnalysisResult,
    analyzed_results: list[tuple[str, AnalysisResult]],
    document_structures: dict[str, DocumentStructure],
    document_summaries: dict[str, HierarchicalDocumentSummary],
    spreadsheet_contexts: dict[str, str],
    spreadsheet_paths: dict[str, Path],
    scoped_sections: dict[str, tuple[str, ...]],
    question: str,
    analysis_mode: str,
    settings: SmolagentsModelSettings,
    minirag: MiniRAG,
    min_source_relevance: float,
    memory_hit_count: int,
    debug_steps: list[str],
) -> tuple[AnalysisResult, tuple[AgentDebugRun, ...]]:
    """Run the model phase after local parsing and indexing are complete."""
    if not question:
        debug_steps.append("未输入分析问题，因此没有调用模型生成最终答案。")
        return result, ()
    if not result.markdown_document:
        debug_steps.append("本地解析没有得到 Markdown 文档内容，因此没有调用模型。")
        return result, ()

    online, connection_message = check_model_connection(settings)
    debug_steps.append(f"模型连接检测：{connection_message}")
    if not online:
        debug_steps.append(f"模型未连接，仅展示本地解析结果：{connection_message}")
        return result, ()

    budget_manager = TokenBudgetManager(
        context_window=settings.context_window_tokens,
        output_reserve=settings.max_tokens,
    )
    analysis_tools = _build_analysis_tools(
        minirag=minirag,
        document_structures=document_structures,
        document_summaries=document_summaries,
        spreadsheet_contexts=spreadsheet_contexts,
        spreadsheet_paths=spreadsheet_paths,
        scoped_sections=scoped_sections,
        min_source_relevance=min_source_relevance,
        budget_manager=budget_manager,
    )
    # 原因：smolagents 是文件分析的统一驱动，服务层只注入当前任务允许访问的能力。
    # 作用：Agent 自行选择文档、Excel 沙箱或 MiniRAG Tool，且无法访问未上传的路径。
    file_names = [file_name for file_name, _ in analyzed_results]
    analysis_run = run_smolagents_file_analysis_with_debug(
        file_names=file_names,
        spreadsheet_names=list(spreadsheet_contexts),
        user_question=question,
        tools=analysis_tools,
        settings=settings,
        analysis_mode=analysis_mode,
    )
    logger.info(
        "analysis_llm_completed files=%s answer_length=%s",
        file_names,
        len(analysis_run.answer),
    )
    debug_steps.extend(analysis_run.debug_steps)
    append_conversation_event(
        "analysis",
        {
            "files": file_names,
            "question": question,
            "answer": analysis_run.answer,
        },
    )
    return (
        AnalysisResult(
            markdown_summary=result.markdown_summary,
            tables=result.tables,
            metadata={
                **result.metadata,
                "minirag_search_hits": memory_hit_count,
                "minirag_context_used": "rag_search" in analysis_run.tool_calls,
                "pandas_sandbox_used": "excel_analysis" in analysis_run.tool_calls,
                "smolagents_tool_calls": analysis_run.tool_calls,
            },
            markdown_document=result.markdown_document,
            llm_analysis=analysis_run.answer,
            document_structures=result.document_structures,
        ),
        analysis_run.debug_runs,
    )


def _scope_sections_by_file(
    structures: dict[str, DocumentStructure],
    selected_sections: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Resolve API document ids to file names and include selected descendants."""
    known_document_keys = {
        key
        for file_name, structure in structures.items()
        for key in (file_name, structure.document_id)
    }
    unknown_document_keys = set(selected_sections) - known_document_keys
    if unknown_document_keys:
        raise ValueError(
            f"Unknown document selection: {', '.join(sorted(unknown_document_keys))}"
        )
    scoped: dict[str, tuple[str, ...]] = {}
    for file_name, structure in structures.items():
        requested = set(
            selected_sections.get(structure.document_id, selected_sections.get(file_name, ()))
        )
        if not requested:
            continue
        known_section_ids = {section.id for section in structure.sections}
        unknown_section_ids = requested - known_section_ids
        if unknown_section_ids:
            # 原因：空的允许列表在 Tool 层表示“不限制”，未知 id 不能静默升级为全文权限。
            # 作用：陈旧或恶意章节选择在读取正文前失败，避免范围绕过。
            raise ValueError(
                f"Unknown section selection: {', '.join(sorted(unknown_section_ids))}"
            )
        selected_paths = {
            section.section_path
            for section in structure.sections
            if section.id in requested
        }
        # 原因：选择父标题在用户语义上包含其子标题，精确 id 过滤会意外丢失子章节。
        # 作用：在进入 MiniRAG/Section Tool 前展开后代，底层检索仍只处理稳定 section id。
        scoped[file_name] = tuple(
            section.id
            for section in structure.sections
            if any(
                section.section_path[: len(parent_path)] == parent_path
                for parent_path in selected_paths
            )
        )
    return scoped


def _build_analysis_tools(
    *,
    minirag: MiniRAG,
    document_structures: dict[str, DocumentStructure],
    document_summaries: dict[str, HierarchicalDocumentSummary],
    spreadsheet_contexts: dict[str, str],
    spreadsheet_paths: dict[str, Path],
    scoped_sections: dict[str, tuple[str, ...]],
    min_source_relevance: float,
    budget_manager: TokenBudgetManager,
) -> list[Any]:
    """Compose task-scoped Tool adapters without adding business decisions."""
    tools: list[Any] = []
    if document_structures:
        tools.extend(
            [
                build_document_outline_tool(document_structures),
                build_document_search_tool(
                    minirag,
                    document_structures,
                    min_relevance=min_source_relevance,
                    selected_section_ids=scoped_sections,
                    budget_manager=budget_manager,
                ),
                build_document_section_tool(
                    document_structures,
                    selected_section_ids=scoped_sections,
                    budget_manager=budget_manager,
                ),
                build_document_summary_tool(
                    document_summaries,
                    budget_manager=budget_manager,
                ),
            ]
        )
    if spreadsheet_contexts:
        tools.extend(
            [
                build_excel_schema_tool(spreadsheet_contexts),
                build_excel_analysis_tool(spreadsheet_paths),
            ]
        )
    tools.append(
        build_minirag_search_tool(
            minirag,
            min_relevance=min_source_relevance,
        )
    )
    return tools


def combine_analysis_results(
    results: list[tuple[str, AnalysisResult]],
) -> AnalysisResult:
    """Combine multiple uploaded-file analysis results."""
    markdown_sections: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    metadata_files: list[dict[str, Any]] = []
    document_sections: list[str] = []
    document_structures: list[DocumentStructure] = []

    for file_name, result in results:
        markdown_sections.append(f"## File: {file_name}\n\n{result.markdown_summary}")
        metadata_files.append(
            {
                "file_name": file_name,
                "metadata": result.metadata,
            }
        )
        if result.markdown_document:
            # 原因：多个上传文件需要合成一个 LLM 上下文，但仍要保留来源。
            # 作用：用文件名分隔每个 Markdown/Excel 安全摘要。
            document_sections.append(f"# File: {file_name}\n\n{result.markdown_document}")
        document_structures.extend(result.document_structures)
        for table_name, dataframe in result.tables.items():
            safe_file_name = Path(file_name).stem or "file"
            tables[f"{safe_file_name}::{table_name}"] = dataframe

    return AnalysisResult(
        markdown_summary="\n\n".join(markdown_sections),
        tables=tables,
        metadata={
            "source_type": "multi_upload",
            "file_count": len(results),
            "files": metadata_files,
        },
        markdown_document="\n\n".join(document_sections),
        document_structures=tuple(document_structures),
    )
