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
from qwopus_agent.documents import save_uploaded_bytes
from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
    run_smolagents_file_analysis_with_debug,
)
from qwopus_agent.integrations.smolagents_tools import (
    build_document_parser_tool,
    build_excel_analysis_tool,
    build_excel_schema_tool,
    build_minirag_search_tool,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.utils.conversation_log import append_conversation_event
from qwopus_agent.utils.logging_config import get_logger

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


def analyze_uploaded_files(
    uploaded_files: list[UploadedFileInput],
    user_question: str,
    settings: SmolagentsModelSettings,
    minirag: MiniRAG,
) -> UploadAnalysisOutcome:
    """Analyze uploaded files, update MiniRAG, and optionally call the LLM."""
    # 原因：服务端模型可能在两次上传分析之间发生切换。
    # 作用：每次分析开始时刷新模型 id，避免继续使用 .env 中的旧名称。
    settings = resolve_model_settings(settings)
    debug_steps: list[str] = []
    analyzed_results: list[tuple[str, AnalysisResult]] = []
    document_contexts: dict[str, str] = {}
    spreadsheet_contexts: dict[str, str] = {}
    spreadsheet_paths: dict[str, Path] = {}

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
        result = analyze_uploaded_file(stored.path, user_question=user_question)
        if result.markdown_document:
            if result.metadata.get("source_type") == "spreadsheet":
                spreadsheet_contexts[stored.original_name] = result.markdown_document
                spreadsheet_paths[stored.original_name] = stored.path
            else:
                document_contexts[stored.original_name] = result.markdown_document
        logger.info(
            "upload_analyzed filename=%s metadata=%s",
            stored.original_name,
            result.metadata,
        )
        debug_steps.append(f"本地预处理完成：{stored.original_name}: {result.metadata}")
        analyzed_results.append((stored.original_name, result))

    result = combine_analysis_results(analyzed_results)
    memory_hit_count = 0
    if result.markdown_document:
        result.metadata["minirag_inserted"] = False
        result.metadata["minirag_search_hits"] = 0
        if user_question.strip():
            # 原因：MiniRAG 应该补充“已有知识”，当前上传文件已经在 document_context 里。
            # 作用：先检索旧知识再入库，避免把当前文件重复算作知识库命中。
            memory_results = minirag.search(user_question)
            memory_hit_count = len(memory_results)
            result.metadata["minirag_search_hits"] = memory_hit_count
            logger.info(
                "minirag_search query_length=%s hits=%s",
                len(user_question),
                len(memory_results),
            )
            debug_steps.append(f"MiniRAG 检索完成：命中 {len(memory_results)} 条已有知识。")

        # 原因：合并多文件后再入库会让单个文件无法安全更新或删除。
        # 作用：每份解析结果按原文件名独立入库，当前 LLM 上下文仍保留合并视图。
        for file_name, analysis_result in analyzed_results:
            if analysis_result.markdown_document:
                minirag.insert(
                    f"# File: {file_name}\n\n{analysis_result.markdown_document}"
                )
        # 原因：第八步要求上传内容进入知识层，但主界面不能暴露原始检索内容。
        # 作用：只把入库状态和命中数写入 metadata，供 UI 展示轻量状态。
        result.metadata["minirag_inserted"] = True
        logger.info(
            "minirag_inserted file_count=%s context_length=%s",
            len(analyzed_results),
            len(result.markdown_document),
        )
        debug_steps.append("MiniRAG 入库完成：已插入当前文件的 Markdown/安全摘要。")

    if user_question.strip() and result.markdown_document:
        online, connection_message = check_model_connection(settings)
        debug_steps.append(f"模型连接检测：{connection_message}")
        if online:
            analysis_tools: list[Any] = []
            if document_contexts:
                analysis_tools.append(build_document_parser_tool(document_contexts))
            if spreadsheet_contexts:
                analysis_tools.extend(
                    [
                        build_excel_schema_tool(spreadsheet_contexts),
                        build_excel_analysis_tool(spreadsheet_paths),
                    ]
                )
            analysis_tools.append(build_minirag_search_tool(minirag))
            # 原因：smolagents 是文件分析的统一驱动，服务层只注入当前任务允许访问的能力。
            # 作用：Agent 自行选择文档、Excel 沙箱或 MiniRAG Tool，且无法访问未上传的路径。
            analysis_run = run_smolagents_file_analysis_with_debug(
                file_names=[file_name for file_name, _ in analyzed_results],
                spreadsheet_names=list(spreadsheet_contexts),
                user_question=user_question,
                tools=analysis_tools,
                settings=settings,
            )
            logger.info(
                "analysis_llm_completed files=%s answer_length=%s",
                [file_name for file_name, _ in analyzed_results],
                len(analysis_run.answer),
            )
            debug_steps.extend(analysis_run.debug_steps)
            result = AnalysisResult(
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
            )
            append_conversation_event(
                "analysis",
                {
                    "files": [file_name for file_name, _ in analyzed_results],
                    "question": user_question,
                    "answer": analysis_run.answer,
                },
            )
        else:
            debug_steps.append(f"模型未连接，仅展示本地解析结果：{connection_message}")
    elif not user_question.strip():
        debug_steps.append("未输入分析问题，因此没有调用模型生成最终答案。")
    elif not result.markdown_document:
        debug_steps.append("本地解析没有得到 Markdown 文档内容，因此没有调用模型。")

    logger.info("analysis_completed file_count=%s", len(analyzed_results))
    return UploadAnalysisOutcome(
        result=result,
        debug_steps=debug_steps,
        analyzed_file_names=[file_name for file_name, _ in analyzed_results],
    )


def combine_analysis_results(
    results: list[tuple[str, AnalysisResult]],
) -> AnalysisResult:
    """Combine multiple uploaded-file analysis results."""
    markdown_sections: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    metadata_files: list[dict[str, Any]] = []
    document_sections: list[str] = []

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
    )
