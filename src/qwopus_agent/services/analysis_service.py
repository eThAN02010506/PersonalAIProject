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
    run_smolagents_document_analysis_with_debug,
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
    debug_steps: list[str] = []
    analyzed_results: list[tuple[str, AnalysisResult]] = []

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
        result = analyze_uploaded_file(stored.path, user_question=user_question)
        logger.info(
            "upload_analyzed filename=%s metadata=%s",
            stored.original_name,
            result.metadata,
        )
        debug_steps.append(f"本地解析完成：{stored.original_name}: {result.metadata}")
        analyzed_results.append((stored.original_name, result))

    result = combine_analysis_results(analyzed_results)
    memory_context = ""
    if result.markdown_document:
        # 原因：上传后的 Markdown/Excel 安全摘要需要进入统一知识层。
        # 作用：后续分析可以通过 MiniRAG.search(query) 复用已上传内容。
        minirag.insert(result.markdown_document)
        logger.info(
            "minirag_inserted file_count=%s context_length=%s",
            len(analyzed_results),
            len(result.markdown_document),
        )
        debug_steps.append("MiniRAG 入库完成：已插入当前文件的 Markdown/安全摘要。")
        if user_question.strip():
            memory_results = minirag.search(user_question)
            memory_context = format_memory_context(memory_results)
            logger.info(
                "minirag_search query_length=%s hits=%s",
                len(user_question),
                len(memory_results),
            )
            debug_steps.append(f"MiniRAG 检索完成：命中 {len(memory_results)} 条。")

    if user_question.strip() and result.markdown_document:
        online, connection_message = check_model_connection(settings)
        debug_steps.append(f"模型连接检测：{connection_message}")
        if online:
            analysis_run = run_smolagents_document_analysis_with_debug(
                document_name=", ".join(file_name for file_name, _ in analyzed_results),
                content=merge_analysis_context(result.markdown_document, memory_context),
                user_question=user_question,
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
                metadata=result.metadata,
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


def format_memory_context(memory_results: list[str], max_chars: int = 4000) -> str:
    """Build bounded MiniRAG context for LLM analysis."""
    if not memory_results:
        return ""

    sections: list[str] = []
    remaining = max_chars
    for index, document in enumerate(memory_results, start=1):
        if remaining <= 0:
            break
        snippet = document[:remaining]
        # 原因：MiniRAG 可能返回长文档，不能无界加入 LLM 上下文。
        # 作用：只附加有限检索片段，让回答能利用知识层但不爆上下文。
        sections.append(f"### MiniRAG Result {index}\n\n{snippet}")
        remaining -= len(snippet)
    return "\n\n".join(sections)


def merge_analysis_context(document_context: str, memory_context: str) -> str:
    """Merge current file context with MiniRAG search context."""
    if not memory_context:
        return document_context
    return (
        f"{document_context}\n\n"
        "## MiniRAG Search Context\n\n"
        f"{memory_context}"
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
