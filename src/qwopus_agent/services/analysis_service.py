"""Upload analysis service.

This module owns the file-analysis business flow so UI layers only collect inputs and render
outputs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qwopus_agent.agents import AgentRouter, Executor, Planner
from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.documents import save_uploaded_bytes
from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
    run_smolagents_document_analysis_with_debug,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.skills import SkillRegistry
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


@dataclass(frozen=True)
class FileAgentAnalysis:
    """One saved file analyzed through the Agent route."""

    result: AnalysisResult

    plan_steps: list[str]


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
        routed_analysis = _analyze_file_with_agent(stored.path, user_question=user_question)
        result = routed_analysis.result
        logger.info(
            "upload_analyzed filename=%s metadata=%s",
            stored.original_name,
            result.metadata,
        )
        debug_steps.append(
            f"Agent 计划执行：{stored.original_name}: "
            f"{' → '.join(routed_analysis.plan_steps) or '无可执行步骤'}"
        )
        debug_steps.append(f"本地解析完成：{stored.original_name}: {result.metadata}")
        analyzed_results.append((stored.original_name, result))

    result = combine_analysis_results(analyzed_results)
    memory_context = ""
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
            result.metadata["minirag_context_used"] = memory_hit_count > 0
            memory_context = format_memory_context(memory_results)
            logger.info(
                "minirag_search query_length=%s hits=%s",
                len(user_question),
                len(memory_results),
            )
            debug_steps.append(f"MiniRAG 检索完成：命中 {len(memory_results)} 条已有知识。")

        # 原因：上传后的 Markdown/Excel 安全摘要需要进入统一知识层。
        # 作用：后续分析可以通过 MiniRAG.search(query) 复用已上传内容。
        minirag.insert(result.markdown_document)
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
                metadata={
                    **result.metadata,
                    "minirag_search_hits": memory_hit_count,
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


def _analyze_file_with_agent(file_path: Path, user_question: str) -> FileAgentAnalysis:
    """Analyze one saved file through Planner, Executor, and Skills."""
    registry = SkillRegistry.discover()
    router = AgentRouter(
        planner=Planner(skill_registry=registry),
        executor=Executor(skill_registry=registry),
    )
    # 原因：上传分析入口必须真实经过 Agent 架构，而不是绕过 Planner/Executor 直接调用解析函数。
    # 作用：UI、CLI、API 后续都能复用同一条“规划 → 执行 → Skill”的能力链路。
    agent_run = asyncio.run(
        router.run(
            user_question or "分析上传文件",
            context={"arguments": {"file_path": str(file_path)}},
        )
    )
    if not agent_run.execution.success:
        raise RuntimeError(agent_run.execution.content)

    plan_steps = [step.skill_name for step in agent_run.plan.steps]
    last_response = agent_run.execution.steps[-1].response
    analysis_result = last_response.data.get("analysis_result")
    if isinstance(analysis_result, AnalysisResult):
        return FileAgentAnalysis(result=analysis_result, plan_steps=plan_steps)

    markdown = str(last_response.data.get("markdown") or last_response.content)
    metadata = dict(last_response.data.get("metadata", {}))
    # 原因：文档解析 Skill 的职责是返回 Markdown，不负责生成完整报告对象。
    # 作用：服务层把 Skill 输出包装成 AnalysisResult，后续 MiniRAG 和 LLM 总结逻辑无需改动。
    return FileAgentAnalysis(
        result=AnalysisResult(
            markdown_summary=markdown,
            tables={},
            metadata=metadata,
            markdown_document=markdown,
        ),
        plan_steps=plan_steps,
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
