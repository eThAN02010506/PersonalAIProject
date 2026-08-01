"""Upload analysis service.

This module owns the file-analysis business flow so UI layers only collect inputs and render
outputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from qwopus_agent.analysis import AnalysisResult, analyze_uploaded_file
from qwopus_agent.documents import (
    DocumentStore,
    DocumentStructure,
    HierarchicalDocumentSummary,
    StoredDocumentNotFoundError,
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
    should_use_grounded_report_composer,
)
from qwopus_agent.integrations.smolagents_tools import (
    build_direct_document_search_tool,
    build_document_collection_summary_tool,
    build_document_outline_tool,
    build_document_search_tool,
    build_document_section_tool,
    build_document_summary_tool,
    build_excel_analysis_tool,
    build_excel_modeling_tool,
    build_excel_schema_tool,
    build_excel_statistics_tool,
    build_minirag_search_tool,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.utils.conversation_log import append_conversation_event
from qwopus_agent.utils.logging_config import get_logger
from qwopus_agent.utils.token_budget import TokenBudgetManager

logger = get_logger("services.analysis_service")


@dataclass(frozen=True)
class UploadedFileInput:
    """Uploaded bytes or one validated local path, independent from UI frameworks."""

    name: str

    content: bytes | None = None

    local_path: Path | None = None

    def __post_init__(self) -> None:
        # 原因：同时提供字节和路径会让保存、索引语义不明确；两者都没有则无法分析。
        # 作用：所有入口在业务流程开始前都固定为上传模式或本地目录模式之一。
        if (self.content is None) == (self.local_path is None):
            raise ValueError("Provide exactly one of content or local_path.")


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
    minirag: MiniRAG | None,
    min_source_relevance: float = 0.55,
    selected_sections: dict[str, tuple[str, ...]] | None = None,
    document_store: DocumentStore | None = None,
    analysis_mode: str = "question",
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
) -> UploadAnalysisOutcome:
    """Analyze uploaded files, update MiniRAG, and optionally call the LLM."""
    if not uploaded_files:
        raise ValueError("At least one file is required.")
    has_local_paths = any(
        uploaded_file.local_path is not None for uploaded_file in uploaded_files
    )
    has_uploaded_bytes = any(
        uploaded_file.content is not None for uploaded_file in uploaded_files
    )
    if has_local_paths and has_uploaded_bytes:
        raise ValueError("Uploaded bytes and local paths cannot be mixed.")
    direct_mode = has_local_paths
    if not direct_mode and minirag is None:
        raise ValueError("Uploaded-file analysis requires a MiniRAG instance.")

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
    document_ids_by_file: dict[str, str] = {}
    saved_documents: list[dict[str, str]] = []
    debug_runs: list[AgentDebugRun] = []
    resolved_store = document_store or (None if direct_mode else DocumentStore())

    for uploaded_file in uploaded_files:
        content_hash: str | None = None
        cached_result: AnalysisResult | None = None
        if uploaded_file.local_path is not None:
            source_path = uploaded_file.local_path.expanduser().resolve(strict=True)
            file_name = uploaded_file.name
            file_size = source_path.stat().st_size
            logger.info(
                "local_file_selected filename=%s path=%s size=%s",
                file_name,
                source_path,
                file_size,
            )
            debug_steps.append(f"已选择本地文件：{file_name}")
        else:
            content = uploaded_file.content
            if content is None:
                raise ValueError(f"Missing uploaded bytes: {uploaded_file.name}")
            content_hash = hashlib.sha256(content).hexdigest()
            cached_document_id = _document_id_for_content(content_hash)
            logger.info(
                "upload_received filename=%s size=%s",
                uploaded_file.name,
                len(content),
            )
            file_name = Path(uploaded_file.name).name
            if resolved_store is not None:
                cached_result, cached_path = _load_cached_upload_analysis(
                    resolved_store,
                    cached_document_id,
                    file_name=file_name,
                    content_hash=content_hash,
                )
                if cached_result is not None and cached_path is not None:
                    source_path = cached_path
                    debug_steps.append(
                        f"命中上传缓存：{file_name} ({content_hash[:12]})"
                    )
                    logger.info(
                        "upload_cache_hit filename=%s document_id=%s",
                        file_name,
                        cached_document_id,
                    )
                else:
                    stored = save_uploaded_bytes(uploaded_file.name, content)
                    source_path = stored.path
                    file_name = stored.original_name
                    logger.info("upload_saved filename=%s path=%s", file_name, source_path)
                    debug_steps.extend(
                        [
                            f"文件已保存：{file_name}",
                            f"保存路径：{source_path}",
                        ]
                    )
            else:
                stored = save_uploaded_bytes(uploaded_file.name, content)
                source_path = stored.path
                file_name = stored.original_name
                logger.info("upload_saved filename=%s path=%s", file_name, source_path)
                debug_steps.extend(
                    [
                        f"文件已保存：{file_name}",
                        f"保存路径：{source_path}",
                    ]
                )
            document_ids_by_file[file_name] = cached_document_id

        # 原因：文件解析是确定性的输入预处理，不应该再启动另一套 Planner/Executor。
        # 作用：只生成 UI、MiniRAG 和 Tool 共用的安全上下文；Agent 决策留给 smolagents。
        result = cached_result or analyze_uploaded_file(
            source_path,
            user_question=effective_question,
            source_name=file_name,
        )
        if content_hash is not None and cached_result is None:
            result = replace(
                result,
                metadata=_metadata_with_content_hash(result.metadata, content_hash),
            )
        if result.markdown_document:
            if result.metadata.get("source_type") == "spreadsheet":
                spreadsheet_contexts[file_name] = result.markdown_document
                spreadsheet_paths[file_name] = source_path
                persistence_structure = chunk_document_structure(
                    build_document_structure(
                        result.markdown_document,
                        source=file_name,
                        document_id=document_ids_by_file.get(file_name),
                        infer_plaintext_headings=False,
                    )
                )
                persistence_summary = summarize_document(persistence_structure)
                document_ids_by_file[file_name] = persistence_structure.document_id
                if not direct_mode and source_path.exists() and resolved_store is not None:
                    resolved_store.persist(
                        original_path=source_path,
                        markdown=result.markdown_document,
                        structure=persistence_structure,
                        metadata=result.metadata,
                    )
                    resolved_store.persist_summary(persistence_summary)
                    saved_documents.append(
                        {
                            "document_id": persistence_structure.document_id,
                            "source": file_name,
                        }
                    )
            else:
                structure = (
                    result.document_structures[0]
                    if (
                        result.document_structures
                        and document_ids_by_file.get(file_name) is None
                    )
                    else chunk_document_structure(
                        build_document_structure(
                            result.markdown_document,
                            source=file_name,
                            document_id=document_ids_by_file.get(file_name),
                        )
                    )
                )
                document_ids_by_file[file_name] = structure.document_id
                document_structures[file_name] = structure
                summary = summarize_document(structure)
                document_summaries[file_name] = summary
                if not direct_mode and source_path.exists() and resolved_store is not None:
                    resolved_store.persist(
                        original_path=source_path,
                        markdown=result.markdown_document,
                        structure=structure,
                        metadata=result.metadata,
                    )
                    resolved_store.persist_summary(summary)
                    saved_documents.append(
                        {
                            "document_id": structure.document_id,
                            "source": file_name,
                        }
                    )
        logger.info(
            "upload_analyzed filename=%s metadata=%s",
            file_name,
            result.metadata,
        )
        debug_steps.append(f"本地预处理完成：{file_name}: {result.metadata}")
        analyzed_results.append((file_name, result))

    result = combine_analysis_results(analyzed_results)
    # 原因：文件持久化发生在服务层，而账号归属只能由已认证的 API 边界决定。
    # 作用：只返回稳定 document_id/source 清单，路由据此写 ACL，不传递本地绝对路径。
    result.metadata["saved_documents"] = saved_documents
    scoped_sections = _scope_sections_by_file(
        document_structures,
        selected_sections or {},
    )
    if analysis_mode == "section" and not scoped_sections:
        raise ValueError("Section analysis requires at least one valid selected section.")
    if direct_mode:
        # 原因：目录分析针对用户本次勾选的原文件，不需要制造持久知识副本或向量索引。
        # 作用：只在内存中使用解析结构和本地 Tool，storage/minirag 不会被读取或写入。
        result.metadata["minirag_inserted"] = False
        result.metadata["minirag_search_hits"] = 0
        result.metadata["analysis_source"] = "local_folder"
        memory_hit_count = 0
        debug_steps.append("本地目录直接分析：未读取或写入 MiniRAG。")
    else:
        if minirag is None:
            raise ValueError("Uploaded-file analysis requires a MiniRAG instance.")
        memory_hit_count = _index_uploaded_results(
            result=result,
            analyzed_results=analyzed_results,
            document_structures=document_structures,
            question=effective_question,
            minirag=minirag,
            min_source_relevance=min_source_relevance,
            debug_steps=debug_steps,
            document_ids_by_file=document_ids_by_file,
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
        response_detail=response_detail,
        settings=settings,
        minirag=minirag,
        min_source_relevance=min_source_relevance,
        memory_hit_count=memory_hit_count,
        direct_mode=direct_mode,
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
    document_ids_by_file: dict[str, str],
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
                else document_ids_by_file.get(file_name)
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


def _document_id_for_content(content_hash: str) -> str:
    """Return the stable upload artifact id derived from file bytes."""
    # 原因：上传文件名每次会带 uuid，不能作为“是否已经分析过”的判断依据。
    # 作用：同一份二进制内容复用同一个文档记录和 MiniRAG document_id。
    return f"document-{content_hash[:32]}"


def _metadata_with_content_hash(
    metadata: dict[str, Any],
    content_hash: str | None,
) -> dict[str, Any]:
    """Attach the upload content hash when the request came from uploaded bytes."""
    if content_hash is None:
        return dict(metadata)
    return {**metadata, "content_sha256": content_hash, "cache_hit": False}


def _load_cached_upload_analysis(
    document_store: DocumentStore,
    document_id: str,
    *,
    file_name: str,
    content_hash: str,
) -> tuple[AnalysisResult | None, Path | None]:
    """Load deterministic parse artifacts for a previously uploaded identical file."""
    try:
        stored = document_store.load_document(document_id)
    except StoredDocumentNotFoundError:
        return None, None

    metadata = {
        **stored.metadata,
        "content_sha256": content_hash,
        "cache_hit": True,
    }
    source_type = metadata.get("source_type")
    if source_type == "spreadsheet":
        # 原因：Excel 统计 Tool 需要原始表格路径，而最终回答只需要安全 Markdown 上下文。
        # 作用：跳过重复 schema/sample 解析，同时仍可对缓存原文件执行本地 pandas 计算。
        return (
            AnalysisResult(
                markdown_summary=stored.normalized_markdown,
                metadata=metadata,
                markdown_document=stored.normalized_markdown,
            ),
            stored.original_path,
        )

    structure = chunk_document_structure(
        build_document_structure(
            stored.normalized_markdown,
            source=file_name,
            document_id=document_id,
        )
    )
    return (
        AnalysisResult(
            markdown_summary=stored.normalized_markdown,
            metadata=metadata,
            markdown_document=stored.normalized_markdown,
            document_structures=(structure,),
        ),
        stored.original_path,
    )


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
    response_detail: Literal["concise", "balanced", "detailed"],
    settings: SmolagentsModelSettings,
    minirag: MiniRAG | None,
    min_source_relevance: float,
    memory_hit_count: int,
    direct_mode: bool,
    debug_steps: list[str],
) -> tuple[AnalysisResult, tuple[AgentDebugRun, ...]]:
    """Run the model phase after local parsing and indexing are complete."""
    if not question:
        debug_steps.append("未输入分析问题，因此没有调用模型生成最终答案。")
        return result, ()
    if not result.markdown_document:
        debug_steps.append("本地解析没有得到 Markdown 文档内容，因此没有调用模型。")
        return result, ()

    file_names = [file_name for file_name, _ in analyzed_results]
    grounded_composer_available = should_use_grounded_report_composer(
        file_names=file_names,
        spreadsheet_names=list(spreadsheet_contexts),
        user_question=question,
        has_collection_summary=(
            analysis_mode != "section" and len(document_summaries) > 1
        ),
    )
    online, connection_message = check_model_connection(settings)
    debug_steps.append(f"模型连接检测：{connection_message}")
    if not online and not grounded_composer_available:
        debug_steps.append(f"模型未连接，仅展示本地解析结果：{connection_message}")
        return result, ()
    if not online:
        debug_steps.append(
            "模型未连接；此全来源长报告改用本地证据合成，不依赖远程生成。"
        )

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
        direct_mode=direct_mode,
        question=question,
        analysis_mode=analysis_mode,
    )
    # 原因：smolagents 是文件分析的统一驱动，服务层只注入当前任务允许访问的能力。
    # 作用：Agent 自行选择文档、Excel 沙箱或 MiniRAG Tool，且无法访问未上传的路径。
    analysis_run = run_smolagents_file_analysis_with_debug(
        file_names=file_names,
        spreadsheet_names=list(spreadsheet_contexts),
        user_question=question,
        tools=analysis_tools,
        settings=settings,
        analysis_mode=analysis_mode,
        # 原因：文档页和聊天页必须对 Detailed 使用同一语义，否则 UI 选择会静默失效。
        # 作用：文件 Agent 收到用户本轮的详略偏好，而不是始终使用隐式默认值。
        response_detail=response_detail,
        spreadsheet_paths=spreadsheet_paths,
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
    required_sources = tuple(file_names)
    covered_sources = tuple(
        file_name
        for file_name in required_sources
        if file_name in analysis_run.inspected_file_names
    )
    missing_sources = tuple(
        file_name for file_name in required_sources if file_name not in covered_sources
    )
    if missing_sources:
        # 原因：API 的 completed 状态必须代表所有显式选择的来源都实际进入 Agent 证据。
        # 作用：即使未来 runtime 被替换或 mocked，也不能把少源答案包装成成功结果。
        raise RuntimeError(
            "Document source coverage incomplete: "
            + ", ".join(missing_sources)
            + "."
        )
    return (
        AnalysisResult(
            markdown_summary=result.markdown_summary,
            tables=result.tables,
            metadata={
                **result.metadata,
                "minirag_search_hits": memory_hit_count,
                "minirag_context_used": (
                    minirag is not None and "rag_search" in analysis_run.tool_calls
                ),
                "pandas_sandbox_used": "excel_analysis" in analysis_run.tool_calls,
                "statistics_skill_used": "excel_statistics" in analysis_run.tool_calls,
                "modeling_skill_used": "excel_modeling" in analysis_run.tool_calls,
                "local_spreadsheet_computation_used": bool(
                    {"excel_statistics", "excel_modeling", "excel_analysis"}.intersection(
                        analysis_run.tool_calls
                    )
                ),
                "smolagents_tool_calls": analysis_run.tool_calls,
                "generation_mode": analysis_run.generation_mode,
                "source_coverage": {
                    "required_sources": list(required_sources),
                    "covered_sources": list(covered_sources),
                    "missing_sources": list(missing_sources),
                    "complete": not missing_sources,
                },
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
    minirag: MiniRAG | None,
    document_structures: dict[str, DocumentStructure],
    document_summaries: dict[str, HierarchicalDocumentSummary],
    spreadsheet_contexts: dict[str, str],
    spreadsheet_paths: dict[str, Path],
    scoped_sections: dict[str, tuple[str, ...]],
    min_source_relevance: float,
    budget_manager: TokenBudgetManager,
    direct_mode: bool,
    question: str,
    analysis_mode: str,
) -> list[Any]:
    """Compose task-scoped Tool adapters without adding business decisions."""
    tools: list[Any] = []
    if document_structures:
        if direct_mode:
            search_tool = build_direct_document_search_tool(
                document_structures,
                selected_section_ids=scoped_sections,
                budget_manager=budget_manager,
            )
        else:
            if minirag is None:
                raise ValueError("Indexed document search requires MiniRAG.")
            search_tool = build_document_search_tool(
                minirag,
                document_structures,
                min_relevance=min_source_relevance,
                selected_section_ids=scoped_sections,
                budget_manager=budget_manager,
            )
        tools.extend(
            [
                build_document_outline_tool(
                    document_structures,
                    budget_manager=budget_manager,
                ),
                search_tool,
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
        if analysis_mode != "section" and len(document_summaries) > 1:
            tools.append(
                build_document_collection_summary_tool(
                    document_summaries,
                    documents=document_structures,
                    query=question,
                    budget_manager=budget_manager,
                )
            )
    if spreadsheet_contexts:
        tools.extend(
            [
                build_excel_schema_tool(
                    spreadsheet_contexts,
                    budget_manager=budget_manager,
                ),
                build_excel_statistics_tool(spreadsheet_paths),
                build_excel_modeling_tool(spreadsheet_paths),
                build_excel_analysis_tool(spreadsheet_paths),
            ]
        )
    if minirag is not None and not direct_mode:
        tools.append(
            build_minirag_search_tool(
                minirag,
                min_relevance=min_source_relevance,
                budget_manager=budget_manager,
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
