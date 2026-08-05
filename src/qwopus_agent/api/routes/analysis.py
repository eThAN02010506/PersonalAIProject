"""Document analysis upload route."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from qwopus_agent.api.auth import current_user
from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    AnalysisView,
    DocumentOutlineView,
    DocumentSectionView,
    SourceCoverageView,
    SpreadsheetSheetView,
    SpreadsheetTableView,
    SpreadsheetWorkbookView,
)
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.documents import DocumentStore
from qwopus_agent.documents.parser import SUPPORTED_DOCUMENT_EXTENSIONS
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import (
    OrchestrationFile,
    OrchestrationRequest,
    OrchestrationResult,
)
from qwopus_agent.utils.debug_store import append_debug_record

MAX_UPLOAD_FILES = 20
MAX_UPLOAD_FILE_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 256 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS | {".csv", ".xlsx", ".xls"}
FULL_ANALYSIS_OBJECTIVE = (
    "Analyze and summarize every selected document, including its central arguments, "
    "evidence, relationships, and limitations."
)
SECTION_ANALYSIS_OBJECTIVE = (
    "Analyze and summarize the selected document sections, including their central "
    "arguments, evidence, relationships, and limitations."
)


def resolve_analysis_objective(
    question: str,
    analysis_mode: Literal["question", "section", "full"],
    selected_sections: dict[str, tuple[str, ...]],
) -> str:
    """Resolve one mode-aware objective before constructing an orchestration request."""
    # 原因：question 必须由用户定义目标，但 full/section 允许使用确定性的默认目标。
    # 作用：在读取文档或启动 Agent 前统一模式语义，保证编排请求永远收到明确任务。
    objective = question.strip()
    if analysis_mode == "question":
        if not objective:
            raise HTTPException(
                status_code=422,
                detail="Enter a question for question-based analysis.",
            )
        return objective
    if analysis_mode == "full":
        return objective or FULL_ANALYSIS_OBJECTIVE

    has_selected_section = False
    for document_id, section_ids in selected_sections.items():
        if not document_id.strip() or any(not section_id.strip() for section_id in section_ids):
            raise HTTPException(status_code=422, detail="Invalid selected sections.")
        has_selected_section = has_selected_section or bool(section_ids)
    if not has_selected_section:
        raise HTTPException(
            status_code=422,
            detail="Select at least one document section for section analysis.",
        )
    return objective or SECTION_ANALYSIS_OBJECTIVE


def build_analysis_router(
    runtime: RuntimeModelController,
    repository: ConversationRepository,
    knowledge: ConversationKnowledgeManager,
    debug_directory: Path,
    document_store: DocumentStore,
) -> APIRouter:
    """Build the upload boundary around the unified Agent orchestrator."""
    router = APIRouter()

    @router.post("/api/analysis", response_model=AnalysisView)
    async def analyze(
        request: Request,
        files: Annotated[list[UploadFile], File()],
        conversation_id: Annotated[str, Form(min_length=1)],
        question: Annotated[str, Form()] = "",
        generate_report: Annotated[bool, Form()] = False,
        min_source_relevance: Annotated[float, Form(ge=0.25, le=0.95)] = 0.55,
        response_detail: Annotated[
            Literal["concise", "balanced", "detailed"],
            Form(),
        ] = "detailed",
        analysis_mode: Annotated[
            Literal["question", "section", "full"],
            Form(),
        ] = "question",
        recipe: Annotated[Literal["generic", "bible"], Form()] = "generic",
        selected_sections: Annotated[str, Form()] = "{}",
    ) -> AnalysisView:
        user = current_user(request)
        if repository.get_conversation_for_user(conversation_id, user.id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        try:
            scoped_sections = _parse_selected_sections(selected_sections)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid selected sections.") from exc
        objective = resolve_analysis_objective(
            question,
            analysis_mode,
            scoped_sections,
        )

        uploads = await _read_uploads(files)

        orchestration_request = OrchestrationRequest(
            objective=objective,
            conversation_id=conversation_id,
            uploaded_files=uploads,
            generate_report=generate_report,
            min_source_relevance=min_source_relevance,
            response_detail=response_detail,
            analysis_mode=analysis_mode,
            recipe=recipe,
            selected_sections=scoped_sections,
            report_title="Qwopus Analysis Report",
            report_basename=f"qwopus_web_analysis_{uuid4().hex[:12]}",
        )

        def run_analysis() -> OrchestrationResult:
            # 原因：同一聊天的两个并发上传会同时改写 documents、向量和图谱派生文件。
            # 作用：只串行化当前 conversation_id；其他聊天仍可独立并行分析。
            with knowledge.lease(conversation_id, global_scope=user.id) as minirag:
                orchestrator = AgentOrchestrator(
                    runtime.current_settings(),
                    minirag=minirag,
                    document_store=document_store,
                )
                return orchestrator.run_sync(orchestration_request)

        result = await asyncio.to_thread(run_analysis)
        # 原因：文档分析不经过 ChatRunRegistry 的完成回调。
        # 作用：把内部步骤写给只读 Console，同时不把 Tool Observation 暴露给正式前端。
        append_debug_record(
            source="document",
            status="completed" if result.success else "failed",
            result=result.final_answer,
            trace=result.trace,
            debug_runs=result.debug_runs,
            user_id=user.id,
            username=user.username,
            directory=debug_directory,
        )
        register_analysis_access(
            result,
            repository=repository,
            conversation_id=conversation_id,
            user_id=user.id,
        )
        if not result.success:
            raise HTTPException(status_code=500, detail=result.final_answer)
        return analysis_view(result)

    return router


def register_analysis_access(
    result: OrchestrationResult,
    *,
    repository: ConversationRepository,
    conversation_id: str,
    user_id: str,
) -> None:
    """Bind newly persisted files and generated reports to one authenticated request."""
    metadata = result.analysis_result.metadata if result.analysis_result is not None else {}
    saved_documents = metadata.get("saved_documents")
    if isinstance(saved_documents, list):
        for item in saved_documents:
            if not isinstance(item, dict):
                continue
            document_id = item.get("document_id")
            if isinstance(document_id, str) and document_id:
                repository.register_document(
                    document_id,
                    owner_user_id=user_id,
                    conversation_id=conversation_id,
                )
    if result.report is not None:
        for artifact in result.report.artifacts:
            repository.register_report(
                artifact.path.name,
                created_by_user_id=user_id,
                conversation_id=conversation_id,
            )


async def _read_uploads(files: list[UploadFile]) -> tuple[OrchestrationFile, ...]:
    """Read supported uploads with bounded per-file and aggregate memory use."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload at most {MAX_UPLOAD_FILES} files per analysis.",
        )

    uploads: list[OrchestrationFile] = []
    total_bytes = 0
    for file in files:
        name = Path(file.filename or "").name
        if not name or Path(name).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported upload type: {name or 'unnamed file'}",
            )

        content = bytearray()
        while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
            content.extend(chunk)
            total_bytes += len(chunk)
            if len(content) > MAX_UPLOAD_FILE_BYTES:
                # 原因：UploadFile 会落盘，但转换成 OrchestrationFile 时仍会进入进程内存。
                # 作用：在 MinerU、pandas 或 MiniRAG 解析前限制单个文档的内存占用。
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MiB: {name}",
                )
            if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                # 原因：多个合法大小文件仍可能合计耗尽本地 Agent 进程内存。
                # 作用：为一次分析设置确定的总内存上界，不依赖客户端 Content-Length。
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Combined uploads exceed "
                        f"{MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MiB."
                    ),
                )
        uploads.append(OrchestrationFile(name=name, content=bytes(content)))
    return tuple(uploads)


def _parse_selected_sections(payload: str) -> dict[str, tuple[str, ...]]:
    """Validate the JSON boundary before section ids enter document tools."""
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise TypeError("selected sections must be an object")
    parsed: dict[str, tuple[str, ...]] = {}
    for document_id, section_ids in raw.items():
        if (
            not isinstance(document_id, str)
            or not document_id.strip()
            or not isinstance(section_ids, list)
            or any(
                not isinstance(section_id, str) or not section_id.strip()
                for section_id in section_ids
            )
        ):
            # 原因：字符串也是可迭代对象，宽松转换会把一个 section id 拆成单个字符。
            # 作用：API 仅接受 {document_id: [section_id, ...]}，拒绝形状错误的数据。
            raise TypeError("selected sections contain invalid ids")
        parsed[document_id.strip()] = tuple(section_id.strip() for section_id in section_ids)
    return parsed


def analysis_view(result: OrchestrationResult) -> AnalysisView:
    """Map internal analysis artifacts onto the public response contract."""
    reports = []
    if result.report is not None:
        reports = [
            {
                "kind": artifact.kind,
                "name": artifact.path.name,
                "url": f"/api/reports/{artifact.path.name}",
            }
            for artifact in result.report.artifacts
        ]
    source_coverage = (
        result.analysis_result.metadata.get("source_coverage")
        if result.analysis_result is not None
        else None
    )
    generation_mode = (
        result.analysis_result.metadata.get("generation_mode")
        if result.analysis_result is not None
        else None
    )
    return AnalysisView(
        answer=result.final_answer,
        route=result.route,
        citations=[item.model_dump(mode="json") for item in result.citations],
        trace=[item.model_dump(mode="json") for item in result.trace],
        reports=reports,
        spreadsheets=_spreadsheet_views(result),
        source_coverage=(
            SourceCoverageView.model_validate(source_coverage)
            if isinstance(source_coverage, dict)
            else None
        ),
        generation_mode=(
            str(generation_mode)
            if isinstance(generation_mode, str)
            else None
        ),
        documents=[
            DocumentOutlineView(
                document_id=structure.document_id,
                source=structure.source,
                sections=[
                    DocumentSectionView(
                        id=section.id,
                        title=section.title,
                        level=section.level,
                        parent_id=section.parent_id,
                        section_path=list(section.section_path),
                        page_start=section.page_start,
                        page_end=section.page_end,
                    )
                    for section in structure.sections
                ],
            )
            for structure in (
                result.analysis_result.document_structures
                if result.analysis_result is not None
                else ()
            )
        ],
    )


def _spreadsheet_views(
    result: OrchestrationResult,
) -> list[SpreadsheetWorkbookView]:
    """Extract only bounded workbook structure from internal analysis metadata."""
    if result.analysis_result is None:
        return []
    root_metadata = result.analysis_result.metadata
    file_records = root_metadata.get("files")
    if not isinstance(file_records, list):
        file_records = [
            {
                "file_name": "spreadsheet",
                "metadata": root_metadata,
            }
        ]

    workbooks: list[SpreadsheetWorkbookView] = []
    for record in file_records:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("source_type") != "spreadsheet":
            continue
        profile = metadata.get("workbook_profile")
        analysis_tables = metadata.get("analysis_tables")
        if not isinstance(profile, dict) or not isinstance(analysis_tables, dict):
            continue

        sheet_views = [
            SpreadsheetSheetView(
                name=str(sheet.get("name", "")),
                kind=sheet.get("kind", "matrix"),
                region_count=len(sheet.get("table_regions", ())),
                formula_count=int(sheet.get("formula_count", 0)),
                merged_range_count=int(sheet.get("merged_range_count", 0)),
                chart_count=int(sheet.get("chart_count", 0)),
                image_count=int(sheet.get("image_count", 0)),
                data_validation_count=int(sheet.get("data_validation_count", 0)),
                hidden=bool(sheet.get("hidden", False)),
            )
            for sheet in profile.get("sheets", ())
            if isinstance(sheet, dict)
        ]
        table_views: list[SpreadsheetTableView] = []
        for table_name, table in analysis_tables.items():
            if not isinstance(table, dict):
                continue
            raw_column_names = table.get("column_names", ())
            column_names = (
                [str(name)[:120] for name in raw_column_names]
                if isinstance(raw_column_names, list)
                else []
            )
            # 原因：复杂模板可能有数百列或公式式字段名，完整透传会拖慢响应并破坏布局。
            # 作用：API 只展示前 40 个 schema 名称；完整结构仍保留在本地分析结果与调试记录中。
            table_views.append(
                SpreadsheetTableView(
                    name=str(table_name),
                    source_sheet=str(table_name).split("::", maxsplit=1)[0],
                    rows=int(table.get("rows", 0)),
                    columns=int(table.get("columns", 0)),
                    column_names=column_names[:40],
                    columns_truncated=len(column_names) > 40,
                )
            )
        workbooks.append(
            SpreadsheetWorkbookView(
                source=str(record.get("file_name", profile.get("source", "spreadsheet"))),
                sheet_count=int(profile.get("sheet_count", len(sheet_views))),
                formula_count=int(profile.get("formula_count", 0)),
                merged_range_count=int(profile.get("merged_range_count", 0)),
                chart_count=int(profile.get("chart_count", 0)),
                image_count=int(profile.get("image_count", 0)),
                data_validation_count=int(profile.get("data_validation_count", 0)),
                sheets=sheet_views,
                tables=table_views,
            )
        )
    return workbooks
