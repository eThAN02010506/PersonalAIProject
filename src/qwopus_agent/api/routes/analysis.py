"""Document analysis upload route."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    AnalysisView,
    DocumentOutlineView,
    DocumentSectionView,
)
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import (
    OrchestrationFile,
    OrchestrationRequest,
    OrchestrationResult,
)
from qwopus_agent.utils.debug_store import append_debug_record

if TYPE_CHECKING:
    from qwopus_agent.memory import MiniRAG

MemoryProvider = Callable[[], "MiniRAG"]


def build_analysis_router(
    runtime: RuntimeModelController,
    get_minirag: MemoryProvider,
    debug_directory: Path,
) -> APIRouter:
    """Build the upload boundary around the unified Agent orchestrator."""
    router = APIRouter()

    @router.post("/api/analysis", response_model=AnalysisView)
    async def analyze(
        files: Annotated[list[UploadFile], File()],
        question: Annotated[str, Form()] = "",
        generate_report: Annotated[bool, Form()] = False,
        min_source_relevance: Annotated[float, Form(ge=0.25, le=0.95)] = 0.55,
        analysis_mode: Annotated[
            Literal["question", "section", "full"],
            Form(),
        ] = "question",
        selected_sections: Annotated[str, Form()] = "{}",
    ) -> AnalysisView:
        try:
            scoped_sections = _parse_selected_sections(selected_sections)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid selected sections.") from exc

        # 原因：生成式中包含 await 时会产生 async_generator，不能直接交给 tuple()。
        # 作用：逐个读取上传字节，确保 MinerU/MiniRAG 编排收到完整且有序的文件集合。
        uploads = tuple(
            [
                OrchestrationFile(
                    name=file.filename or "upload",
                    content=await file.read(),
                )
                for file in files
            ]
        )
        if not uploads:
            raise HTTPException(status_code=400, detail="At least one file is required.")

        request = OrchestrationRequest(
            objective=question,
            uploaded_files=uploads,
            generate_report=generate_report,
            min_source_relevance=min_source_relevance,
            analysis_mode=analysis_mode,
            selected_sections=scoped_sections,
            report_title="Qwopus Analysis Report",
            report_basename="qwopus_web_analysis",
        )
        orchestrator = AgentOrchestrator(
            runtime.current_settings(),
            minirag=get_minirag(),
        )
        result = await asyncio.to_thread(orchestrator.run_sync, request)
        # 原因：文档分析不经过 ChatRunRegistry 的完成回调。
        # 作用：把内部步骤写给只读 Console，同时不把 Tool Observation 暴露给正式前端。
        append_debug_record(
            source="document",
            status="completed" if result.success else "failed",
            result=result.final_answer,
            trace=result.trace,
            debug_runs=result.debug_runs,
            directory=debug_directory,
        )
        if not result.success:
            raise HTTPException(status_code=500, detail=result.final_answer)
        return _analysis_view(result)

    return router


def _parse_selected_sections(payload: str) -> dict[str, tuple[str, ...]]:
    """Validate the JSON boundary before section ids enter document tools."""
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise TypeError("selected sections must be an object")
    parsed: dict[str, tuple[str, ...]] = {}
    for document_id, section_ids in raw.items():
        if (
            not isinstance(document_id, str)
            or not document_id
            or not isinstance(section_ids, list)
            or any(not isinstance(section_id, str) or not section_id for section_id in section_ids)
        ):
            # 原因：字符串也是可迭代对象，宽松转换会把一个 section id 拆成单个字符。
            # 作用：API 仅接受 {document_id: [section_id, ...]}，拒绝形状错误的数据。
            raise TypeError("selected sections contain invalid ids")
        parsed[document_id] = tuple(section_ids)
    return parsed


def _analysis_view(result: OrchestrationResult) -> AnalysisView:
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
    return AnalysisView(
        answer=result.final_answer,
        route=result.route,
        citations=[item.model_dump(mode="json") for item in result.citations],
        trace=[item.model_dump(mode="json") for item in result.trace],
        reports=reports,
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
