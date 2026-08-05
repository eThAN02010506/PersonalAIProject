"""Inventory, attachment, and explicit analysis routes for saved documents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from qwopus_agent.api.auth import current_user
from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    AnalysisView,
    SavedDocumentsAnalysisRequest,
    SavedDocumentsAttachRequest,
    SavedDocumentsAttachView,
    SavedDocumentView,
)
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.routes.analysis import (
    analysis_view,
    register_analysis_access,
    resolve_analysis_objective,
)
from qwopus_agent.documents import (
    CorruptStoredDocumentError,
    DocumentStore,
    InvalidDocumentIdError,
    StoredDocumentContent,
    StoredDocumentNotFoundError,
)
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import (
    OrchestrationFile,
    OrchestrationRequest,
    OrchestrationResult,
)
from qwopus_agent.utils.debug_store import append_debug_record


def build_document_router(
    document_store: DocumentStore,
    repository: ConversationRepository,
    knowledge: ConversationKnowledgeManager,
    runtime: RuntimeModelController,
    debug_directory: Path,
) -> APIRouter:
    """Expose persisted documents only through explicit, validated selections."""
    router = APIRouter()

    @router.get("/api/documents", response_model=list[SavedDocumentView])
    def list_documents(request: Request) -> list[SavedDocumentView]:
        user = current_user(request)
        allowed_ids = repository.accessible_document_ids(user.id)
        return [
            SavedDocumentView.model_validate(document, from_attributes=True)
            for document in document_store.list_documents()
            if document.document_id in allowed_ids
        ]

    @router.post(
        "/api/conversations/{conversation_id}/documents/attach",
        response_model=SavedDocumentsAttachView,
    )
    def attach_documents(
        conversation_id: str,
        payload: SavedDocumentsAttachRequest,
        request: Request,
    ) -> SavedDocumentsAttachView:
        user = current_user(request)
        if repository.get_conversation_for_user(conversation_id, user.id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        selected = _load_selected_documents(
            document_store,
            payload.document_ids,
            allowed_ids=repository.accessible_document_ids(user.id),
        )

        try:
            with knowledge.lease(conversation_id, global_scope=user.id) as minirag:
                for saved in selected:
                    # 原因：Saved documents 过去只是全局清单，当前聊天并没有这些证据。
                    # 作用：用户明确勾选后，把规范化全文按稳定 document_id 写入该会话私库。
                    minirag.insert(
                        (
                            f"# File: {saved.document.source}\n\n"
                            f"{saved.normalized_markdown}"
                        ),
                        document_id=saved.document.document_id,
                    )
                    repository.link_document_to_conversation(
                        saved.document.document_id,
                        conversation_id=conversation_id,
                        attached_by_user_id=user.id,
                    )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail="Conversation knowledge is no longer available.",
            ) from exc

        return SavedDocumentsAttachView(
            conversation_id=conversation_id,
            attached_count=len(selected),
            documents=[
                SavedDocumentView.model_validate(
                    saved.document,
                    from_attributes=True,
                )
                for saved in selected
            ],
        )

    @router.post("/api/documents/analyze", response_model=AnalysisView)
    async def analyze_documents(
        payload: SavedDocumentsAnalysisRequest,
        request: Request,
    ) -> AnalysisView:
        user = current_user(request)
        if repository.get_conversation_for_user(payload.conversation_id, user.id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        objective = resolve_analysis_objective(
            payload.question,
            payload.analysis_mode,
            payload.selected_sections,
        )
        selected = _load_selected_documents(
            document_store,
            payload.document_ids,
            allowed_ids=repository.accessible_document_ids(user.id),
        )
        if payload.analysis_mode == "section":
            _validate_saved_section_scope(selected, payload.selected_sections)
        orchestration_request = OrchestrationRequest(
            objective=objective,
            conversation_id=payload.conversation_id,
            uploaded_files=tuple(
                OrchestrationFile(
                    name=saved.document.source,
                    # 原因：PDF/DOCX 的规范化全文可直接复用，但表格分析仍需要原工作簿。
                    # 作用：普通文档避免重复启动 MinerU，Excel/CSV 则保留真实表结构。
                    local_path=(
                        saved.original_path
                        if saved.document.file_type in {"csv", "xlsx", "xls"}
                        else saved.normalized_path
                    ),
                )
                for saved in selected
            ),
            generate_report=payload.generate_report,
            min_source_relevance=payload.min_source_relevance,
            response_detail=payload.response_detail,
            analysis_mode=payload.analysis_mode,
            recipe=payload.recipe,
            selected_sections=payload.selected_sections,
            report_title="Qwopus Saved Documents Analysis",
            report_basename=f"qwopus_saved_documents_analysis_{uuid4().hex[:12]}",
        )
        orchestrator = AgentOrchestrator(
            runtime.current_settings(),
            minirag=None,
            document_store=document_store,
        )
        result: OrchestrationResult = await asyncio.to_thread(
            orchestrator.run_sync,
            orchestration_request,
        )
        append_debug_record(
            source="saved_documents",
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
            conversation_id=payload.conversation_id,
            user_id=user.id,
        )
        if not result.success:
            raise HTTPException(status_code=500, detail=result.final_answer)
        return analysis_view(result)

    return router


def _load_selected_documents(
    document_store: DocumentStore,
    document_ids: list[str],
    *,
    allowed_ids: set[str] | None = None,
) -> list[StoredDocumentContent]:
    """Preflight every selected record before attachment or model execution."""
    selected: list[StoredDocumentContent] = []
    for document_id in document_ids:
        try:
            document_store.validate_document_id(document_id)
        except InvalidDocumentIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if allowed_ids is not None and document_id not in allowed_ids:
            # 原因：物理文件可能存在，但确认它会向另一个账号泄漏 document_id。
            # 作用：无权访问与不存在使用同一 404 语义，并在读磁盘前拒绝。
            raise HTTPException(
                status_code=404,
                detail=f"Saved document not found: {document_id}",
            )
        try:
            selected.append(document_store.load_document(document_id))
        except StoredDocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CorruptStoredDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    normalized_sources = [saved.document.source.casefold() for saved in selected]
    if len(normalized_sources) != len(set(normalized_sources)):
        # 原因：MiniRAG 用 source 识别文件版本，两个同名版本会互相替换。
        # 作用：拒绝含糊选择，避免响应声称覆盖了实际已被覆盖的文档。
        raise HTTPException(
            status_code=422,
            detail="Selected saved documents must have unique source names.",
        )
    return selected


def _validate_saved_section_scope(
    selected: list[StoredDocumentContent],
    selected_sections: dict[str, tuple[str, ...]],
) -> None:
    """Reject section ids that do not belong to the selected saved documents."""
    available = {
        saved.structure.document_id: {section.id for section in saved.structure.sections}
        for saved in selected
    }
    for document_id, section_ids in selected_sections.items():
        if not section_ids:
            continue
        if document_id not in available or any(
            section_id not in available[document_id] for section_id in section_ids
        ):
            # 原因：旧分析结果中的章节可能不属于本次勾选的 Saved Documents。
            # 作用：在创建 Agent 前拒绝越界或过期章节，避免 Planner 处理无效任务范围。
            raise HTTPException(
                status_code=422,
                detail="Selected sections must belong to the selected saved documents.",
            )
