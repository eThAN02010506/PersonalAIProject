"""Inventory, attachment, and explicit analysis routes for saved documents."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException

from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    AnalysisView,
    SavedDocumentsAnalysisRequest,
    SavedDocumentsAttachRequest,
    SavedDocumentsAttachView,
    SavedDocumentView,
)
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.routes.analysis import analysis_view
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
    def list_documents() -> list[SavedDocumentView]:
        return [
            SavedDocumentView.model_validate(document, from_attributes=True)
            for document in document_store.list_documents()
        ]

    @router.post(
        "/api/conversations/{conversation_id}/documents/attach",
        response_model=SavedDocumentsAttachView,
    )
    def attach_documents(
        conversation_id: str,
        request: SavedDocumentsAttachRequest,
    ) -> SavedDocumentsAttachView:
        if repository.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        selected = _load_selected_documents(document_store, request.document_ids)

        try:
            with knowledge.lease(conversation_id) as minirag:
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
        request: SavedDocumentsAnalysisRequest,
    ) -> AnalysisView:
        if repository.get_conversation(request.conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        selected = _load_selected_documents(document_store, request.document_ids)
        orchestration_request = OrchestrationRequest(
            objective=request.question,
            conversation_id=request.conversation_id,
            uploaded_files=tuple(
                OrchestrationFile(
                    name=saved.document.source,
                    # 原因：已保存记录已经持有经过验证的规范化全文，重复解析原 PDF/DOCX
                    # 会再次启动 MinerU，既慢又可能产生与首次持久化不同的结果。
                    # 作用：复用 confined normalized.md；name 仍保留原始 source，
                    # 因此文档结构、覆盖统计和引用继续显示用户看到的文件名。
                    local_path=saved.normalized_path,
                )
                for saved in selected
            ),
            generate_report=request.generate_report,
            min_source_relevance=request.min_source_relevance,
            analysis_mode=request.analysis_mode,
            selected_sections=request.selected_sections,
            report_title="Qwopus Saved Documents Analysis",
            report_basename="qwopus_saved_documents_analysis",
        )
        orchestrator = AgentOrchestrator(runtime.current_settings(), minirag=None)
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
            directory=debug_directory,
        )
        if not result.success:
            raise HTTPException(status_code=500, detail=result.final_answer)
        return analysis_view(result)

    return router


def _load_selected_documents(
    document_store: DocumentStore,
    document_ids: list[str],
) -> list[StoredDocumentContent]:
    """Preflight every selected record before attachment or model execution."""
    selected: list[StoredDocumentContent] = []
    for document_id in document_ids:
        try:
            selected.append(document_store.load_document(document_id))
        except InvalidDocumentIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
