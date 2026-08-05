"""Local-folder discovery routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from qwopus_agent.api.auth import require_admin
from qwopus_agent.api.debug_access import require_debug_client
from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    AnalysisView,
    LocalFolderAnalysisRequest,
    LocalFolderNodeView,
    LocalFolderScanRequest,
    LocalFolderTreeView,
)
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.routes.analysis import (
    analysis_view,
    register_analysis_access,
    resolve_analysis_objective,
)
from qwopus_agent.documents.local_folder import (
    MAX_LOCAL_FOLDER_SELECTION,
    LocalFolderError,
    LocalFolderNode,
    resolve_selected_files,
    scan_local_folder,
)
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import (
    OrchestrationFile,
    OrchestrationRequest,
    OrchestrationResult,
)
from qwopus_agent.utils.debug_store import append_debug_record


def build_local_folder_router(
    runtime: RuntimeModelController,
    repository: ConversationRepository,
    debug_directory: Path,
) -> APIRouter:
    """Build the local-only folder boundary for the formal frontend."""
    router = APIRouter()

    @router.post("/api/local-folders/scan", response_model=LocalFolderTreeView)
    def scan_folder(
        payload: LocalFolderScanRequest,
        request: Request,
    ) -> LocalFolderTreeView:
        require_admin(request)
        require_debug_client(request)
        try:
            folder = scan_local_folder(payload.path)
        except LocalFolderError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # 原因：前端只需要可选择的相对路径，不能获得后端目录以外的额外文件信息。
        # 作用：把内部 Path/dataclass 映射成受 Pydantic 约束的安全树结构。
        return LocalFolderTreeView(
            root=str(folder.root),
            file_count=folder.file_count,
            # 原因：扫描上限和单次分析上限不同，前端不能复制后端约束并等待两边漂移。
            # 作用：树形选择直接使用本次 API 声明的上限，在提交前阻止无效请求。
            max_selection=MAX_LOCAL_FOLDER_SELECTION,
            tree=_node_view(folder.tree),
        )

    @router.post("/api/local-folders/analyze", response_model=AnalysisView)
    async def analyze_folder(
        payload: LocalFolderAnalysisRequest,
        request: Request,
    ) -> AnalysisView:
        user = require_admin(request)
        require_debug_client(request)
        if repository.get_conversation_for_user(payload.conversation_id, user.id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        objective = resolve_analysis_objective(
            payload.question,
            payload.analysis_mode,
            payload.selected_sections,
        )
        try:
            files = resolve_selected_files(payload.root, payload.selected_files)
        except LocalFolderError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        root = Path(payload.root).expanduser().resolve(strict=True)

        # 原因：目录模式必须读取用户勾选的原文件，不能复制上传或隐式扩大到未勾选文件。
        # 作用：Orchestrator 只收到根目录内已验证的路径，并使用相对路径区分同名文件。
        orchestration_request = OrchestrationRequest(
            objective=objective,
            conversation_id=payload.conversation_id,
            uploaded_files=tuple(
                OrchestrationFile(
                    name=file_path.relative_to(root).as_posix(),
                    local_path=file_path,
                )
                for file_path in files
            ),
            generate_report=payload.generate_report,
            response_detail=payload.response_detail,
            analysis_mode=payload.analysis_mode,
            recipe=payload.recipe,
            selected_sections=payload.selected_sections,
            report_title="Qwopus Local Folder Analysis",
            report_basename=f"qwopus_folder_analysis_{uuid4().hex[:12]}",
        )
        orchestrator = AgentOrchestrator(runtime.current_settings(), minirag=None)
        result: OrchestrationResult = await asyncio.to_thread(
            orchestrator.run_sync,
            orchestration_request,
        )
        append_debug_record(
            source="local_folder",
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


def _node_view(node: LocalFolderNode) -> LocalFolderNodeView:
    return LocalFolderNodeView(
        name=node.name,
        relative_path=node.relative_path,
        kind=node.kind,
        children=[_node_view(child) for child in node.children],
    )
