"""Host-only administrator routes for reviewed source-code changes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from qwopus_agent.api.auth import require_admin
from qwopus_agent.api.debug_access import require_debug_client
from qwopus_agent.api.repository import UserRecord
from qwopus_agent.code_workspace.models import (
    CodeChangeView,
    CodeChatMessage,
    CodeChatReply,
    CodeCommandView,
    CodeFileView,
    CodeSearchMatch,
    CodeTestResult,
    CodeWorkspaceTree,
)
from qwopus_agent.code_workspace.security import CodeWorkspaceError
from qwopus_agent.services.code_workspace_service import CodeWorkspaceService
from qwopus_agent.utils.debug_store import append_debug_record


class CodeWorkspaceScanRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class CodeWorkspaceReadRequest(BaseModel):
    root: str = Field(min_length=1, max_length=4096)
    path: str = Field(min_length=1, max_length=4096)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=600, ge=1)


class CodeWorkspaceSearchRequest(BaseModel):
    root: str = Field(min_length=1, max_length=4096)
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=100, ge=1, le=200)


class CodeWorkspaceRootRequest(BaseModel):
    root: str = Field(min_length=1, max_length=4096)


class CodeChatRequest(BaseModel):
    root: str = Field(min_length=1, max_length=4096)
    message: str = Field(min_length=1, max_length=8000)
    history: list[CodeChatMessage] = Field(default_factory=list, max_length=20)
    selected_files: list[str] = Field(default_factory=list, max_length=8)


class CodeProposalRequest(BaseModel):
    root: str = Field(min_length=1, max_length=4096)
    objective: str = Field(min_length=1, max_length=4000)
    selected_files: list[str] = Field(min_length=1, max_length=8)
    context_files: list[str] = Field(default_factory=list, max_length=8)


class CodeTestRequest(BaseModel):
    command_id: str = Field(min_length=1, max_length=100)


def build_code_workspace_router(
    service: CodeWorkspaceService,
    debug_directory: Path,
) -> APIRouter:
    """Build the local-only Code Workspace API."""
    router = APIRouter()

    @router.post("/api/code-workspaces/scan", response_model=CodeWorkspaceTree)
    def scan_workspace(
        payload: CodeWorkspaceScanRequest,
        request: Request,
    ) -> CodeWorkspaceTree:
        user = _require_code_access(request)
        try:
            result = service.scan(payload.path)
        except CodeWorkspaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _audit(
            debug_directory,
            user,
            action="scan",
            status="completed",
            result=f"Scanned {result.file_count} source files in {result.root}.",
        )
        return result

    @router.post("/api/code-workspaces/read", response_model=CodeFileView)
    def read_file(
        payload: CodeWorkspaceReadRequest,
        request: Request,
    ) -> CodeFileView:
        user = _require_code_access(request)
        try:
            result = service.read(
                payload.root,
                payload.path,
                start_line=payload.start_line,
                end_line=payload.end_line,
            )
        except CodeWorkspaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _audit(
            debug_directory,
            user,
            action="read",
            status="completed",
            result=f"Read {result.path}, lines {result.start_line}-{result.end_line}.",
        )
        return result

    @router.post(
        "/api/code-workspaces/search",
        response_model=list[CodeSearchMatch],
    )
    def search_workspace(
        payload: CodeWorkspaceSearchRequest,
        request: Request,
    ) -> list[CodeSearchMatch]:
        user = _require_code_access(request)
        try:
            result = service.search(payload.root, payload.query, limit=payload.limit)
        except CodeWorkspaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _audit(
            debug_directory,
            user,
            action="search",
            status="completed",
            result=f"Found {len(result)} literal source matches.",
        )
        return result

    @router.post(
        "/api/code-workspaces/commands",
        response_model=list[CodeCommandView],
    )
    def list_commands(
        payload: CodeWorkspaceRootRequest,
        request: Request,
    ) -> list[CodeCommandView]:
        _require_code_access(request)
        try:
            return service.list_commands(payload.root)
        except CodeWorkspaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/api/code-workspaces/chat", response_model=CodeChatReply)
    async def chat_about_workspace(
        payload: CodeChatRequest,
        request: Request,
    ) -> CodeChatReply:
        user = _require_code_access(request)
        try:
            result = await asyncio.to_thread(
                service.chat,
                root=payload.root,
                message=payload.message,
                history=payload.history,
                selected_files=payload.selected_files,
            )
        except CodeWorkspaceError as exc:
            _audit(
                debug_directory,
                user,
                action="chat",
                status="failed",
                result=str(exc),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            _audit(
                debug_directory,
                user,
                action="chat",
                status="failed",
                result=f"{type(exc).__name__}: {exc}",
            )
            raise HTTPException(
                status_code=502,
                detail=f"Code conversation model failed: {type(exc).__name__}: {exc}",
            ) from exc
        _audit(
            debug_directory,
            user,
            action="chat",
            status="completed",
            result=(
                f"Code conversation returned {result.mode}; "
                f"inspected {len(result.inspected_files)} files."
            ),
            files=result.inspected_files,
        )
        return result

    @router.get("/api/code-changes", response_model=list[CodeChangeView])
    def list_changes(request: Request) -> list[CodeChangeView]:
        user = _require_code_access(request)
        return service.list_changes(user.id)

    @router.get("/api/code-changes/{change_id}", response_model=CodeChangeView)
    def get_change(change_id: str, request: Request) -> CodeChangeView:
        user = _require_code_access(request)
        try:
            return service.get_change(change_id, user.id)
        except CodeWorkspaceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/code-changes/propose", response_model=CodeChangeView)
    async def propose_change(
        payload: CodeProposalRequest,
        request: Request,
    ) -> CodeChangeView:
        user = _require_code_access(request)
        try:
            result = await asyncio.to_thread(
                service.propose,
                root=payload.root,
                objective=payload.objective,
                selected_files=payload.selected_files,
                context_files=payload.context_files,
                owner_user_id=user.id,
            )
        except CodeWorkspaceError as exc:
            _audit(
                debug_directory,
                user,
                action="propose",
                status="failed",
                result=str(exc),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            _audit(
                debug_directory,
                user,
                action="propose",
                status="failed",
                result=f"{type(exc).__name__}: {exc}",
            )
            raise HTTPException(
                status_code=502,
                detail=f"Code proposal model failed: {type(exc).__name__}: {exc}",
            ) from exc
        _audit(
            debug_directory,
            user,
            action="propose",
            status="completed",
            result=f"Created proposal {result.id}: {result.summary}",
            files=result.changed_files,
        )
        return result

    @router.post("/api/code-changes/{change_id}/apply", response_model=CodeChangeView)
    def apply_change(change_id: str, request: Request) -> CodeChangeView:
        return _mutate_change(service, debug_directory, change_id, request, "apply")

    @router.post("/api/code-changes/{change_id}/reject", response_model=CodeChangeView)
    def reject_change(change_id: str, request: Request) -> CodeChangeView:
        return _mutate_change(service, debug_directory, change_id, request, "reject")

    @router.post("/api/code-changes/{change_id}/rollback", response_model=CodeChangeView)
    def rollback_change(change_id: str, request: Request) -> CodeChangeView:
        return _mutate_change(service, debug_directory, change_id, request, "rollback")

    @router.post(
        "/api/code-changes/{change_id}/test",
        response_model=CodeTestResult,
    )
    async def test_change(
        change_id: str,
        payload: CodeTestRequest,
        request: Request,
    ) -> CodeTestResult:
        user = _require_code_access(request)
        try:
            result = await asyncio.to_thread(
                service.run_test,
                change_id,
                user.id,
                payload.command_id,
            )
        except CodeWorkspaceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _audit(
            debug_directory,
            user,
            action="test",
            status="completed" if result.success else "failed",
            result=(
                f"Verification {result.command_id} returned {result.return_code}; "
                f"output length {len(result.output)}."
            ),
        )
        return result

    return router


def _require_code_access(request: Request) -> UserRecord:
    user = require_admin(request)
    require_debug_client(request)
    return user


def _mutate_change(
    service: CodeWorkspaceService,
    debug_directory: Path,
    change_id: str,
    request: Request,
    action: str,
) -> CodeChangeView:
    user = _require_code_access(request)
    operation = {
        "apply": service.apply,
        "reject": service.reject,
        "rollback": service.rollback,
    }[action]
    try:
        result = operation(change_id, user.id)
    except CodeWorkspaceError as exc:
        _audit(
            debug_directory,
            user,
            action=action,
            status="failed",
            result=str(exc),
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _audit(
        debug_directory,
        user,
        action=action,
        status="completed",
        result=f"{action.title()} completed for proposal {change_id}.",
        files=result.changed_files,
    )
    return result


def _audit(
    directory: Path,
    user: UserRecord,
    *,
    action: str,
    status: str,
    result: str,
    files: list[str] | None = None,
) -> None:
    append_debug_record(
        source="code_workspace",
        status=status,
        result=result,
        trace=[{"phase": action, "files": files or []}],
        debug_runs=[],
        user_id=user.id,
        username=user.username,
        directory=directory,
    )
