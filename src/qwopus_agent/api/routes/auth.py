"""Account bootstrap, login, password, and administrator user routes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response

from qwopus_agent.api.auth import (
    SESSION_COOKIE_NAME,
    AuthService,
    clear_session_cookie,
    current_user,
    require_admin,
    set_session_cookie,
)
from qwopus_agent.api.models import (
    AuthStatusView,
    InitialAdminCreate,
    LoginRequest,
    PasswordChange,
    UserActiveUpdate,
    UserCreate,
    UserView,
)
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.runs import ChatRunRegistry
from qwopus_agent.documents import DocumentStore
from qwopus_agent.memory import ConversationKnowledgeManager


def build_auth_router(
    auth: AuthService,
    repository: ConversationRepository,
    document_store: DocumentStore,
    knowledge: ConversationKnowledgeManager,
    report_directory: Path,
    runs: ChatRunRegistry,
) -> APIRouter:
    """Build the only anonymous API entry points and account administration."""
    router = APIRouter()

    @router.get("/api/auth/status", response_model=AuthStatusView)
    def auth_status(request: Request) -> AuthStatusView:
        user = getattr(request.state, "current_user", None)
        return AuthStatusView(
            bootstrap_required=not repository.has_users(),
            user=UserView.model_validate(user) if user is not None else None,
        )

    @router.post("/api/auth/bootstrap", response_model=AuthStatusView, status_code=201)
    def bootstrap(
        payload: InitialAdminCreate,
        request: Request,
        response: Response,
    ) -> AuthStatusView:
        try:
            user = auth.bootstrap(
                username=payload.username,
                display_name=payload.display_name,
                password=payload.password,
            )
        except (ValueError, sqlite3.IntegrityError) as exc:
            if isinstance(exc, sqlite3.IntegrityError):
                raise HTTPException(
                    status_code=409,
                    detail="Username is already in use.",
                ) from exc
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if user is None:
            raise HTTPException(
                status_code=409,
                detail="Qwopus-Agent has already been initialized.",
            )

        # 原因：账号功能加入前的本地文档和报告没有所有者，直接启用默认拒绝会让它们消失。
        # 作用：仅首次管理员认领历史资源；之后创建的账号不会自动获得任何旧内容。
        repository.claim_legacy_files(
            user.id,
            document_ids=[
                document.document_id
                for document in document_store.list_documents()
            ],
            report_filenames=[
                path.name
                for path in report_directory.iterdir()
                if path.is_file() and not path.is_symlink()
            ] if report_directory.is_dir() else [],
        )
        knowledge.claim_legacy_global(user.id)
        grant = auth.issue_session(user)
        set_session_cookie(response, grant, request)
        return AuthStatusView(bootstrap_required=False, user=UserView.model_validate(user))

    @router.post("/api/auth/login", response_model=AuthStatusView)
    def login(
        payload: LoginRequest,
        request: Request,
        response: Response,
    ) -> AuthStatusView:
        try:
            user = auth.authenticate(payload.username, payload.password)
        except ValueError:
            user = None
        if user is None:
            raise HTTPException(status_code=401, detail="Incorrect username or password.")
        grant = auth.issue_session(user)
        set_session_cookie(response, grant, request)
        return AuthStatusView(bootstrap_required=False, user=UserView.model_validate(user))

    @router.post("/api/auth/logout", status_code=204)
    def logout(request: Request) -> Response:
        auth.revoke_session(request.cookies.get(SESSION_COOKIE_NAME))
        response = Response(status_code=204)
        clear_session_cookie(response)
        return response

    @router.get("/api/auth/me", response_model=UserView)
    def me(request: Request) -> UserView:
        return UserView.model_validate(current_user(request))

    @router.post("/api/auth/password", response_model=AuthStatusView)
    def change_password(
        payload: PasswordChange,
        request: Request,
        response: Response,
    ) -> AuthStatusView:
        user = current_user(request)
        try:
            changed = auth.change_password(
                user,
                current_password=payload.current_password,
                new_password=payload.new_password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not changed:
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        refreshed = repository.get_user(user.id)
        if refreshed is None:
            raise HTTPException(status_code=401, detail="Account is no longer available.")
        grant = auth.issue_session(refreshed)
        set_session_cookie(response, grant, request)
        return AuthStatusView(
            bootstrap_required=False,
            user=UserView.model_validate(refreshed),
        )

    @router.get("/api/users", response_model=list[UserView])
    def list_users(request: Request) -> list[UserView]:
        require_admin(request)
        return [UserView.model_validate(user) for user in repository.list_users()]

    @router.post("/api/users", response_model=UserView, status_code=201)
    def create_user(payload: UserCreate, request: Request) -> UserView:
        require_admin(request)
        try:
            user = auth.create_user(
                username=payload.username,
                display_name=payload.display_name,
                password=payload.password,
                role=payload.role,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="Username is already in use.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return UserView.model_validate(user)

    @router.patch("/api/users/{user_id}", response_model=UserView)
    def update_user(
        user_id: str,
        payload: UserActiveUpdate,
        request: Request,
    ) -> UserView:
        administrator = require_admin(request)
        target = repository.get_user(user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        if target.id == administrator.id and not payload.active:
            raise HTTPException(
                status_code=409,
                detail="You cannot disable your current account.",
            )
        if (
            target.role == "admin"
            and target.active
            and not payload.active
            and repository.active_admin_count() <= 1
        ):
            raise HTTPException(
                status_code=409,
                detail="At least one active administrator is required.",
            )
        updated = repository.set_user_active(user_id, payload.active)
        if updated is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        if not payload.active:
            runs.cancel_user(user_id)
        return UserView.model_validate(updated)

    return router
