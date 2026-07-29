"""Server-side account authentication and request-wide authorization context."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from urllib.parse import urlsplit

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp

from qwopus_agent.api.debug_access import debug_host_is_allowed
from qwopus_agent.api.repository import ConversationRepository, UserRecord

SESSION_COOKIE_NAME = "qwopus_session"
SESSION_LIFETIME = timedelta(days=7)
_PUBLIC_API_PATHS = {
    "/api/auth/status",
    "/api/auth/bootstrap",
    "/api/auth/login",
    "/api/health",
}
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_DEV_ORIGINS = {"http://127.0.0.1:5173", "http://localhost:5173"}


@dataclass(frozen=True)
class SessionGrant:
    """Opaque browser token plus the safe account it represents."""

    token: str
    user: UserRecord
    expires_at: datetime


class AuthService:
    """Hash passwords and keep opaque browser sessions in SQLite."""

    def __init__(self, repository: ConversationRepository) -> None:
        self.repository = repository
        # 原因：快速 SHA-256 适合令牌索引，但不适合抵抗离线密码猜测。
        # 作用：使用 OWASP 的 Argon2id 最低配置；salt 由 argon2-cffi 自动生成。
        self._passwords = PasswordHasher(
            time_cost=2,
            memory_cost=19_456,
            parallelism=1,
        )
        # 原因：未知用户名若直接返回，会通过耗时差异泄漏账号是否存在。
        # 作用：不存在的用户也执行一次真实 Argon2id 验证，保持登录路径相近。
        self._dummy_hash = self._passwords.hash(secrets.token_urlsafe(24))

    def bootstrap(
        self,
        *,
        username: str,
        display_name: str,
        password: str,
    ) -> UserRecord | None:
        return self.repository.create_initial_admin(
            username=normalize_username(username),
            display_name=normalize_display_name(display_name),
            password_hash=self.hash_password(password),
        )

    def create_user(
        self,
        *,
        username: str,
        display_name: str,
        password: str,
        role: Literal["admin", "member"],
    ) -> UserRecord:
        if role not in {"admin", "member"}:
            raise ValueError("Unsupported account role.")
        return self.repository.create_user(
            username=normalize_username(username),
            display_name=normalize_display_name(display_name),
            password_hash=self.hash_password(password),
            role=role,
        )

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        normalized = normalize_username(username)
        stored = self.repository.get_user_with_password(normalized)
        password_hash = stored[1] if stored is not None else self._dummy_hash
        if not self.verify_password(password_hash, password):
            return None
        if stored is None or not stored[0].active:
            return None
        return stored[0]

    def change_password(
        self,
        user: UserRecord,
        *,
        current_password: str,
        new_password: str,
    ) -> bool:
        stored = self.repository.get_user_with_password(user.username)
        if stored is None or not self.verify_password(stored[1], current_password):
            return False
        return self.repository.set_user_password(
            user.id,
            self.hash_password(new_password),
        )

    def hash_password(self, password: str) -> str:
        validate_password(password)
        return self._passwords.hash(password)

    def verify_password(self, password_hash: str, password: str) -> bool:
        try:
            return bool(self._passwords.verify(password_hash, password))
        except (InvalidHashError, VerificationError):
            return False

    def issue_session(self, user: UserRecord) -> SessionGrant:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + SESSION_LIFETIME
        self.repository.create_session(
            token_hash=_token_hash(token),
            user_id=user.id,
            expires_at=expires_at.isoformat(),
        )
        return SessionGrant(token=token, user=user, expires_at=expires_at)

    def resolve_session(self, token: str | None) -> UserRecord | None:
        if not token:
            return None
        return self.repository.user_for_session(_token_hash(token))

    def revoke_session(self, token: str | None) -> None:
        if token:
            self.repository.delete_session(_token_hash(token))


class AccountAuthMiddleware(BaseHTTPMiddleware):
    """Resolve one account for every request and deny private APIs by default."""

    def __init__(self, app: ASGIApp, *, auth: AuthService) -> None:
        super().__init__(app)
        self.auth = auth

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method in _UNSAFE_METHODS and not _origin_is_allowed(request):
            return JSONResponse(
                {"detail": "Request origin is not allowed."},
                status_code=403,
            )

        token = request.cookies.get(SESSION_COOKIE_NAME)
        user = await asyncio.to_thread(self.auth.resolve_session, token)
        request.state.current_user = user
        path = request.url.path

        if path == "/debug" or path.startswith("/debug/"):
            host = request.client.host if request.client is not None else ""
            if not debug_host_is_allowed(host):
                # 原因：Debug 记录包含所有账号的 Prompt、文档片段和运行轨迹。
                # 作用：即使外层 LAN Basic Auth 成功，非本机请求仍无法加载诊断入口。
                return JSONResponse(
                    {"detail": "Debug Console is available only on the host machine."},
                    status_code=403,
                )
            if user is None:
                return RedirectResponse(url="/", status_code=303)
            if user.role != "admin":
                return JSONResponse(
                    {"detail": "Administrator access is required."},
                    status_code=403,
                )

        if (
            path.startswith("/api/")
            and path not in _PUBLIC_API_PATHS
            and user is None
        ):
            response: Response = JSONResponse(
                {"detail": "Authentication required."},
                status_code=401,
            )
        else:
            response = await call_next(request)

        if path.startswith("/api/"):
            # 原因：账号、对话与文档响应不应残留在共享浏览器或中间缓存中。
            # 作用：所有 API 数据均标记为不可存储，退出账号后不会由缓存重新显示。
            response.headers["Cache-Control"] = "no-store"
        return response


def current_user(request: Request) -> UserRecord:
    """Return the middleware-resolved account or fail closed."""
    user = getattr(request.state, "current_user", None)
    if not isinstance(user, UserRecord):
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_admin(request: Request) -> UserRecord:
    """Require the current account to own administrative capabilities."""
    user = current_user(request)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access is required.")
    return user


def set_session_cookie(response: Response, grant: SessionGrant, request: Request) -> None:
    """Store only the opaque token in a hardened first-party cookie."""
    response.set_cookie(
        SESSION_COOKIE_NAME,
        grant.token,
        max_age=int(SESSION_LIFETIME.total_seconds()),
        expires=grant.expires_at,
        path="/",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="strict",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="strict",
    )


def normalize_username(value: str) -> str:
    """Normalize a stable login identifier without accepting path-like syntax."""
    username = unicodedata.normalize("NFKC", value).strip().casefold()
    if not 3 <= len(username) <= 32:
        raise ValueError("Username must contain 3 to 32 characters.")
    if not username[0].isalnum() or any(
        not (character.isalnum() or character in {"_", "-", "."})
        for character in username
    ):
        raise ValueError(
            "Username must start with a letter or number and use only letters, "
            "numbers, '.', '_' or '-'."
        )
    return username


def normalize_display_name(value: str) -> str:
    display_name = unicodedata.normalize("NFKC", value).strip()
    if not 1 <= len(display_name) <= 80:
        raise ValueError("Display name must contain 1 to 80 characters.")
    return display_name


def validate_password(password: str) -> None:
    """Apply a length policy while allowing passphrases and every writing system."""
    if not 8 <= len(password) <= 256:
        raise ValueError("Password must contain 8 to 256 characters.")


def _token_hash(token: str) -> str:
    # 原因：数据库泄漏不应立即暴露仍然有效的浏览器 Bearer Token。
    # 作用：只保存固定长度 SHA-256 索引；随机令牌本身仅存在于 HttpOnly Cookie。
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _origin_is_allowed(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        # 原因：CLI、测试和本地脚本通常没有 Origin；CSRF 攻击来自浏览器并会携带它。
        # 作用：保留非浏览器 API 使用，同时验证所有浏览器写请求的同源关系。
        return True
    if origin in _DEV_ORIGINS:
        return True
    parsed = urlsplit(origin)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.casefold() == request.headers.get("host", "").casefold()
    )
