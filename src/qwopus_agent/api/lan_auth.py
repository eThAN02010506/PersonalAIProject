"""Authentication boundary for non-loopback web access."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import secrets
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

LAN_USERNAME_ENV = "QWOPUS_LAN_USERNAME"
LAN_PASSWORD_ENV = "QWOPUS_LAN_PASSWORD"
DEFAULT_LAN_USERNAME = "qwopus"
_LOCAL_CLIENTS = {"localhost", "testclient"}
_AUTH_CHALLENGE = 'Basic realm="Qwopus-Agent", charset="UTF-8"'


@dataclass(frozen=True)
class LanAuthConfig:
    """Credentials required only when the direct client is not loopback."""

    username: str = DEFAULT_LAN_USERNAME
    password: str | None = None

    def __post_init__(self) -> None:
        if not self.username or ":" in self.username:
            raise ValueError("LAN username must be non-empty and must not contain ':'.")
        if self.password == "":
            raise ValueError("LAN password must not be empty.")

    @classmethod
    def from_environment(cls) -> LanAuthConfig:
        """Load LAN credentials without changing or writing the user's .env file."""
        username = os.getenv(LAN_USERNAME_ENV, DEFAULT_LAN_USERNAME).strip()
        password = os.getenv(LAN_PASSWORD_ENV)
        return cls(username=username, password=password)


class LanAuthMiddleware:
    """Require one browser-compatible credential for every non-loopback request."""

    def __init__(self, app: ASGIApp, *, config: LanAuthConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or _client_is_loopback(scope):
            await self.app(scope, receive, send)
            return

        if self.config.password is None:
            # 原因：监听 0.0.0.0 时若凭据未配置，静默放行会公开对话、文档和 Debug 数据。
            # 作用：非本机访问默认失败关闭；本机开发与测试仍无需额外登录。
            response = JSONResponse(
                {
                    "detail": (
                        "LAN access is disabled. Set QWOPUS_LAN_PASSWORD before "
                        "binding Qwopus-Agent to a non-loopback interface."
                    )
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return

        authorization = dict(scope.get("headers", ())).get(b"authorization", b"")
        if not _credentials_match(
            authorization,
            username=self.config.username,
            password=self.config.password,
        ):
            # 原因：401 + WWW-Authenticate 让正式 React 页面和 API 共用浏览器原生登录。
            # 作用：不在前端保存密码，也不为每个 fetch 重复实现认证状态。
            response = JSONResponse(
                {"detail": "Incorrect LAN username or password."},
                status_code=401,
                headers={"WWW-Authenticate": _AUTH_CHALLENGE},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _client_is_loopback(scope: Scope) -> bool:
    """Trust only the direct ASGI client address, never a spoofable proxy header."""
    client = scope.get("client")
    if not client:
        return False
    host = str(client[0]).split("%", 1)[0]
    if host.casefold() in _LOCAL_CLIENTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _credentials_match(
    authorization: bytes,
    *,
    username: str,
    password: str,
) -> bool:
    """Decode Basic credentials and compare both fields in constant time."""
    try:
        scheme, encoded = authorization.split(b" ", 1)
        if scheme.lower() != b"basic":
            return False
        decoded = base64.b64decode(encoded, validate=True)
        supplied_username, separator, supplied_password = decoded.partition(b":")
    except (ValueError, binascii.Error):
        return False
    if not separator:
        return False
    expected_username = username.encode("utf-8")
    expected_password = password.encode("utf-8")
    # 原因：普通字符串比较会通过响应时间泄漏已匹配的凭据前缀。
    # 作用：始终比较用户名和密码两个 UTF-8 byte 序列，降低 timing attack 风险。
    username_matches = secrets.compare_digest(supplied_username, expected_username)
    password_matches = secrets.compare_digest(supplied_password, expected_password)
    return username_matches and password_matches
