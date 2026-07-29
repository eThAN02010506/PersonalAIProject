"""Host-only network boundary for diagnostics and sensitive local operations."""

from __future__ import annotations

from fastapi import HTTPException, Request

_LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def require_debug_client(request: Request) -> None:
    """Reject diagnostics access from every non-loopback client."""
    host = request.client.host if request.client is not None else ""
    if not debug_host_is_allowed(host):
        # 原因：诊断和候选 spec 可能包含完整 Prompt、文档片段及模型原始输出。
        # 作用：即使 API 监听局域网，只有运行 Qwopus-Agent 的主机能访问这些数据。
        raise HTTPException(
            status_code=403,
            detail="Debug Console is available only on the host machine.",
        )


def debug_host_is_allowed(host: str) -> bool:
    """Return whether the direct client is the application host."""
    return host.split("%", 1)[0] in _LOCAL_CLIENTS
