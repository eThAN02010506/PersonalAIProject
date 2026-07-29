"""Shared network boundary for diagnostics and Skill authoring."""

from __future__ import annotations

import ipaddress
import os

from fastapi import HTTPException, Request

_LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_LAN_DEBUG_ENV = "QWOPUS_DEBUG_ALLOW_LAN"


def debug_lan_enabled(explicit: bool | None) -> bool:
    """Resolve the one permission shared by every Debug Console endpoint."""
    if explicit is not None:
        return explicit
    return os.getenv(_LAN_DEBUG_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def require_debug_client(request: Request, *, allow_lan: bool) -> None:
    """Reject access outside the explicitly approved network scope."""
    host = request.client.host if request.client is not None else ""
    if not debug_host_is_allowed(host, allow_lan=allow_lan):
        # 原因：诊断和候选 spec 可能包含完整 Prompt、文档片段及模型原始输出。
        # 作用：LAN 必须显式开启，并继续阻止公网来源读取或创建调试数据。
        raise HTTPException(
            status_code=403,
            detail="Debug Console is not available to this network client.",
        )


def debug_host_is_allowed(host: str, *, allow_lan: bool) -> bool:
    """Return whether one direct client address is inside the approved scope."""
    if host in _LOCAL_CLIENTS:
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return allow_lan and (address.is_private or address.is_link_local)
