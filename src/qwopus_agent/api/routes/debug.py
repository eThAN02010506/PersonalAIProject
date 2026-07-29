"""Read-only access to persisted Agent diagnostics from approved clients."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    DebugOverviewView,
    DebugRecordSummaryView,
    DebugRuntimeLogView,
    ModelSettingsView,
)
from qwopus_agent.api.runs import ChatRunRegistry
from qwopus_agent.utils.debug_store import load_debug_record, load_debug_records

_LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_LAN_DEBUG_ENV = "QWOPUS_DEBUG_ALLOW_LAN"


def build_debug_router(
    runtime: RuntimeModelController,
    runs: ChatRunRegistry,
    debug_directory: Path,
    runtime_log_path: Path,
    *,
    started_at: float,
    allow_lan: bool | None = None,
) -> APIRouter:
    """Build a read-only diagnostics route without exposing mutation endpoints."""
    router = APIRouter()
    lan_enabled = _environment_flag(_LAN_DEBUG_ENV) if allow_lan is None else allow_lan

    @router.get("/api/debug", response_model=DebugOverviewView)
    async def debug_overview(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        log_lines: int = Query(default=500, ge=0, le=2_000),
    ) -> DebugOverviewView:
        _require_debug_client(request, allow_lan=lan_enabled)
        records = await asyncio.to_thread(
            load_debug_records,
            limit=limit,
            directory=debug_directory,
        )
        model_status = await asyncio.to_thread(runtime.status)
        active_runs, completed_runs = runs.debug_counts()
        source_counts = Counter(str(record.get("source") or "unknown") for record in records)
        status_counts = Counter(str(record.get("status") or "unknown") for record in records)
        return DebugOverviewView(
            generated_at=datetime.now(UTC).isoformat(),
            uptime_seconds=round(max(0.0, time.monotonic() - started_at), 3),
            process_id=os.getpid(),
            python_version=platform.python_version(),
            platform=platform.platform(),
            model=ModelSettingsView(
                mode=model_status.mode,
                model_online=model_status.online,
                message=model_status.message,
                model=model_status.settings.model_id,
                base_url=model_status.settings.base_url,
                local_model_path=model_status.local_model_path,
                context_window_tokens=(
                    model_status.settings.capabilities.context_window_tokens
                ),
                agent_mode=model_status.settings.capabilities.agent_mode,
                supports_structured_output=(
                    model_status.settings.capabilities.supports_structured_output
                ),
                supports_vision=model_status.settings.capabilities.supports_vision,
            ),
            active_runs=active_runs,
            completed_runs=completed_runs,
            record_count=len(records),
            record_storage_bytes=_directory_size(debug_directory),
            source_counts=dict(source_counts),
            status_counts=dict(status_counts),
            records=[_record_summary(record) for record in records],
            runtime_log=await asyncio.to_thread(_read_runtime_log, runtime_log_path, log_lines),
        )

    @router.get("/api/debug/records/{record_id}", response_model=dict[str, object])
    async def debug_record(request: Request, record_id: str) -> dict[str, object]:
        _require_debug_client(request, allow_lan=lan_enabled)
        record = await asyncio.to_thread(
            load_debug_record,
            record_id,
            directory=debug_directory,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Debug record not found.")
        return record

    return router


def _require_debug_client(request: Request, *, allow_lan: bool) -> None:
    """Reject raw diagnostic access outside the explicitly approved network scope."""
    host = request.client.host if request.client is not None else ""
    if not _debug_host_is_allowed(host, allow_lan=allow_lan):
        # 原因：记录可能包含完整 Prompt、文档片段和 Tool Observation。
        # 作用：LAN 必须显式开启，并继续阻止公网来源读取原始诊断。
        raise HTTPException(
            status_code=403,
            detail="Debug diagnostics are not available to this network client.",
        )


def _debug_host_is_allowed(host: str, *, allow_lan: bool) -> bool:
    """Return whether one direct client address is inside the approved scope."""
    if host in _LOCAL_CLIENTS:
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return allow_lan and (address.is_private or address.is_link_local)


def _environment_flag(name: str) -> bool:
    """Read one explicit boolean environment switch."""
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _directory_size(directory: Path) -> int:
    """Return the size of complete records without following unrelated files."""
    try:
        return sum(path.stat().st_size for path in directory.glob("*.json") if path.is_file())
    except OSError:
        return 0


def _record_summary(record: dict[str, object]) -> DebugRecordSummaryView:
    """Remove large raw fields from the auto-refreshing overview response."""
    trace = record.get("trace")
    debug_runs = record.get("debug_runs")
    result = str(record.get("result") or "")
    return DebugRecordSummaryView(
        id=str(record.get("id") or ""),
        timestamp=str(record["timestamp"]) if record.get("timestamp") else None,
        source=str(record.get("source") or "unknown"),
        status=str(record.get("status") or "unknown"),
        run_id=str(record.get("run_id") or record.get("id") or "unknown"),
        result_preview=result[:240],
        trace_events=len(trace) if isinstance(trace, list) else 0,
        agent_runs=len(debug_runs) if isinstance(debug_runs, list) else 0,
    )


def _read_runtime_log(path: Path, line_limit: int) -> DebugRuntimeLogView:
    """Read a bounded log tail while preserving metadata for troubleshooting."""
    if not path.is_file():
        return DebugRuntimeLogView(path=str(path), exists=False)
    try:
        stat = path.stat()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return DebugRuntimeLogView(
            path=str(path),
            exists=True,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            total_lines=len(lines),
            lines=lines[-line_limit:] if line_limit else [],
        )
    except OSError as exc:
        # 原因：日志轮转可能恰好发生在 stat 与读取之间。
        # 作用：Debug Console 显示读取错误，但不会让整个诊断接口失败。
        return DebugRuntimeLogView(path=str(path), exists=True, error=str(exc))
