"""Read-only access to persisted Agent diagnostics from approved clients."""

from __future__ import annotations

import asyncio
import os
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from qwopus_agent.api.auth import require_admin
from qwopus_agent.api.debug_access import require_debug_client
from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    DebugOverviewView,
    DebugRecordSummaryView,
    DebugRuntimeLogView,
    ModelSettingsView,
)
from qwopus_agent.api.runs import ChatRunRegistry
from qwopus_agent.utils.debug_store import load_debug_record, load_debug_records


def build_debug_router(
    runtime: RuntimeModelController,
    runs: ChatRunRegistry,
    debug_directory: Path,
    runtime_log_path: Path,
    *,
    started_at: float,
) -> APIRouter:
    """Build read-only diagnostics routes for approved clients."""
    router = APIRouter()
    @router.get("/api/debug", response_model=DebugOverviewView)
    async def debug_overview(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        log_lines: int = Query(default=500, ge=0, le=2_000),
    ) -> DebugOverviewView:
        require_admin(request)
        require_debug_client(request)
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
                request_timeout_seconds=model_status.settings.timeout_seconds,
                max_retries=model_status.settings.max_retries,
                run_timeout_seconds=model_status.settings.run_timeout_seconds,
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
        require_admin(request)
        require_debug_client(request)
        record = await asyncio.to_thread(
            load_debug_record,
            record_id,
            directory=debug_directory,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Debug record not found.")
        return record

    return router


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
        user_id=str(record["user_id"]) if record.get("user_id") else None,
        username=str(record["username"]) if record.get("username") else None,
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
        lines = _read_tail_lines(path, line_limit)
        return DebugRuntimeLogView(
            path=str(path),
            exists=True,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            total_lines=_count_lines(path),
            lines=lines,
        )
    except OSError as exc:
        # 原因：日志轮转可能恰好发生在 stat 与读取之间。
        # 作用：Debug Console 显示读取错误，但不会让整个诊断接口失败。
        return DebugRuntimeLogView(path=str(path), exists=True, error=str(exc))


_TAIL_READ_CHUNK_BYTES = 64 * 1024
_MAX_TAIL_READ_BYTES = 16 * 1024 * 1024


def _read_tail_lines(path: Path, line_limit: int) -> list[str]:
    """Return the newest ``line_limit`` lines without reading the whole file.

    原因：大日志一次性 read_text 会占用整个文件大小的内存，日志轮转前可能
    膨胀到数百 MB。作用：从文件尾部按块反向扫描，只保留最后有界的行集。
    """
    if line_limit <= 0:
        return []
    size = path.stat().st_size
    if size == 0:
        return []
    block_end = size
    read_bytes = 0
    chunks: list[str] = []
    while block_end > 0 and read_bytes < _MAX_TAIL_READ_BYTES:
        block_start = max(0, block_end - _TAIL_READ_CHUNK_BYTES)
        read_bytes += block_end - block_start
        with path.open("rb") as handle:
            handle.seek(block_start)
            block = handle.read(block_end - block_start)
        block_end = block_start
        chunks.append(block.decode("utf-8", errors="replace"))
        combined = "".join(reversed(chunks))
        line_count = combined.count("\n")
        if line_count >= line_limit:
            break
    full_tail = "".join(reversed(chunks))
    lines = full_tail.splitlines()
    return lines[-line_limit:] if line_limit else []


def _count_lines(path: Path) -> int:
    """Count newlines with a bounded read to avoid loading the whole file."""
    count = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            count += block.count(b"\n")
    return count
