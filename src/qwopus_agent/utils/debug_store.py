"""Local persistence for raw Agent diagnostics consumed by the debug console."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_DEBUG_DIRECTORY = Path("logs/debug_runs")
MAX_DEBUG_RECORDS = 200
MAX_DEBUG_STORAGE_BYTES = 64 * 1024 * 1024
MAX_DEBUG_RECORD_AGE = timedelta(days=14)
logger = logging.getLogger("qwopus_agent.debug_store")


def append_debug_record(
    *,
    source: str,
    status: str,
    trace: Any,
    debug_runs: Any,
    result: str = "",
    run_id: str | None = None,
    user_id: str | None = None,
    username: str | None = None,
    directory: Path = DEFAULT_DEBUG_DIRECTORY,
) -> Path | None:
    """Persist one internal run without exposing it through the public HTTP response."""
    record_id = uuid4().hex
    timestamp = datetime.now(UTC)
    record = {
        "id": record_id,
        "timestamp": timestamp.isoformat(),
        "source": source,
        "status": status,
        "run_id": run_id or record_id,
        "user_id": user_id,
        "username": username,
        "result": result,
        "trace": _json_value(trace),
        "debug_runs": _json_value(debug_runs),
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{record_id}.json"
        path = directory / filename
        temporary_path = directory / f".{filename}.tmp"
        # 原因：后台任务与 Debug API 并发运行，Console 可能正好读到写入中的大 Trace。
        # 作用：先写临时文件再原子替换，保证读取方只会看到完整 JSON 记录。
        temporary_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary_path.replace(path)
        _prune_debug_records(directory)
        return path
    except (OSError, TypeError, ValueError):
        # 原因：调试记录属于旁路能力，磁盘问题不能让用户的正式 Agent 请求失败。
        # 作用：保留错误日志并让主业务响应继续完成。
        logger.exception("debug_record_write_failed source=%s run_id=%s", source, run_id)
        return None


def load_debug_records(
    *,
    limit: int = 50,
    directory: Path = DEFAULT_DEBUG_DIRECTORY,
) -> list[dict[str, Any]]:
    """Load newest complete debug records while ignoring interrupted files."""
    if limit <= 0 or not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def load_debug_record(
    record_id: str,
    *,
    directory: Path = DEFAULT_DEBUG_DIRECTORY,
) -> dict[str, Any] | None:
    """Load one immutable record without returning every raw trace to the client."""
    if not record_id or len(record_id) > 128:
        return None
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("id") == record_id:
            return payload
    return None


def _json_value(value: Any) -> Any:
    """Convert dataclasses, Pydantic models and nested containers into JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _prune_debug_records(
    directory: Path,
    keep: int = MAX_DEBUG_RECORDS,
    max_bytes: int = MAX_DEBUG_STORAGE_BYTES,
    max_age: timedelta = MAX_DEBUG_RECORD_AGE,
) -> None:
    """Bound complete diagnostics by age, count, and aggregate bytes."""
    paths = sorted(directory.glob("*.json"), reverse=True)
    cutoff = datetime.now(UTC).timestamp() - max_age.total_seconds()
    retained: list[tuple[Path, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime < cutoff:
            _unlink_debug_record(path)
            continue
        retained.append((path, stat.st_size))

    total_bytes = sum(size for _, size in retained)
    # 原因：单个 Trace 大小时，仅限制记录数量仍可能让 Debug 目录持续占满磁盘。
    # 作用：从最旧记录开始删除，直到数量和总字节数同时回到固定上限。
    for path, size in retained[keep:]:
        if _unlink_debug_record(path):
            total_bytes -= size
    retained = retained[:keep]
    while retained and total_bytes > max_bytes:
        path, size = retained.pop()
        if _unlink_debug_record(path):
            total_bytes -= size


def _unlink_debug_record(path: Path) -> bool:
    """Delete one complete record while keeping cleanup failure non-fatal."""
    try:
        path.unlink()
    except OSError:
        logger.warning("debug_record_prune_failed path=%s", path)
        return False
    return True
