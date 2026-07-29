"""Private worker entry point for restricted pandas execution."""

from __future__ import annotations

import ast
import json
import math
import pickle
import resource
import sys
from typing import Any

import pandas as pd

from qwopus_agent.analysis.pandas_sandbox import (
    _MAX_SERIALIZED_INPUT_BYTES,
    _execute_validated_code,
    _validate_ast,
)

_CPU_SECONDS = 6
_MAX_DATA_BYTES = 1024 * 1024 * 1024
_MAX_OPEN_FILES = 32


def main() -> None:
    """Read one trusted parent payload and emit one inert JSON response."""
    try:
        serialized = sys.stdin.buffer.read(_MAX_SERIALIZED_INPUT_BYTES + 1)
        if len(serialized) > _MAX_SERIALIZED_INPUT_BYTES:
            raise ValueError("Pandas sandbox input is too large.")
        _apply_resource_limits()
        code, dataframes = _validated_request(pickle.loads(serialized))
        # 原因：父进程校验可能与 worker 版本不一致，不能把 IPC 当作信任边界。
        # 作用：执行前在受限进程中再次解析和验证同一段生成代码。
        _validate_ast(ast.parse(code, mode="exec"))
        result = _execute_validated_code(code, dataframes)
        response: dict[str, Any] = {
            "status": "ok",
            "value": _json_safe_value(result.value),
            "markdown": result.markdown,
        }
    except Exception as exc:  # noqa: BLE001 - serialize every worker failure uniformly.
        response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    # 原因：pickle 返回值会让受限 worker 控制父进程反序列化对象，不适合作为安全边界。
    # 作用：JSON 只携带数据，不会在父进程中触发任意对象构造或代码执行。
    sys.stdout.buffer.write(
        json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
    )
    sys.stdout.buffer.flush()


def _validated_request(payload: object) -> tuple[str, dict[str, pd.DataFrame]]:
    """Reject malformed IPC values before generated code reaches exec."""
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("Invalid pandas sandbox request.")
    code, dataframes = payload
    if not isinstance(code, str) or not isinstance(dataframes, dict):
        raise ValueError("Invalid pandas sandbox request.")
    if not all(
        isinstance(name, str) and isinstance(dataframe, pd.DataFrame)
        for name, dataframe in dataframes.items()
    ):
        raise ValueError("Pandas sandbox accepts only named DataFrames.")
    return code, dataframes


def _json_safe_value(value: Any) -> Any:
    """Project common pandas results onto JSON values for the parent process."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, pd.DataFrame):
        return [_json_safe_value(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.to_dict().items()
        }
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe_value(item())
    return str(value)


def _apply_resource_limits() -> None:
    """Apply kernel-enforced limits before executing generated code."""
    # 原因：墙钟超时不能限制高 CPU、core dump、文件增长或描述符耗尽。
    # 作用：Unix 内核在 worker 层独立约束这些资源；不依赖 AST 验证正确性。
    _lower_limit(resource.RLIMIT_CPU, _CPU_SECONDS)
    _lower_limit(resource.RLIMIT_CORE, 0)
    _lower_limit(resource.RLIMIT_FSIZE, 0)
    _lower_limit(resource.RLIMIT_NOFILE, _MAX_OPEN_FILES)
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        # 原因：Darwin 会拒绝在已加载 Python/Pandas 后降低内存 rlimit，且 RLIMIT_RSS
        # 本身只是 advisory；强行设置会让所有合法分析在 exec 前失败。
        # 作用：Linux/Unix 使用地址空间硬上限，macOS 继续由输入上限、Seatbelt、
        # CPU 限制和父进程墙钟超时约束，而不声明一个实际上不可用的内存限制。
        _lower_limit(resource.RLIMIT_AS, _MAX_DATA_BYTES)


def _lower_limit(resource_id: int, requested: int) -> None:
    """Lower one supported soft/hard resource limit without trying to raise it."""
    _soft, hard = resource.getrlimit(resource_id)
    effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(resource_id, (effective, effective))


if __name__ == "__main__":
    main()
