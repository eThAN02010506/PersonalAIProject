"""Runtime model endpoint selection and local MLX server lifecycle."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock, RLock
from typing import BinaryIO, Literal
from urllib.parse import urlparse, urlunparse

from qwopus_agent.integrations.smolagents_model import probe_model_settings
from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
)
from qwopus_agent.llm import ModelCapabilities

LOCAL_MLX_LOG = Path("logs/local_mlx_server.log")
MODEL_STATUS_CACHE_SECONDS = 5.0


class ModelRuntimeError(RuntimeError):
    """Raised when a requested model endpoint cannot be activated."""


@dataclass(frozen=True)
class RuntimeModelStatus:
    """Current model selection returned to HTTP and UI layers."""

    mode: Literal["remote", "local"]
    settings: SmolagentsModelSettings
    online: bool
    message: str
    local_model_path: str | None = None


class RuntimeModelController:
    """Own mutable API settings and the optional child MLX server."""

    def __init__(
        self,
        settings: SmolagentsModelSettings | None = None,
        *,
        startup_timeout_seconds: int = 300,
    ) -> None:
        self._settings = settings or SmolagentsModelSettings.from_env()
        self._mode: Literal["remote", "local"] = "remote"
        self._local_model_path: Path | None = None
        self._local_process: subprocess.Popen[bytes] | None = None
        self._local_log_handle: BinaryIO | None = None
        self._startup_timeout_seconds = startup_timeout_seconds
        self._state_lock = RLock()
        self._configuration_lock = Lock()
        self._status_lock = Lock()
        self._cached_status: RuntimeModelStatus | None = None
        self._cached_status_at = 0.0

    def current_settings(self) -> SmolagentsModelSettings:
        """Return a snapshot with the model id currently reported by the server."""
        with self._state_lock:
            settings = self._settings
        resolved = resolve_model_settings(settings)
        with self._state_lock:
            if self._settings.base_url == settings.base_url:
                self._settings = resolved
        return resolved

    def status(self, *, force: bool = False) -> RuntimeModelStatus:
        """Probe and describe the currently selected endpoint."""
        now = time.monotonic()
        with self._state_lock:
            cached = self._cached_status
            if (
                not force
                and cached is not None
                and now - self._cached_status_at < MODEL_STATUS_CACHE_SECONDS
            ):
                return cached

        with self._status_lock:
            now = time.monotonic()
            with self._state_lock:
                cached = self._cached_status
                if (
                    not force
                    and cached is not None
                    and now - self._cached_status_at < MODEL_STATUS_CACHE_SECONDS
                ):
                    return cached
                settings = self._settings
            settings, online, message = probe_model_settings(settings)
            with self._state_lock:
                if self._settings.base_url == settings.base_url:
                    self._settings = settings
                status = RuntimeModelStatus(
                    mode=self._mode,
                    settings=settings,
                    online=online,
                    message=message,
                    local_model_path=(
                        str(self._local_model_path)
                        if self._local_model_path is not None
                        else None
                    ),
                )
                # 原因：多个 Debug 标签页会在同一五秒窗口重复探测相同模型地址。
                # 作用：短 TTL 复用状态并由 _status_lock 合并并发探测，不缓存配置变更。
                self._cached_status = status
                self._cached_status_at = now
                return status

    def require_online_settings(self) -> SmolagentsModelSettings:
        """Return current settings only when the selected model endpoint is reachable."""
        status = self.status(force=True)
        if not status.online:
            # 原因：模型已离线时启动后台 Agent 只会等待 HTTP 超时并暴露 provider 异常。
            # 作用：在创建进程和保存用户消息前快速失败，向 API 提供明确的服务不可用原因。
            raise ModelRuntimeError(status.message)
        return status.settings

    def configure_remote(
        self,
        base_url: str,
        capabilities: ModelCapabilities | None = None,
        *,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        run_timeout_seconds: int | None = None,
    ) -> RuntimeModelStatus:
        """Switch to a reachable OpenAI-compatible server."""
        normalized_url = _normalize_base_url(base_url)
        with self._configuration_lock:
            self._invalidate_status()
            with self._state_lock:
                candidate = replace(
                    self._settings,
                    base_url=normalized_url,
                    capabilities=capabilities or self._settings.capabilities,
                    timeout_seconds=(
                        timeout_seconds
                        if timeout_seconds is not None
                        else self._settings.timeout_seconds
                    ),
                    max_retries=(
                        max_retries
                        if max_retries is not None
                        else self._settings.max_retries
                    ),
                    run_timeout_seconds=(
                        run_timeout_seconds
                        if run_timeout_seconds is not None
                        else self._settings.run_timeout_seconds
                    ),
                )
            candidate, online, message = probe_model_settings(candidate)
            if not online:
                raise ModelRuntimeError(message)

            with self._state_lock:
                local_process = self._detach_local_process()
                self._settings = candidate
                self._mode = "remote"
                self._local_model_path = None
                status = self._record_status(candidate, True, message)
            _stop_process(local_process)
            self._close_local_log()
        return status

    def configure_local(
        self,
        model_path: str,
        capabilities: ModelCapabilities | None = None,
        *,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        run_timeout_seconds: int | None = None,
    ) -> RuntimeModelStatus:
        """Start an MLX server for a local model directory and select it."""
        path = _validate_model_path(model_path)
        executable = _find_mlx_server(path)

        with self._configuration_lock:
            self._invalidate_status()
            with self._state_lock:
                if (
                    self._mode == "local"
                    and self._local_model_path == path
                    and self._local_process is not None
                    and self._local_process.poll() is None
                ):
                    return self.status()
                previous_process = self._detach_local_process()
            _stop_process(previous_process)
            self._close_local_log()

            port = _available_port()
            base_url = f"http://127.0.0.1:{port}/v1"
            candidate = replace(
                self._settings,
                model_id=path.name,
                base_url=base_url,
                capabilities=capabilities or self._settings.capabilities,
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self._settings.timeout_seconds
                ),
                max_retries=(
                    max_retries
                    if max_retries is not None
                    else self._settings.max_retries
                ),
                run_timeout_seconds=(
                    run_timeout_seconds
                    if run_timeout_seconds is not None
                    else self._settings.run_timeout_seconds
                ),
            )
            process = self._start_local_process(executable, path, port)
            try:
                _wait_for_server(
                    process,
                    candidate,
                    timeout_seconds=self._startup_timeout_seconds,
                )
            except ModelRuntimeError as exc:
                _stop_process(process)
                self._close_local_log()
                details = _log_tail(LOCAL_MLX_LOG)
                suffix = f"\n{details}" if details else ""
                raise ModelRuntimeError(f"{exc}{suffix}") from exc

            with self._state_lock:
                self._local_process = process
                self._local_model_path = path
                self._settings = resolve_model_settings(candidate)
                self._mode = "local"
        return self.status(force=True)

    def close(self) -> None:
        """Stop only the local server started by this controller."""
        with self._configuration_lock:
            with self._state_lock:
                process = self._detach_local_process()
            _stop_process(process)
            self._close_local_log()
            self._invalidate_status()

    def _invalidate_status(self) -> None:
        with self._state_lock:
            self._cached_status = None
            self._cached_status_at = 0.0

    def _record_status(
        self,
        settings: SmolagentsModelSettings,
        online: bool,
        message: str,
    ) -> RuntimeModelStatus:
        """Cache a status while the caller holds the state lock."""
        status = RuntimeModelStatus(
            mode=self._mode,
            settings=settings,
            online=online,
            message=message,
            local_model_path=(
                str(self._local_model_path)
                if self._local_model_path is not None
                else None
            ),
        )
        # 原因：配置切换已经得到可信探测结果，立即再次请求同一端点没有新增信息。
        # 作用：直接记录切换结果，使远程模型配置只访问一次 /models。
        self._cached_status = status
        self._cached_status_at = time.monotonic()
        return status

    def _start_local_process(
        self,
        executable: Path,
        model_path: Path,
        port: int,
    ) -> subprocess.Popen[bytes]:
        LOCAL_MLX_LOG.parent.mkdir(parents=True, exist_ok=True)
        self._local_log_handle = LOCAL_MLX_LOG.open("ab")
        command = [
            str(executable),
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        # 原因：模型路径来自 UI，拼接 shell 字符串会产生命令注入风险。
        # 作用：以参数数组直接启动 MLX，并把完整启动错误写入本地调试日志。
        return subprocess.Popen(
            command,
            stdout=self._local_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _detach_local_process(self) -> subprocess.Popen[bytes] | None:
        process = self._local_process
        self._local_process = None
        return process

    def _close_local_log(self) -> None:
        if self._local_log_handle is not None:
            self._local_log_handle.close()
            self._local_log_handle = None


def _normalize_base_url(base_url: str) -> str:
    value = base_url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelRuntimeError("Model address must be a complete http:// or https:// URL.")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _validate_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ModelRuntimeError("Local model path must be an existing directory.")
    if not any(path.glob("*.safetensors")):
        raise ModelRuntimeError("Local model directory contains no .safetensors weights.")
    return path


def _find_mlx_server(model_path: Path) -> Path:
    configured = os.getenv("QWOPUS_MLX_SERVER_EXECUTABLE")
    candidates = [
        Path(configured).expanduser() if configured else None,
        model_path.parent / ".venv/bin/mlx_lm.server",
        Path(sys.executable).with_name("mlx_lm.server"),
    ]
    discovered = shutil.which("mlx_lm.server")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ModelRuntimeError(
        "mlx_lm.server was not found. Install mlx-lm in the project or model parent .venv."
    )


def _available_port() -> int:
    # 原因：8080 可能已被用户手工启动的服务占用，强行复用会切到错误模型。
    # 作用：让操作系统分配仅供本次本地 MLX 子进程使用的空闲端口。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _wait_for_server(
    process: subprocess.Popen[bytes],
    settings: SmolagentsModelSettings,
    *,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise ModelRuntimeError(f"Local MLX server exited during startup ({return_code}).")
        online, _ = check_model_connection(settings)
        if online:
            return
        time.sleep(1)
    raise ModelRuntimeError(f"Local MLX server did not become ready within {timeout_seconds}s.")


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _log_tail(path: Path, limit: int = 2000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:].strip()
