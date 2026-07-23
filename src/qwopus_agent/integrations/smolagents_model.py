"""OpenAI-compatible model settings and endpoint discovery for smolagents."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SmolagentsModelSettings:
    """Configuration for an arbitrary OpenAI-compatible model server."""

    model_id: str
    base_url: str
    api_key: str = "sk-optiq-local"
    timeout_seconds: int = 120
    temperature: float = 0.2
    max_tokens: int = 1024
    context_window_tokens: int = 32768

    @classmethod
    def from_env(cls) -> SmolagentsModelSettings:
        return cls(
            model_id=os.getenv(
                "QWOPUS_MLX_MODEL",
                "gemma-4-12B-it-qat-OptiQ-4bit",
            ),
            base_url=os.getenv("QWOPUS_MLX_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("QWOPUS_SMOLAGENTS_API_KEY", "sk-optiq-local"),
            timeout_seconds=int(os.getenv("QWOPUS_SMOLAGENTS_TIMEOUT_SECONDS", "120")),
            temperature=float(os.getenv("QWOPUS_SMOLAGENTS_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("QWOPUS_SMOLAGENTS_MAX_TOKENS", "1024")),
            context_window_tokens=int(
                os.getenv("QWOPUS_SMOLAGENTS_CONTEXT_WINDOW_TOKENS", "32768")
            ),
        )


def resolve_model_settings(
    settings: SmolagentsModelSettings | None = None,
) -> SmolagentsModelSettings:
    """Return settings updated with the model currently exposed by the server."""
    settings = settings or SmolagentsModelSettings.from_env()
    try:
        status, payload = _request_models(settings)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return settings

    model_id = _extract_server_model_id(payload) if 200 <= status < 300 else None
    if not model_id:
        return settings

    # 原因：服务器加载的模型会变化，环境变量中的静态名称可能已经过期。
    # 作用：每次请求模型列表后使用实时 id，同时保留其他连接参数不变。
    return replace(settings, model_id=model_id)


def check_model_connection(
    settings: SmolagentsModelSettings | None = None,
) -> tuple[bool, str]:
    """Probe the configured OpenAI-compatible model-list endpoint."""
    settings = settings or SmolagentsModelSettings.from_env()
    try:
        status, payload = _request_models(settings)
        if 200 <= status < 300:
            model_id = _extract_server_model_id(payload) or settings.model_id
            return True, (
                f"模型服务在线: {settings.base_url} (当前模型: {_display_model_name(model_id)})"
            )
        return False, f"模型服务异常: {status}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", str(exc))
        return False, f"无法连接模型服务: {settings.base_url} ({reason})"


def _request_models(settings: SmolagentsModelSettings) -> tuple[int, dict[str, Any]]:
    """Request the OpenAI-compatible model list once."""
    request = urllib.request.Request(
        f"{settings.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {settings.api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload


def _extract_server_model_id(payload: dict[str, Any]) -> str | None:
    """Read a model id from common OpenAI-compatible response shapes."""
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        model_id = data[0].get("id")
        if isinstance(model_id, str) and model_id:
            return model_id

    models = payload.get("models")
    if isinstance(models, list) and models and isinstance(models[0], dict):
        for key in ("model", "name"):
            model_id = models[0].get(key)
            if isinstance(model_id, str) and model_id:
                return model_id
    return None


def _display_model_name(model_id: str) -> str:
    """Return a readable filename for Unix or Windows model paths."""
    return model_id.replace("\\", "/").rsplit("/", 1)[-1]
