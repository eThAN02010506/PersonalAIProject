"""OpenAI-compatible model settings and endpoint discovery for smolagents."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from dotenv import load_dotenv

from qwopus_agent.llm import ModelCapabilities, ModelSettings

load_dotenv()

# 原因：旧名称曾被 API、测试和插件引用，直接删除会造成不必要的导入破坏。
# 作用：保留兼容名称，但真实配置类型只在 llm.config 中维护一份。
SmolagentsModelSettings = ModelSettings


def probe_model_settings(
    settings: SmolagentsModelSettings | None = None,
) -> tuple[SmolagentsModelSettings, bool, str]:
    """Probe once and return refreshed settings together with connection status."""
    settings = settings or SmolagentsModelSettings.from_env()
    try:
        status, payload = _request_models(settings)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", str(exc))
        return (
            settings,
            False,
            f"无法连接模型服务: {settings.base_url} ({reason})",
        )

    if not 200 <= status < 300:
        return settings, False, f"模型服务异常: {status}"

    model_entry = _extract_server_model_entry(payload)
    model_id = _extract_server_model_id(payload)
    resolved = settings
    if model_entry is not None and model_id:
        capabilities = _capabilities_from_server(
            model_entry,
            settings.capabilities,
        )
        # 原因：状态检测和模型发现读取的是同一个 /models 响应，分开调用会重复网络请求。
        # 作用：一次探测同步刷新模型 ID 与能力，降低局域网服务负载和瞬时误报概率。
        resolved = replace(
            settings,
            model_id=model_id,
            capabilities=capabilities,
        )
    return (
        resolved,
        True,
        f"模型服务在线: {settings.base_url} "
        f"(当前模型: {_display_model_name(resolved.model_id)})",
    )


def resolve_model_settings(
    settings: SmolagentsModelSettings | None = None,
) -> SmolagentsModelSettings:
    """Return settings updated with the model currently exposed by the server."""
    resolved, _online, _message = probe_model_settings(settings)
    return resolved


def check_model_connection(
    settings: SmolagentsModelSettings | None = None,
) -> tuple[bool, str]:
    """Probe the configured OpenAI-compatible model-list endpoint."""
    _resolved, online, message = probe_model_settings(settings)
    return online, message


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
    entry = _extract_server_model_entry(payload)
    if entry is not None:
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id:
            return model_id

        # 原因：部分兼容服务器返回 models[].model 或 models[].name，而不是 id。
        # 作用：模型实时发现不绑定某一种服务端响应形状。
        for key in ("model", "name"):
            model_id = entry.get(key)
            if isinstance(model_id, str) and model_id:
                return model_id
    return None


def _extract_server_model_entry(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("data", "models"):
        items = payload.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return items[0]
    return None


def _capabilities_from_server(
    entry: dict[str, Any],
    fallback: ModelCapabilities,
) -> ModelCapabilities:
    """Read optional non-standard capability metadata conservatively."""
    context_window = next(
        (
            int(entry[key])
            for key in ("context_window", "context_length", "max_model_len")
            if isinstance(entry.get(key), (int, float)) and int(entry[key]) >= 2048
        ),
        fallback.context_window_tokens,
    )
    return replace(
        fallback,
        context_window_tokens=context_window,
        supports_vision=bool(entry.get("supports_vision", fallback.supports_vision)),
        supports_structured_output=bool(
            entry.get(
                "supports_structured_output",
                fallback.supports_structured_output,
            )
        ),
    )


def _display_model_name(model_id: str) -> str:
    """Return a readable filename for Unix or Windows model paths."""
    return model_id.replace("\\", "/").rsplit("/", 1)[-1]
