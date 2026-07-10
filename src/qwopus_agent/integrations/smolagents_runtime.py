"""smolagents runtime integration for Qwopus-Agent.

This module is intentionally isolated from Planner, Executor, and Skills. smolagents is an external
agent runtime, so keeping it here lets Qwopus-Agent swap or remove it without changing core logic.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class SmolagentsDependencyError(RuntimeError):
    """Raised when smolagents is required but not installed."""


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class SmolagentsModelSettings:
    """Configuration for connecting smolagents to an OpenAI-compatible model server."""

    # 原因：模型名必须来自配置，不能写死在 Agent 逻辑里。
    # 作用：传给 smolagents model，让任意可用模型都能替换当前 Gemma。
    model_id: str

    # 原因：本地 MLX、远程 API、其他 OpenAI-compatible 服务只需要换 endpoint。
    # 作用：指向 `/v1` 形式的 OpenAI-compatible base URL。
    base_url: str

    # 原因：部分服务需要 key，本地 MLX 可以使用占位 token。
    # 作用：传给 smolagents/HF InferenceClient 的 api_key 参数。
    api_key: str = "local_token"

    # 原因：本地大模型可能响应较慢，需要显式超时配置。
    # 作用：限制 smolagents 请求等待时间。
    timeout_seconds: int = 120

    # 原因：第一阶段只测试对话连通性，默认保持低随机性。
    # 作用：控制模型采样温度。
    temperature: float = 0.2

    # 原因：smoke test 不需要长输出，避免本地推理时间过长。
    # 作用：限制单次模型输出 token 数。
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> SmolagentsModelSettings:
        """Create settings from environment variables."""
        return cls(
            model_id=os.getenv("QWOPUS_MLX_MODEL", "gemma-4-12B-it-qat-OptiQ-4bit"),
            base_url=os.getenv("QWOPUS_MLX_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.getenv("QWOPUS_SMOLAGENTS_API_KEY", "local_token"),
            timeout_seconds=int(os.getenv("QWOPUS_SMOLAGENTS_TIMEOUT_SECONDS", "120")),
            temperature=float(os.getenv("QWOPUS_SMOLAGENTS_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("QWOPUS_SMOLAGENTS_MAX_TOKENS", "1024")),
        )


def _models_endpoint(base_url: str) -> str:
    """Build the OpenAI-compatible /v1/models probe URL."""
    return f"{base_url.rstrip('/')}/models"


def check_model_connection(settings: SmolagentsModelSettings | None = None) -> tuple[bool, str]:
    """Probe the configured OpenAI-compatible model server for availability."""
    settings = settings or SmolagentsModelSettings.from_env()
    request = urllib.request.Request(
        _models_endpoint(settings.base_url),
        headers={"Authorization": f"Bearer {settings.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if 200 <= response.status < 300:
                return True, f"模型服务在线：{settings.base_url}"
            return False, f"模型服务返回异常状态码：{response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 404, 405}:
            return True, f"模型服务在线：{settings.base_url}"
        return False, f"模型服务不可用（HTTP {exc.code}）：{settings.base_url}"
    except urllib.error.URLError as exc:
        return False, f"无法连接模型服务：{settings.base_url}（{exc.reason}）"
    except TimeoutError:
        return False, f"连接模型服务超时：{settings.base_url}"


def format_chat_prompt(history: list[ChatMessage], user_message: str) -> str:
    """Format chat history and the latest user message into one prompt."""
    role_labels = {
        "user": "用户",
        "assistant": "助手",
    }
    lines = [
        "你是一个 helpful 的中文对话助手。请根据以下对话历史，直接回答用户最后的问题。",
        "",
        "对话历史：",
    ]
    for message in history:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        label = role_labels.get(role, role)
        lines.append(f"{label}：{content}")
    lines.extend(["", f"用户：{user_message.strip()}", "", "助手："])
    return "\n".join(lines)


def build_smolagents_model(settings: SmolagentsModelSettings | None = None) -> Any:
    """Build a smolagents model connected to the configured local LLM server."""
    settings = settings or SmolagentsModelSettings.from_env()
    try:
        from smolagents import InferenceClientModel
    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError(
            "smolagents is not installed. Run: pip install -e '.[dev]'"
        ) from exc

    # 原因：官方 smolagents 当前支持 InferenceClientModel(base_url=...) 连接本地 endpoint。
    # 作用：把 mlx_lm.server 暴露的 OpenAI-compatible API 接入 smolagents。
    return InferenceClientModel(
        model_id=settings.model_id,
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_seconds,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def build_smolagents_code_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Build a minimal smolagents CodeAgent for local model smoke testing."""
    try:
        from smolagents import CodeAgent
    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError(
            "smolagents is not installed. Run: pip install -e '.[dev]'"
        ) from exc

    model = build_smolagents_model(settings)

    # 原因：第一阶段只验证大模型对话连通性，不提前接 Excel、MiniRAG、Web 工具。
    # 作用：创建无工具 CodeAgent，后续再逐个注入 Skill/Tool。
    return CodeAgent(tools=tools or [], model=model)


def run_smolagents_smoke_test(prompt: str, settings: SmolagentsModelSettings | None = None) -> str:
    """Run one prompt through smolagents and return the text result."""
    agent = build_smolagents_code_agent(settings=settings, tools=[])
    result = agent.run(prompt)
    return str(result)


def run_smolagents_chat_turn(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
) -> str:
    """Run one chat turn through a tool-less smolagents CodeAgent."""
    prompt = format_chat_prompt(history=history, user_message=user_message)
    return run_smolagents_smoke_test(prompt, settings=settings)
