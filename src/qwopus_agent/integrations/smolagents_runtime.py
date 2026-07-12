"""smolagents runtime integration for Qwopus-Agent.

This module connects Qwopus-Agent with an OpenAI-compatible local LLM server
such as optiq serve / mlx_lm.server.
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
    """Raised when smolagents is required but missing."""


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class SmolagentsModelSettings:
    """Configuration for OpenAI-compatible model server."""

    model_id: str

    base_url: str

    api_key: str = "sk-optiq-local"

    timeout_seconds: int = 120

    temperature: float = 0.2

    max_tokens: int = 1024

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.getenv(
                "QWOPUS_MLX_MODEL",
                "mlx-community/gemma-4-12B-it-qat-OptiQ-4bit"
            ),
            base_url=os.getenv(
                "QWOPUS_MLX_BASE_URL",
                "http://127.0.0.1:8080/v1"
            ),
            api_key=os.getenv(
                "QWOPUS_SMOLAGENTS_API_KEY",
                "sk-optiq-local"
            ),
            timeout_seconds=int(
                os.getenv(
                    "QWOPUS_SMOLAGENTS_TIMEOUT_SECONDS",
                    "120"
                )
            ),
            temperature=float(
                os.getenv(
                    "QWOPUS_SMOLAGENTS_TEMPERATURE",
                    "0.2"
                )
            ),
            max_tokens=int(
                os.getenv(
                    "QWOPUS_SMOLAGENTS_MAX_TOKENS",
                    "1024"
                )
            ),
        )


def _models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"



def check_model_connection(
    settings: SmolagentsModelSettings | None = None,
) -> tuple[bool, str]:

    settings = settings or SmolagentsModelSettings.from_env()

    request = urllib.request.Request(
        _models_endpoint(settings.base_url),
        headers={
            "Authorization": f"Bearer {settings.api_key}"
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=10,
        ) as response:

            if 200 <= response.status < 300:
                return True, f"模型服务在线: {settings.base_url}"

            return False, f"模型服务异常: {response.status}"


    except urllib.error.URLError as exc:
        return False, (
            f"无法连接模型服务: "
            f"{settings.base_url} ({exc.reason})"
        )



def format_chat_prompt(
    history: list[ChatMessage],
    user_message: str,
) -> str:

    lines = [
        "你是一个 helpful 的中文 AI 助手。",
        "",
        "历史对话:",
    ]

    for message in history:

        role = message.get("role")
        content = message.get("content")

        if role == "user":
            lines.append(
                f"用户: {content}"
            )

        elif role == "assistant":
            lines.append(
                f"助手: {content}"
            )


    lines.extend(
        [
            "",
            f"用户: {user_message}",
            "",
            "助手:",
        ]
    )

    return "\n".join(lines)



def build_smolagents_model(
    settings: SmolagentsModelSettings | None = None
) -> Any:

    settings = settings or SmolagentsModelSettings.from_env()

    try:
        from smolagents import OpenAIModel

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError(
            "smolagents is not installed"
        ) from exc


    return OpenAIModel(
        model_id=settings.model_id,
        api_base=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )



def build_smolagents_code_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
):

    try:
        from smolagents import CodeAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError(
            "Install smolagents first."
        ) from exc


    model = build_smolagents_model(settings)

    return CodeAgent(
        tools=tools or [],
        model=model,
        additional_authorized_imports=[
            "os",
            "pathlib",
            "glob",
            "json",
            "datetime",
            "re",
            "subprocess",
        ],
    )



def run_smolagents_smoke_test(
    prompt: str,
    settings: SmolagentsModelSettings | None = None,
):

    agent = build_smolagents_code_agent(
        settings=settings,
        tools=[],
    )

    return str(agent.run(prompt))



def run_smolagents_chat_turn(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
):

    prompt = format_chat_prompt(
        history,
        user_message,
    )

    return run_smolagents_smoke_test(
        prompt,
        settings,
    )