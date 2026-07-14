"""smolagents runtime integration for Qwopus-Agent.

This module connects Qwopus-Agent with an OpenAI-compatible local LLM server
such as optiq serve / mlx_lm.server.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
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
                "gemma-4-12B-it-qat-OptiQ-4bit",
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


@dataclass(frozen=True)
class DocumentAnalysisRun:
    """Document analysis answer with a UI-visible debug trace."""

    answer: str

    debug_steps: list[str]


def _models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


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

    # 原因：服务器加载的模型会变化，.env 中的静态名称可能已经过期。
    # 作用：每次请求模型列表后使用实时 id，同时保留其他连接参数不变。
    return replace(settings, model_id=model_id)


def check_model_connection(
        settings: SmolagentsModelSettings | None = None,
) -> tuple[bool, str]:
    settings = settings or SmolagentsModelSettings.from_env()

    try:
        status, payload = _request_models(settings)
        if 200 <= status < 300:
            model_id = _extract_server_model_id(payload) or settings.model_id
            return True, (
                f"模型服务在线: {settings.base_url} "
                f"(当前模型: {_display_model_name(model_id)})"
            )
        return False, f"模型服务异常: {status}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", str(exc))
        return False, (
            f"无法连接模型服务: "
            f"{settings.base_url} ({reason})"
        )


def _request_models(settings: SmolagentsModelSettings) -> tuple[int, dict[str, Any]]:
    """Request the OpenAI-compatible model list once."""
    request = urllib.request.Request(
        _models_endpoint(settings.base_url),
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


def format_chat_prompt(
        history: list[ChatMessage],
        user_message: str,
) -> str:
    lines = [
        """
    You are a CodeAgent.

    Always answer using:

    Thought: ...

    final_answer

    Never output plain text.
    """,
        "",
        "历史对话:",
    ]

    for message in history:

        role = message.get("role")
        content = message.get("content")

        if role == "user":
            lines.append(
                f"用户：{content}"
            )

        elif role == "assistant":
            lines.append(
                f"助手：{content}"
            )

    lines.extend(
        [
            "",
            f"用户：{user_message}",
            "",
            "助手：",
        ]
    )

    return "\n".join(lines)


def build_chat_messages(
        history: list[ChatMessage],
        user_message: str,
) -> list[ChatMessage]:
    """Build plain chat messages for direct model generation."""
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": (
                "你是 Qwopus-Agent 的本地办公助手。"
                "请直接、清晰地回答用户问题。"
                # 原因：普通聊天只收到对话历史，无法读取上传文件或 MiniRAG 内容。
                # 作用：禁止模型假装分析旧文件，并引导用户使用文档分析流程。
                "普通聊天无法自动访问用户之前上传的文件或 MiniRAG。"
                "如果用户要求分析未附带内容的历史文件，请明确说明无法读取，"
                "并请用户在文档分析页面重新选择文件；不要编造文件内容。"
                "不要输出 Thought、代码块或 final_answer 包装，除非用户明确要求。"
            ),
        }
    ]

    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            # 原因：普通对话要保留上下文，但不应该把 Streamlit 内部状态泄漏给模型。
            # 作用：只传递模型需要理解上下文的 role/content。
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


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
    model = build_smolagents_model(settings)
    response = model.generate(
        build_chat_messages(history, user_message),
        max_tokens=(settings or SmolagentsModelSettings.from_env()).max_tokens,
    )

    # 原因：不同 smolagents 版本返回 ChatMessage 对象或 dict-like 结构。
    # 作用：把返回值统一成 Streamlit 可展示的纯文本。
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(response, dict):
        return str(response.get("content", response))
    return str(response)


def run_smolagents_document_analysis(
        document_name: str,
        content: str,
        user_question: str,
        settings: SmolagentsModelSettings | None = None,
        max_context_chars: int = 16000,
) -> str:
    """Ask the configured model to analyze a parsed document or spreadsheet summary."""
    return run_smolagents_document_analysis_with_debug(
        document_name=document_name,
        content=content,
        user_question=user_question,
        settings=settings,
        max_context_chars=max_context_chars,
    ).answer


def run_smolagents_document_analysis_with_debug(
        document_name: str,
        content: str,
        user_question: str,
        settings: SmolagentsModelSettings | None = None,
        max_context_chars: int = 16000,
) -> DocumentAnalysisRun:
    """Ask the model to analyze a parsed document and return traceable steps."""
    debug_steps: list[str] = []
    question = user_question.strip() or "请总结这份文件的主要内容。"
    clipped_content = content[:max_context_chars]
    if len(content) > max_context_chars:
        clipped_content += "\n\n[内容过长，已截断。后续将接入 MiniRAG 处理长文档。]"
        debug_steps.append(f"文档内容超过 {max_context_chars} 字符，已截断后发送给模型。")
    else:
        debug_steps.append(f"文档内容长度 {len(content)} 字符，未截断。")

    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": (
                "你是 Qwopus-Agent 的文件分析助手。"
                "请严格基于用户上传文件的解析内容回答，不要编造文件里没有的信息。"
                "如果内容不足以回答，请明确说明缺少什么。"
                "回答语言必须跟随用户问题的语言；如果用户问题语言不明确，再跟随文件主要语言。"
                "请输出结构化分析。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"文件名：{document_name}\n\n"
                f"用户问题：{question}\n\n"
                f"文件解析内容如下：\n\n{clipped_content}"
            ),
        },
    ]

    debug_steps.append(f"准备调用模型：{settings.model_id if settings else SmolagentsModelSettings.from_env().model_id}")
    debug_steps.append(f"用户问题：{question}")
    output_token_limit = _document_output_token_limit(settings)
    model = build_smolagents_model(settings)
    response = model.generate(messages, max_tokens=output_token_limit)
    answer = _response_to_text(response)
    debug_steps.append(f"第一次模型返回结构：{_response_debug_snapshot(response)}")
    debug_steps.append(f"第一次模型返回前 500 字：{answer[:500]}")
    if not answer.strip() or _response_finished_by_length(response):
        # 原因：推理模型可能把 token 用在 reasoning_content，导致 content 为空或被截断。
        # 作用：增加输出预算并强制要求只返回最终答案，避免把思考草稿展示给用户。
        debug_steps.append("第一次模型未生成完整 content，触发第二轮重试。")
        response = model.generate(
            [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        "请直接给出最终答案，不要输出思考、推理过程、工具过程或草稿。"
                        "回答语言必须跟随用户问题的语言；如果用户问题语言不明确，再跟随文件主要语言。\n\n"
                        f"文件名：{document_name}\n\n"
                        f"用户问题：{question}\n\n"
                        f"文件解析内容如下：\n\n{clipped_content}\n\n"
                        "请生成较完整的分析，不要只返回简短 bullet point。"
                        "请包含：整体概览、关键内容、重要数据或结论、可执行建议。"
                        "只有当用户明确要求简短时，才使用简短要点。"
                    ),
                },
            ],
            max_tokens=output_token_limit * 2,
        )
        answer = _response_to_text(response)
        debug_steps.append(f"第二次模型返回结构：{_response_debug_snapshot(response)}")
        debug_steps.append(f"第二次模型返回前 500 字：{answer[:500]}")
    elif _looks_like_tool_observation(answer):
        # 原因：模型有时会停在工具 Observation 阶段。
        # 作用：明确追加一轮，让模型把 Observation 转成用户可读的最终答案。
        debug_steps.append("检测到 Observation/Document Analysis/Preview，触发第二轮最终答案生成。")
        response = model.generate(
            [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "上一步只是工具 Observation。请继续推理，基于 Observation 生成最终答案。"
                        "只输出给用户看的自然语言总结，不要输出 Observation、Document Analysis、Preview 或工具原文。"
                    ),
                },
            ]
        )
        answer = _response_to_text(response)
        debug_steps.append(f"第二次模型返回结构：{_response_debug_snapshot(response)}")
        debug_steps.append(f"第二次模型返回前 500 字：{answer[:500]}")
    else:
        debug_steps.append("第一次返回不像工具 Observation，直接进入 final_answer 提取。")

    final_answer = _extract_final_answer(answer)
    if final_answer != answer.strip():
        debug_steps.append("检测到 final_answer(...) 包装，已提取内部答案。")
    else:
        debug_steps.append("未检测到 final_answer(...) 包装，使用模型返回文本作为最终答案。")

    if _looks_like_tool_observation(final_answer):
        debug_steps.append("警告：最终答案仍像工具 Observation，说明模型第二轮没有完成总结。")
        # 原因：用户主界面不能展示工具 Observation 或原始解析预览。
        # 作用：最多再做一次强制收敛，把工具输出转换成自然语言最终答案。
        response = model.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "你只负责输出给最终用户看的答案。"
                        "回答语言必须跟随用户问题的语言；如果用户问题语言不明确，再跟随文件主要语言。"
                        "禁止输出 Observation、Document Analysis、Preview、工具日志或草稿。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n\n"
                        f"工具观察内容：\n{final_answer}\n\n"
                        "请基于上面的内容生成完整最终答案。"
                    ),
                },
            ],
            max_tokens=output_token_limit * 2,
        )
        final_answer = _extract_final_answer(_response_to_text(response))
        debug_steps.append(f"第三次模型返回结构：{_response_debug_snapshot(response)}")
        debug_steps.append(f"第三次模型返回前 500 字：{final_answer[:500]}")

    return DocumentAnalysisRun(answer=final_answer, debug_steps=debug_steps)


def _response_to_text(response: Any) -> str:
    """Normalize smolagents model responses to text."""
    content_value = getattr(response, "content", None)
    if isinstance(content_value, str) and content_value:
        return content_value
    if isinstance(response, dict):
        return str(response.get("content", response))
    raw_text = _extract_text_from_raw_response(response)
    if raw_text:
        return raw_text
    if isinstance(content_value, str):
        return content_value
    return str(response)


def _extract_text_from_raw_response(response: Any) -> str:
    """Read fallback text from OpenAI-compatible raw responses."""
    raw = getattr(response, "raw", None)
    choices = getattr(raw, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    for source in (message, first_choice):
        if source is None:
            continue
        for field_name in ("content", "text"):
            value = getattr(source, field_name, None)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _document_output_token_limit(settings: SmolagentsModelSettings | None) -> int:
    """Use a larger answer budget for document analysis."""
    configured_limit = settings.max_tokens if settings else SmolagentsModelSettings.from_env().max_tokens
    return max(configured_limit, 2048)


def _response_finished_by_length(response: Any) -> bool:
    """Detect truncated OpenAI-compatible responses."""
    raw = getattr(response, "raw", None)
    choices = getattr(raw, "choices", None)
    if not choices:
        return False
    return getattr(choices[0], "finish_reason", None) == "length"


def _response_debug_snapshot(response: Any) -> str:
    """Return compact response metadata for Streamlit debugging."""
    raw = getattr(response, "raw", None)
    choices = getattr(raw, "choices", None)
    content_value = getattr(response, "content", None)
    parts = [
        f"type={type(response).__name__}",
        f"content_len={len(content_value) if isinstance(content_value, str) else 'n/a'}",
        f"has_raw={raw is not None}",
    ]
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        parts.append(f"finish_reason={getattr(first_choice, 'finish_reason', None)}")
        parts.append(f"tool_calls={getattr(message, 'tool_calls', None) if message else None}")
        parts.append(f"raw_content={repr(getattr(message, 'content', None)) if message else None}")
        parts.append(
            f"reasoning_content={repr(getattr(message, 'reasoning_content', None)) if message else None}"
        )
    return ", ".join(parts)


def _looks_like_tool_observation(text: str) -> bool:
    """Detect model output that is still exposing tool observations."""
    lowered = text.lower()
    return "observation:" in lowered or "document analysis:" in lowered or "## preview" in lowered


def _extract_final_answer(text: str) -> str:
    """Extract final_answer(...) when a CodeAgent-style answer leaks through."""
    match = re.search(r"final_answer\((?P<quote>['\"])(?P<answer>.*?)(?P=quote)\)", text, re.S)
    if match:
        return match.group("answer").strip()
    return text.strip()
