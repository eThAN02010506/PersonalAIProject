"""Direct non-tool chat path for smolagents-compatible models."""

from __future__ import annotations

from qwopus_agent.integrations import smolagents_factory, smolagents_model
from qwopus_agent.prompts import smolagents as smolagents_prompts

ChatMessage = smolagents_prompts.ChatMessage
SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings


def run_smolagents_chat_turn(
    user_message: str,
    history: list[ChatMessage],
    settings: SmolagentsModelSettings | None = None,
) -> str:
    """Run one direct chat turn without Agent tools."""
    model = smolagents_factory.build_smolagents_model(settings)
    response = model.generate(
        smolagents_prompts.build_chat_messages(history, user_message),
        max_tokens=(settings or SmolagentsModelSettings.from_env()).max_tokens,
    )

    # 原因：不同 smolagents 版本返回 ChatMessage 对象或 dict-like 结构。
    # 作用：把返回值统一成前端和调试台可展示的纯文本。
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(response, dict):
        return str(response.get("content", response))
    return str(response)
