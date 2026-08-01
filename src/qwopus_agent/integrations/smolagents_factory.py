"""Factory helpers for constructing smolagents runtimes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qwopus_agent.integrations import smolagents_model

SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings


class SmolagentsDependencyError(RuntimeError):
    """Raised when smolagents is required but missing."""


def build_smolagents_model(settings: SmolagentsModelSettings | None = None) -> Any:
    """Build the OpenAI-compatible model adapter used by smolagents."""
    settings = settings or SmolagentsModelSettings.from_env()

    try:
        from smolagents import OpenAIModel

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("smolagents is not installed") from exc

    return OpenAIModel(
        model_id=settings.model_id,
        api_base=settings.base_url,
        api_key=settings.api_key,
        client_kwargs={
            "timeout": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        },
        temperature=settings.temperature,
        # 原因：smolagents Agent.run 不会把应用层 max_tokens 自动传给模型。
        # 作用：长文档报告使用显式输出预算，不再由兼容服务的短默认值截成“略”。
        max_tokens=settings.max_tokens,
    )


def build_smolagents_code_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
    final_answer_checks: list[Callable[..., bool]] | None = None,
    planning_interval: int | None = None,
) -> Any:
    """Build a CodeAgent with only registered Tool access."""
    try:
        from smolagents import CodeAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

    model = build_smolagents_model(settings)

    return CodeAgent(
        tools=tools or [],
        model=model,
        # 原因：聊天 Tool 不需要任意文件、进程或 shell 访问；授权 os/subprocess 会扩大风险。
        # 作用：Code 兼容模式只能组合已注册 Tool，数据计算继续使用独立 pandas 沙箱。
        additional_authorized_imports=[],
        final_answer_checks=final_answer_checks,
        planning_interval=planning_interval,
    )


def build_smolagents_tool_calling_agent(
    settings: SmolagentsModelSettings | None = None,
    tools: list[Any] | None = None,
    final_answer_checks: list[Callable[..., bool]] | None = None,
    planning_interval: int | None = None,
) -> Any:
    """Build the smolagents Agent runtime used as Qwopus' chat driver."""
    try:
        from smolagents import ToolCallingAgent

    except ModuleNotFoundError as exc:
        raise SmolagentsDependencyError("Install smolagents first.") from exc

    settings = settings or SmolagentsModelSettings.from_env()
    if settings.capabilities.agent_mode == "code":
        return build_smolagents_code_agent(
            settings=settings,
            tools=tools,
            final_answer_checks=final_answer_checks,
            planning_interval=planning_interval,
        )

    model = build_smolagents_model(settings)
    # 原因：smolagents 是整体 Agent 驱动入口，工具选择应由 Agent runtime 处理。
    # 作用：Streamlit 不再手动先搜索再拼 prompt，而是把受控 Tool 交给 Agent。
    return ToolCallingAgent(
        tools=tools or [],
        model=model,
        final_answer_checks=final_answer_checks,
        planning_interval=planning_interval,
    )


def run_smolagents_smoke_test(
    prompt: str,
    settings: SmolagentsModelSettings | None = None,
) -> str:
    """Run a minimal connectivity check through the CodeAgent path."""
    agent = build_smolagents_code_agent(
        settings=settings,
        tools=[],
    )

    return str(agent.run(prompt))
