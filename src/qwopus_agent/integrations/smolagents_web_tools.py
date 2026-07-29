"""smolagents adapter for the unified web-search Skill."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qwopus_agent.integrations.skill_tools import build_skill_tool
from qwopus_agent.integrations.tavily import TavilySearchConfig, TavilySearchProvider
from qwopus_agent.skills.web_search import WebSearchSkill


def build_tavily_search_tool(
    config: TavilySearchConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    """Build a smolagents Tool that searches Tavily."""
    resolved_config = config or TavilySearchConfig()
    skill = WebSearchSkill(
        provider=TavilySearchProvider(
            config=resolved_config,
            progress_callback=progress_callback,
        )
    )
    # 原因：联网业务只能存在于 WebSearchSkill，Tool 不应再复制 HTTP 和格式化逻辑。
    # 作用：正式 Agent、Planner/Executor 和测试共享同一个 Tavily Provider。
    return build_skill_tool(
        skill,
        tool_name="tavily_search",
        description=(
            "Search the live web with Tavily when current external information is needed. "
            "Output is concise Markdown evidence."
        ),
        inputs={"query": {"type": "string", "description": "The web search query."}},
    )
