"""smolagents adapter for the unified web-search Skill."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qwopus_agent.integrations.playwright_browser import PlaywrightBrowserProvider
from qwopus_agent.integrations.skill_tools import build_skill_tool
from qwopus_agent.integrations.tavily import TavilySearchConfig, TavilySearchProvider
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.browser import BrowserAutomationProvider, BrowserAutomationSkill
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


def build_browser_open_tool(
    provider: BrowserAutomationProvider | None = None,
    progress_callback: Callable[[str], None] | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    """Build a read-only smolagents Tool over the restricted Playwright provider."""
    skill = BrowserAutomationSkill(
        provider=provider or PlaywrightBrowserProvider(),
    )
    # 原因：模型不需要管理 open/snapshot 状态机，一次调用应直接得到渲染后的页面文本。
    # 作用：保留 Browser Skill 的 provider 边界，同时向 smolagents 暴露最小 URL schema。
    return build_skill_tool(
        skill,
        tool_name="browser_open",
        description=(
            "Open one public HTTP(S) page in an isolated browser and return its rendered "
            "title, URL, and visible text. Use this for JavaScript pages or a specific URL."
        ),
        inputs={
            "url": {
                "type": "string",
                "description": "The public HTTP(S) page URL to render.",
            }
        },
        request_factory=lambda values: SkillRequest(
            query=str(values["url"]),
            arguments={"action": "open", "url": str(values["url"])},
        ),
        max_output_tokens=max_output_tokens,
        progress_callback=progress_callback,
        start_phase="browsing",
    )
