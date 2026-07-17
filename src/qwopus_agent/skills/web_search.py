"""Unified web search skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from qwopus_agent.integrations.smolagents_tools import TavilySearchConfig, build_tavily_search_tool
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class WebSearchProvider(Protocol):
    """Provider contract for external web search."""

    def search(self, query: str) -> list[str]:
        """Search external sources and return concise text results."""


@dataclass
class UnconfiguredWebSearchProvider:
    """Default provider used when no real search backend is configured."""

    def search(self, query: str) -> list[str]:
        """Fail clearly instead of pretending web search is available."""
        raise RuntimeError("web_search provider is not configured yet.")


@dataclass
class SmolagentsTavilySearchProvider:
    """Web search provider backed by a smolagents Tavily Tool."""

    max_results: int = 5

    def search(self, query: str) -> list[str]:
        """Search through Qwopus' smolagents Tavily tool adapter."""
        # 原因：smolagents 是 Agent 驱动入口，但 Tavily 才是稳定搜索 provider。
        # 作用：保留 Skill search(query) 接口，同时让执行路径经过 smolagents Tool。
        tool = build_tavily_search_tool(TavilySearchConfig(max_results=self.max_results))
        return _normalize_smolagents_search_results(tool(query))


@dataclass
class WebSearchSkill(BaseSkill):
    """Provide one search(query) capability for the Planner."""

    # Reason: Planner should choose web search without caring about the provider.
    name: str = "web_search"

    # Role: Unified external search capability; provider wiring comes later.
    description: str = "Search the web through a single provider-independent interface."

    provider: WebSearchProvider = field(default_factory=UnconfiguredWebSearchProvider)

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Search through the configured provider."""
        query = request.query.strip()
        if not query:
            return SkillResponse(success=False, content="web_search requires a non-empty query.")

        try:
            # 原因：Planner 只应该选择 web_search，不应该知道具体搜索服务。
            # 作用：把联网实现藏在 provider 后面，测试和生产可以替换不同后端。
            results = self.provider.search(query)
        except Exception as exc:
            return SkillResponse(
                success=False,
                content=str(exc),
                data={"query": query},
            )

        return SkillResponse(
            success=True,
            content="\n".join(results) if results else "No web search results.",
            data={"query": query, "results": results},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return WebSearchSkill()


def _normalize_smolagents_search_results(raw_result: Any) -> list[str]:
    """Normalize smolagents Tool output into text results."""
    if isinstance(raw_result, str):
        return [raw_result.strip()] if raw_result.strip() else []
    if isinstance(raw_result, list):
        return [str(item).strip() for item in raw_result if str(item).strip()]
    if raw_result is None:
        return []
    return [str(raw_result).strip()]
