import asyncio
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.web_search import SmolagentsTavilySearchProvider, WebSearchSkill


@dataclass
class FakeWebSearchProvider:
    def search(self, query: str) -> list[str]:
        return [f"result for {query}", "second result"]


class WebSearchSkillTests(unittest.TestCase):
    def test_web_search_skill_uses_injected_provider(self) -> None:
        skill = WebSearchSkill(provider=FakeWebSearchProvider())

        # 原因：web_search 要暴露统一 search(query)，但不能绑定具体搜索服务。
        # 作用：验证 Skill 能通过依赖注入使用任意 provider。
        response = asyncio.run(skill.run(SkillRequest(query="Qwopus Agent")))

        self.assertTrue(response.success)
        self.assertIn("result for Qwopus Agent", response.content)
        self.assertEqual(response.data["results"], ["result for Qwopus Agent", "second result"])

    def test_web_search_skill_fails_clearly_without_provider(self) -> None:
        response = asyncio.run(WebSearchSkill().run(SkillRequest(query="Qwopus Agent")))

        self.assertFalse(response.success)
        self.assertIn("not configured", response.content)

    def test_smolagents_provider_uses_tavily_tool_adapter(self) -> None:
        calls = {}

        class FakeTool:
            def __call__(self, query: str) -> str:
                calls["query"] = query
                return "rice cooking result"

        def fake_build_tool(config):
            calls["max_results"] = config.max_results
            return FakeTool()

        provider = SmolagentsTavilySearchProvider(max_results=3)

        # 原因：联网搜索应由 smolagents Tool 驱动，但搜索 provider 使用 Tavily。
        # 作用：验证 Skill provider 经由 Tavily Tool adapter，而不是直接请求外部 API。
        with patch("qwopus_agent.skills.web_search.build_tavily_search_tool", side_effect=fake_build_tool):
            results = provider.search("how to cook rice")

        self.assertEqual(results, ["rice cooking result"])
        self.assertEqual(calls, {"max_results": 3, "query": "how to cook rice"})

    def test_smolagents_provider_normalizes_list_results(self) -> None:
        class FakeTool:
            def __call__(self, query: str) -> list[str]:
                return ["first", "second"]

        def fake_build_tool(config):
            return FakeTool()

        with patch("qwopus_agent.skills.web_search.build_tavily_search_tool", side_effect=fake_build_tool):
            results = SmolagentsTavilySearchProvider().search("Qwopus")

        self.assertEqual(results, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
