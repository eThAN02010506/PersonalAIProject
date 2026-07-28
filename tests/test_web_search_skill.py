import asyncio
import unittest
from dataclasses import dataclass

from qwopus_agent.integrations.tavily import TavilySearchConfig, TavilySearchProvider
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.web_search import WebSearchSkill


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

    def test_tavily_provider_configuration_is_provider_owned(self) -> None:
        provider = TavilySearchProvider(TavilySearchConfig(max_results=3))

        # 原因：WebSearchSkill 不应反向依赖 smolagents Tool。
        # 作用：确认 Tavily 配置属于独立 Provider，Tool 只负责协议适配。
        self.assertEqual(provider.config.max_results, 3)


if __name__ == "__main__":
    unittest.main()
