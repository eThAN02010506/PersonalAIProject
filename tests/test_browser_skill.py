import asyncio
import unittest
from dataclasses import dataclass, field

from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.browser import BrowserAutomationSkill


@dataclass
class FakeBrowserProvider:
    opened: list[str] = field(default_factory=list)

    def open(self, url: str) -> str:
        self.opened.append(url)
        return f"opened {url}"

    def snapshot(self) -> str:
        return "snapshot text"


class BrowserAutomationSkillTests(unittest.TestCase):
    def test_browser_skill_uses_injected_provider(self) -> None:
        provider = FakeBrowserProvider()
        skill = BrowserAutomationSkill(provider=provider)

        # 原因：浏览器自动化需要可替换后端，不能把 Agent 绑死在某个 UI 工具上。
        # 作用：验证 Skill 通过 provider 执行 open/snapshot 两类基础动作。
        opened = asyncio.run(
            skill.run(SkillRequest(query="open", arguments={"action": "open", "url": "http://localhost"}))
        )
        snapshot = asyncio.run(skill.run(SkillRequest(query="snapshot")))

        self.assertTrue(opened.success)
        self.assertEqual(provider.opened, ["http://localhost"])
        self.assertEqual(snapshot.content, "snapshot text")

    def test_browser_skill_fails_clearly_without_provider(self) -> None:
        response = asyncio.run(BrowserAutomationSkill().run(SkillRequest(query="snapshot")))

        self.assertFalse(response.success)
        self.assertIn("not configured", response.content)


if __name__ == "__main__":
    unittest.main()
