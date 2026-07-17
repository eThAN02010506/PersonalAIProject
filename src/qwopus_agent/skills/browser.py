"""Browser automation skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class BrowserAutomationProvider(Protocol):
    """Provider contract for browser automation backends."""

    def open(self, url: str) -> str:
        """Open a URL and return a status message."""

    def snapshot(self) -> str:
        """Return a text snapshot of the current page."""


@dataclass
class UnconfiguredBrowserProvider:
    """Default provider used before a browser backend is attached."""

    def open(self, url: str) -> str:
        """Fail clearly instead of pretending browser automation is available."""
        raise RuntimeError("browser automation provider is not configured yet.")

    def snapshot(self) -> str:
        """Fail clearly instead of pretending browser automation is available."""
        raise RuntimeError("browser automation provider is not configured yet.")


@dataclass
class BrowserAutomationSkill(BaseSkill):
    """Expose browser automation through one Skill."""

    name: str = "browser"

    description: str = "Open pages and inspect browser snapshots through a provider."

    provider: BrowserAutomationProvider = field(default_factory=UnconfiguredBrowserProvider)

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Run one browser action."""
        action = str(request.arguments.get("action", "snapshot"))
        try:
            if action == "open":
                url = str(request.arguments.get("url", "")).strip()
                if not url:
                    return SkillResponse(success=False, content="browser.open requires arguments.url.")
                # 原因：浏览器自动化具体实现可能来自 Playwright、Chrome 插件或桌面工具。
                # 作用：Skill 只处理动作路由，避免核心 Agent 绑定某个浏览器后端。
                content = self.provider.open(url)
            elif action == "snapshot":
                content = self.provider.snapshot()
            else:
                return SkillResponse(success=False, content=f"Unsupported browser action: {action}")
        except Exception as exc:
            return SkillResponse(success=False, content=str(exc), data={"action": action})

        return SkillResponse(success=True, content=content, data={"action": action})


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return BrowserAutomationSkill()
