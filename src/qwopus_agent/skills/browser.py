"""Browser automation skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

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

    # 原因：浏览器可访问外部站点并执行交互，必须独立于普通 Skill 的默认权限。
    # 作用：后续接入真实 provider 时仍需本轮显式开启 browser 权限。
    agent_tool_permission: ClassVar[str | None] = "browser"

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
                    return SkillResponse(
                        success=False,
                        content="browser.open requires arguments.url.",
                    )
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
    # 原因：默认工厂若继续返回占位 Provider，Registry 虽能发现但永远无法真实执行。
    # 作用：延迟创建 Playwright Provider；只有调用 browser Skill 时才要求可选依赖。
    from qwopus_agent.integrations.playwright_browser import PlaywrightBrowserProvider

    return BrowserAutomationSkill(provider=PlaywrightBrowserProvider())
