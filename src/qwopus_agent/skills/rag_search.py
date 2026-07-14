"""RAG search skill."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.memory.minirag import MiniRAG
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class RagSearchSkill(BaseSkill):
    """Search the knowledge layer through the MiniRAG facade."""

    # Reason: Planner should see retrieval as one capability, not the internals of memory storage.
    name: str = "rag_search"

    # Role: Answers queries by delegating only to MiniRAG.search(query).
    description: str = "Search local knowledge through the MiniRAG interface."

    minirag: MiniRAG | None = None

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Search MiniRAG without exposing internal retrieval implementation."""
        if self.minirag is None:
            return SkillResponse(
                success=False,
                content="rag_search requires a MiniRAG instance.",
            )

        results = self.minirag.search(request.query)
        return SkillResponse(
            success=True,
            content="\n".join(results),
            data={"results": results},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    # 原因：自动发现的 rag_search 也应该开箱可用，而不是必须由 UI 手动注入依赖。
    # 作用：默认连接持久化 MiniRAG facade；测试仍可直接构造 RagSearchSkill 注入临时实例。
    return RagSearchSkill(minirag=MiniRAG())
