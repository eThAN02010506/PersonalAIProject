"""RAG search skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from qwopus_agent.memory.knowledge_store import KnowledgeStore
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class RagSearchSkill(BaseSkill):
    """Search the knowledge layer through the MiniRAG facade."""

    # 原因：本地知识可能包含其他会话或全局文档，自动暴露会破坏作用域授权。
    # 作用：只有运行时注入当前获准 KnowledgeStore 后才允许构建 Agent Tool。
    agent_tool_permission: ClassVar[str | None] = "knowledge"

    # Reason: Planner should see retrieval as one capability, not the internals of memory storage.
    name: str = "rag_search"

    # Role: Answers queries by delegating only to MiniRAG.search(query).
    description: str = "Search local knowledge through the MiniRAG interface."

    # 原因：Skill 只需要 insert/search 契约，不应依赖 NanoVectorDB 或项目图谱实现。
    # 作用：MiniRAG 适配器和测试替身都可通过依赖注入使用同一个检索 Skill。
    minirag: KnowledgeStore | None = None

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Search MiniRAG without exposing internal retrieval implementation."""
        if self.minirag is None:
            return SkillResponse(
                success=False,
                content="rag_search requires a MiniRAG instance.",
            )

        min_relevance = float(request.arguments.get("min_relevance", 0.25))
        max_results = max(1, min(20, int(request.arguments.get("max_results", 12))))
        results = self.minirag.search(
            request.query,
            min_relevance=min_relevance,
        )[:max_results]
        content = (
            "\n\n".join(
                f"## MiniRAG Result {index}\n\n{result}"
                for index, result in enumerate(results, start=1)
            )
            if results
            else "No relevant MiniRAG results."
        )
        return SkillResponse(
            # 原因：空检索结果不是可用于回答的证据，标记成功会让上层继续自由生成。
            # 作用：直接 Skill 调用与 smolagents Tool 适配器都能统一识别“无证据”。
            success=bool(results),
            content=content,
            data={"results": results},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    # 原因：扫描 Skill 目录不应加载 Torch、embedding 模型和持久化索引。
    # 作用：Registry 仍可零手工发现名称，生产装配时再通过 override 注入共享 MiniRAG。
    return RagSearchSkill()
