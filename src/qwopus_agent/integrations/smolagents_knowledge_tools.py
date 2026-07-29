"""smolagents adapters for semantic and graph knowledge Skills."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from qwopus_agent.integrations.skill_tools import build_skill_tool
from qwopus_agent.memory.knowledge_store import KnowledgeStore
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.graph_search import GraphSearchSkill
from qwopus_agent.skills.rag_search import RagSearchSkill
from qwopus_agent.utils.token_budget import TokenBudgetManager

if TYPE_CHECKING:
    from qwopus_agent.memory.knowledge_graph import KnowledgeGraphIndex


def build_minirag_search_tool(
    minirag: KnowledgeStore,
    min_relevance: float = 0.55,
    max_results: int = 12,
    source_hints: Sequence[str] = (),
    budget_manager: TokenBudgetManager | None = None,
    progress_callback: Callable[[str], None] | None = None,
    tool_name: str = "rag_search",
    description: str | None = None,
) -> Any:
    """Expose the knowledge adapter's search contract as a bounded Tool."""
    budget = budget_manager or TokenBudgetManager()
    return build_skill_tool(
        RagSearchSkill(minirag=minirag),
        tool_name=tool_name,
        inputs={
            "query": {
                "type": "string",
                "description": "Semantic search query for the local knowledge base.",
            }
        },
        request_factory=lambda values: SkillRequest(
            query=_search_query(str(values["query"]), source_hints),
            arguments={
                "min_relevance": min_relevance,
                "max_results": max_results,
            },
        ),
        description=description
        or (
            "Search documents uploaded in the current conversation through MiniRAG. "
            "Use this only when this conversation's prior files may help answer the question."
        ),
        max_output_tokens=budget.observation_budget,
        progress_callback=progress_callback,
        start_phase="retrieving",
    )


def _search_query(query: str, source_hints: Sequence[str]) -> str:
    """Preserve user-named source constraints when the model rewrites a query."""
    if not source_hints:
        return query
    # 原因：不同模型生成 Tool 参数时可能丢掉用户明确写出的文件名。
    # 作用：将已授权来源作为 metadata 提示附回查询，避免正确文件被相关性阈值漏掉。
    return f"{query}\nUser-named sources: {', '.join(source_hints)}"


def build_graph_search_tool(
    index: KnowledgeGraphIndex,
    max_hops: int = 4,
    max_results: int = 5,
    budget_manager: TokenBudgetManager | None = None,
    progress_callback: Callable[[str], None] | None = None,
    tool_name: str = "graph_search",
    description: str | None = None,
) -> Any:
    """Expose bounded persistent graph traversal as a smolagents Tool."""
    budget = budget_manager or TokenBudgetManager()
    return build_skill_tool(
        GraphSearchSkill(index=index),
        tool_name=tool_name,
        inputs={
            "query": {
                "type": "string",
                "description": "A relationship or graph-path question containing entity names.",
            }
        },
        request_factory=lambda values: SkillRequest(
            query=str(values["query"]),
            arguments={"max_hops": max_hops, "limit": max_results},
        ),
        description=description
        or (
            "Search explicit entity relationships, cross-document evidence, and multi-hop "
            "paths in the current conversation's knowledge graph. Use this instead of "
            "rag_search when the question asks how named entities are related."
        ),
        max_output_tokens=budget.observation_budget,
        progress_callback=progress_callback,
        start_phase="retrieving",
    )
