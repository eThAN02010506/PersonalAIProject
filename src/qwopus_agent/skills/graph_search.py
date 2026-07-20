"""Independent Skill for bounded knowledge-graph path retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.memory.graph_backend import PersistentKnowledgeGraph
from qwopus_agent.memory.graph_extraction import RuleBasedGraphExtractor
from qwopus_agent.memory.graph_models import GraphPath
from qwopus_agent.memory.knowledge_graph import (
    DEFAULT_KNOWLEDGE_GRAPH_PATH,
    KnowledgeGraphIndex,
)
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class GraphSearchSkill(BaseSkill):
    """Search explicit entity relations without exposing graph storage internals."""

    name: str = "graph_search"
    description: str = (
        "Search persistent entity relations, multi-hop paths, and cross-document evidence."
    )
    index: KnowledgeGraphIndex | None = None

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Return bounded graph paths and their source evidence."""
        if self.index is None:
            return SkillResponse(
                success=False,
                content="graph_search requires a KnowledgeGraphIndex instance.",
            )
        max_hops = _bounded_integer(request.arguments.get("max_hops"), default=4, low=1, high=6)
        limit = _bounded_integer(request.arguments.get("limit"), default=5, low=1, high=20)
        paths = self.index.search(request.query, max_hops=max_hops, limit=limit)
        if not paths:
            return SkillResponse(
                success=True,
                content="No matching knowledge-graph path was found.",
                data={"paths": []},
            )
        return SkillResponse(
            success=True,
            content="\n\n".join(
                _render_path(path, number=index)
                for index, path in enumerate(paths, start=1)
            ),
            data={"paths": [path.model_dump(mode="json") for path in paths]},
        )


def create_skill() -> BaseSkill:
    """Factory discovered automatically by SkillRegistry."""
    # 原因：新增 Skill 必须做到零手工注册，并能读取同一个持久化全局图谱。
    # 作用：Registry 扫描本文件时直接构造可查询实例，生产环境仍可通过 override 注入。
    return GraphSearchSkill(
        index=KnowledgeGraphIndex(
            graph=PersistentKnowledgeGraph(DEFAULT_KNOWLEDGE_GRAPH_PATH),
            extractor=RuleBasedGraphExtractor(),
        )
    )


def _render_path(path: GraphPath, *, number: int) -> str:
    names_by_id = dict(zip(path.entity_ids, path.entity_names, strict=False))
    edges = [
        (
            f"{names_by_id[relation.source_id]} -[{relation.relation}]-> "
            f"{names_by_id[relation.target_id]}"
        )
        for relation in path.relations
    ]
    evidence_lines = []
    for evidence in path.evidence:
        citation = evidence.source
        if evidence.page is not None:
            citation += f", page {evidence.page}"
        evidence_lines.append(f"- [{citation}] {evidence.text}")
    return (
        f"## Graph Path {number}\n\n"
        + "\n".join(f"- {edge}" for edge in edges)
        + "\n\n### Evidence\n\n"
        + "\n".join(evidence_lines)
    )


def _bounded_integer(value: object, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(high, max(low, parsed))
