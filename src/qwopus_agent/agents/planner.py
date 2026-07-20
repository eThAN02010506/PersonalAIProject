"""Planner Agent.

Planner is responsible only for turning a user objective into an execution plan. It never calls
skills, reads files, or performs side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qwopus_agent.skills import SkillRegistry

SPREADSHEET_EXTENSIONS = {".csv", ".xls", ".xlsx"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".png", ".jpeg", ".jpg"}


@dataclass(frozen=True)
class PlanStep:
    """One planned skill call."""

    # Reason: Executor needs a stable skill key without knowing how the Planner chose it.
    skill_name: str

    # Role: Natural-language instruction passed to the selected Skill.
    query: str

    # Role: Structured inputs such as file paths, sheet names, or result limits.
    arguments: dict[str, Any] = field(default_factory=dict)

    # Role: Planner explanation useful for debugging and future reflection.
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    """A complete execution plan produced by Planner."""

    # Reason: Every plan should preserve the original user intent for reporting and audit logs.
    objective: str

    # Role: Ordered list of actions the Executor will run.
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Planner:
    """Creates plans from objectives and available skills."""

    # Reason: Dependency injection keeps Planner testable and avoids hard-coded skill knowledge.
    skill_registry: SkillRegistry

    async def plan(self, objective: str, context: dict[str, Any] | None = None) -> Plan:
        """Create a plan without executing it."""
        context = context or {}
        requested_skill = context.get("skill_name")
        arguments = dict(context.get("arguments", {}))
        available_skills = set(self.skill_registry.list_names())

        if isinstance(requested_skill, str):
            if requested_skill not in available_skills:
                raise KeyError(f"Unknown skill: {requested_skill}")
            return Plan(
                objective=objective,
                steps=[
                    PlanStep(
                        skill_name=requested_skill,
                        query=objective,
                        arguments=arguments,
                        reason="Skill was explicitly provided by caller context.",
                    )
                ],
            )

        file_path = arguments.get("file_path")
        if isinstance(file_path, str):
            suffix = Path(file_path).suffix.lower()
            if suffix in SPREADSHEET_EXTENSIONS:
                # 原因：Excel 分析必须先检查结构，再运行本地统计分析。
                # 作用：生成固定顺序的 schema → analysis 两步计划，避免整表进入 LLM。
                spreadsheet_steps = [
                    PlanStep(
                        skill_name="excel_schema",
                        query=objective,
                        arguments=arguments,
                        reason="Inspect spreadsheet schema and safe samples first.",
                    ),
                    PlanStep(
                        skill_name="excel_analysis",
                        query=objective,
                        arguments=arguments,
                        reason="Run local spreadsheet analysis after schema inspection.",
                    ),
                ]
                return Plan(
                    objective=objective,
                    steps=[
                        step for step in spreadsheet_steps
                        if step.skill_name in available_skills
                    ],
                )

            if suffix in DOCUMENT_EXTENSIONS and "document_parser" in available_skills:
                # 原因：非结构化文档必须先转换成统一 Markdown。
                # 作用：Planner 只安排解析任务，不在规划阶段读取文件内容。
                return Plan(
                    objective=objective,
                    steps=[
                        PlanStep(
                            skill_name="document_parser",
                            query=objective,
                            arguments=arguments,
                            reason="Normalize the document into Markdown.",
                        )
                    ],
                )

        lowered_objective = objective.lower()
        if (
            "web_search" in available_skills
            and any(term in lowered_objective for term in ("web", "网页", "联网", "互联网"))
        ):
            return Plan(
                objective=objective,
                steps=[
                    PlanStep(
                        skill_name="web_search",
                        query=objective,
                        reason="The objective explicitly requires web search.",
                    )
                ],
            )

        if (
            "graph_search" in available_skills
            and any(
                term in lowered_objective
                for term in (
                    "knowledge graph",
                    "relationship",
                    "multi-hop",
                    "graph path",
                    "知识图谱",
                    "关系路径",
                    "多跳",
                    "实体关系",
                )
            )
        ):
            # 原因：关系路径问题需要真实图遍历，普通向量 RAG 只能碰巧召回完整证据链。
            # 作用：Planner 明确选择 graph_search，Executor 仍只负责执行已生成的计划。
            return Plan(
                objective=objective,
                steps=[
                    PlanStep(
                        skill_name="graph_search",
                        query=objective,
                        reason="The objective requires persistent entity relationship traversal.",
                    )
                ],
            )

        if (
            "rag_search" in available_skills
            and any(term in lowered_objective for term in ("rag", "minirag", "知识库", "长期记忆"))
        ):
            return Plan(
                objective=objective,
                steps=[
                    PlanStep(
                        skill_name="rag_search",
                        query=objective,
                        reason="The objective explicitly requires local knowledge retrieval.",
                    )
                ],
            )

        # 原因：无法判断任务类型时自动选第一个 Skill 会执行错误能力。
        # 作用：返回空计划，让上层明确处理“无法规划”，而不是产生隐式副作用。
        return Plan(objective=objective, steps=[])
