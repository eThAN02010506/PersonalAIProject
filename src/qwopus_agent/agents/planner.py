"""Side-effect-free planners for Agent DAGs and direct Skill workflows.

Planner creates the production Agent DAG. SkillPlanner retains the lightweight CLI workflow
without executing skills, reading files, or producing side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qwopus_agent.agents.multi_agent import DelegatedTask, DelegationPlan
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
class SkillPlanner:
    """Create direct Skill plans for CLI and learned workflows."""

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


@dataclass(frozen=True)
class AgentPlanningRequest:
    """Capability facts needed to plan one production request."""

    objective: str
    has_documents: bool = False
    enable_web_search: bool = False
    enable_local_knowledge: bool = False
    generate_report: bool = False


@dataclass(frozen=True)
class AgentPlan:
    """One production Agent plan and its user-visible execution route."""

    route: Literal["single_agent", "multi_agent"]
    delegation: DelegationPlan
    terminal_task_id: str


@dataclass
class Planner:
    """Plan named Agent tasks and dependencies without executing any capability."""

    async def plan(self, request: AgentPlanningRequest) -> AgentPlan:
        """Build the smallest dependency DAG that can satisfy the request."""
        tasks: list[DelegatedTask] = []
        evidence_ids: list[str] = []
        if request.has_documents:
            tasks.append(
                DelegatedTask("document", request.objective, "document_agent")
            )
            evidence_ids.append("document")
        if request.enable_web_search:
            tasks.append(
                DelegatedTask("research", request.objective, "research_agent")
            )
            evidence_ids.append("research")
        if request.enable_local_knowledge:
            # 原因：上传文档必须先完成入库，当前请求的知识 Agent 才能检索到它。
            # 作用：仅在同次请求含上传文件时增加依赖，历史知识检索仍可并行执行。
            dependencies = ("document",) if request.has_documents else ()
            tasks.append(
                DelegatedTask(
                    "knowledge",
                    request.objective,
                    "knowledge_agent",
                    dependencies,
                )
            )
            evidence_ids.append("knowledge")
        if not evidence_ids:
            tasks.append(DelegatedTask("chat", request.objective, "chat_agent"))
            evidence_ids.append("chat")

        capability_count = sum(
            (
                request.has_documents,
                request.enable_web_search,
                request.enable_local_knowledge,
            )
        )
        route: Literal["single_agent", "multi_agent"] = (
            "multi_agent"
            if capability_count > 1 or request.generate_report
            else "single_agent"
        )
        terminal_task_id = evidence_ids[-1]
        if route == "multi_agent":
            tasks.append(
                DelegatedTask(
                    "synthesis",
                    request.objective,
                    "synthesis_agent",
                    tuple(evidence_ids),
                )
            )
            terminal_task_id = "synthesis"
        if request.generate_report:
            tasks.append(
                DelegatedTask(
                    "report",
                    request.objective,
                    "report_agent",
                    ("synthesis",),
                )
            )
            terminal_task_id = "report"
        return AgentPlan(
            route=route,
            delegation=DelegationPlan(
                objective=request.objective,
                tasks=tuple(tasks),
            ),
            terminal_task_id=terminal_task_id,
        )
