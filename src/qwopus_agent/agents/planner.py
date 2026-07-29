"""Side-effect-free planners for Agent DAGs and direct Skill workflows.

Planner creates the production Agent DAG. SkillPlanner retains the lightweight CLI workflow
without executing skills, reading files, or producing side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qwopus_agent.agents.multi_agent import DelegatedTask, DelegationPlan
from qwopus_agent.skills import SkillRegistry
from qwopus_agent.skills.workflow import WorkflowSkill

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

        promoted_skill = _match_promoted_workflow(
            objective,
            self.skill_registry,
            previous_objective=context.get("previous_objective"),
        )
        if promoted_skill is not None:
            return Plan(
                objective=objective,
                steps=[
                    PlanStep(
                        skill_name=promoted_skill,
                        query=objective,
                        arguments=arguments,
                        reason=(
                            "The request matched validated intent examples from a promoted Skill."
                        ),
                    )
                ],
            )

        # 原因：无法判断任务类型时自动选第一个 Skill 会执行错误能力。
        # 作用：返回空计划，让上层明确处理“无法规划”，而不是产生隐式副作用。
        return Plan(objective=objective, steps=[])


def _match_promoted_workflow(
    objective: str,
    registry: SkillRegistry,
    *,
    previous_objective: Any = None,
) -> str | None:
    """Return one unambiguous promoted workflow matching validated intent examples."""
    query = objective
    if isinstance(previous_objective, str) and previous_objective.strip():
        # 原因：“再做一次”“和刚才一样”等模糊请求本身没有足够的意图词。
        # 作用：调用方显式提供上一任务时，用它补充匹配上下文但仍执行当前问题。
        query = f"{previous_objective} {objective}"
    query_units = _intent_units(query)
    if len(query_units) < 2:
        return None

    scores: list[tuple[float, str]] = []
    for skill_name in registry.list_names():
        skill = registry.get(skill_name)
        if not isinstance(skill, WorkflowSkill):
            continue
        score = max(
            (
                _intent_similarity(query, query_units, example)
                for example in skill.spec.intent_examples
            ),
            default=0.0,
        )
        if score >= 0.5:
            scores.append((score, skill_name))
    scores.sort(reverse=True)
    if not scores:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.12:
        # 原因：两个 Skill 得分接近时自动选择会把模糊问题路由到错误工作流。
        # 作用：保持空计划，由上层追问或要求用户显式选择 Skill。
        return None
    return scores[0][1]


def _intent_similarity(query: str, query_units: set[str], example: str) -> float:
    normalized_query = " ".join(query.casefold().split())
    normalized_example = " ".join(example.casefold().split())
    if (
        len(normalized_query) >= 6
        and (
            normalized_query in normalized_example
            or normalized_example in normalized_query
        )
    ):
        return 1.0
    example_units = _intent_units(example)
    overlap = query_units & example_units
    if len(overlap) < 2:
        return 0.0
    return len(overlap) / min(len(query_units), len(example_units))


def _intent_units(value: str) -> set[str]:
    """Tokenize Latin words and CJK bigrams without adding a model dependency."""
    lowered = value.casefold()
    words = {word for word in re.findall(r"[a-z0-9_]{3,}", lowered)}
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    bigrams = {
        run[index : index + 2]
        for run in cjk_runs
        for index in range(len(run) - 1)
    }
    return words | bigrams


@dataclass(frozen=True)
class AgentPlanningRequest:
    """Capability facts needed to plan one production request."""

    objective: str
    has_documents: bool = False
    enable_web_search: bool = False
    enable_browser: bool = False
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
        if request.enable_browser:
            # 原因：浏览器渲染和 Tavily 搜索是不同能力，Planner 不应把两者混为一个 Agent。
            # 作用：用户单独授权后创建 browser_agent，Executor 仍只执行确定的 DAG。
            tasks.append(
                DelegatedTask("browser", request.objective, "browser_agent")
            )
            evidence_ids.append("browser")
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
                request.enable_browser,
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
            synthesis_dependencies = list(evidence_ids)
            if len(evidence_ids) > 1:
                # 原因：多个独立来源可能互相矛盾，直接综合会让最终答案静默选择一方。
                # 作用：先由无工具 reviewer 标出一致点、冲突和证据缺口，再交给 synthesis。
                tasks.append(
                    DelegatedTask(
                        "review",
                        request.objective,
                        "review_agent",
                        tuple(evidence_ids),
                    )
                )
                synthesis_dependencies.append("review")
            tasks.append(
                DelegatedTask(
                    "synthesis",
                    request.objective,
                    "synthesis_agent",
                    tuple(synthesis_dependencies),
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
