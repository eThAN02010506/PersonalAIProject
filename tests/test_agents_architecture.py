import asyncio
import unittest
from unittest.mock import patch

from qwopus_agent.agents import (
    AgentProfile,
    AgentRouter,
    Executor,
    Planner,
    SkillExecutor,
    SkillPlanner,
)
from qwopus_agent.agents.planner import AgentPlanningRequest
from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class EchoSkill(BaseSkill):
    name = "echo"
    description = "Echoes the request query."

    async def run(self, request: SkillRequest) -> SkillResponse:
        return SkillResponse(success=True, content=f"echo: {request.query}")


class AgentArchitectureTests(unittest.TestCase):
    def test_planner_only_creates_plan(self) -> None:
        registry = SkillRegistry()
        registry.register(EchoSkill())
        planner = SkillPlanner(skill_registry=registry)

        plan = asyncio.run(
            planner.plan("Analyze uploaded file", context={"skill_name": "echo"})
        )

        self.assertEqual(plan.objective, "Analyze uploaded file")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].skill_name, "echo")

    def test_executor_only_executes_existing_plan(self) -> None:
        registry = SkillRegistry()
        registry.register(EchoSkill())
        planner = SkillPlanner(skill_registry=registry)
        executor = SkillExecutor(skill_registry=registry)
        plan = asyncio.run(
            planner.plan("Analyze uploaded file", context={"skill_name": "echo"})
        )

        # 原因：仅检查最终文本无法证明 Executor 没有绕过 Registry。
        # 作用：锁定 Executor 必须通过 SkillRegistry.execute() 执行计划步骤。
        with patch.object(registry, "execute", wraps=registry.execute) as execute_skill:
            result = asyncio.run(executor.execute(plan))

        self.assertTrue(result.success)
        self.assertEqual(result.content, "echo: Analyze uploaded file")
        execute_skill.assert_awaited_once()

    def test_router_orchestrates_planner_and_executor(self) -> None:
        registry = SkillRegistry()
        registry.register(EchoSkill())
        router = AgentRouter(
            planner=SkillPlanner(skill_registry=registry),
            executor=SkillExecutor(skill_registry=registry),
        )

        run = asyncio.run(
            router.run("Analyze uploaded file", context={"skill_name": "echo"})
        )

        self.assertTrue(run.execution.success)
        self.assertEqual(run.plan.steps[0].skill_name, "echo")

    def test_planner_routes_excel_through_schema_then_analysis(self) -> None:
        planner = SkillPlanner(skill_registry=SkillRegistry.discover())

        # 原因：Excel 不应直接进入通用文档解析或一次性发送给模型。
        # 作用：验证 Planner 固定生成 schema → local analysis 的安全顺序。
        plan = asyncio.run(
            planner.plan(
                "分析销售数据",
                context={"arguments": {"file_path": "sales.xlsx"}},
            )
        )

        self.assertEqual(
            [step.skill_name for step in plan.steps],
            ["excel_schema", "excel_analysis"],
        )

    def test_planner_routes_document_to_parser(self) -> None:
        planner = SkillPlanner(skill_registry=SkillRegistry.discover())

        plan = asyncio.run(
            planner.plan(
                "总结这份文件",
                context={"arguments": {"file_path": "report.pdf"}},
            )
        )

        self.assertEqual([step.skill_name for step in plan.steps], ["document_parser"])

    def test_planner_routes_knowledge_and_web_queries(self) -> None:
        planner = SkillPlanner(skill_registry=SkillRegistry.discover())

        rag_plan = asyncio.run(planner.plan("在 MiniRAG 知识库中查找项目记录"))
        graph_plan = asyncio.run(planner.plan("查询 Company A 到 Company B 的多跳关系路径"))
        web_plan = asyncio.run(planner.plan("联网搜索 web 上的最新资料"))

        self.assertEqual([step.skill_name for step in rag_plan.steps], ["rag_search"])
        self.assertEqual([step.skill_name for step in graph_plan.steps], ["graph_search"])
        self.assertEqual([step.skill_name for step in web_plan.steps], ["web_search"])

    def test_planner_does_not_guess_when_route_is_unknown(self) -> None:
        planner = SkillPlanner(skill_registry=SkillRegistry.discover())

        plan = asyncio.run(planner.plan("处理这个任务"))

        self.assertEqual(plan.steps, [])

    def test_executor_reports_empty_plan_as_failure(self) -> None:
        registry = SkillRegistry.discover()
        planner = SkillPlanner(skill_registry=registry)
        plan = asyncio.run(planner.plan("处理这个任务"))

        result = asyncio.run(SkillExecutor(skill_registry=registry).execute(plan))

        self.assertFalse(result.success)
        self.assertEqual(result.content, "No executable plan steps.")

    def test_production_planner_only_builds_agent_dependency_dag(self) -> None:
        plan = asyncio.run(
            Planner().plan(
                AgentPlanningRequest(
                    objective="compare evidence",
                    enable_web_search=True,
                    enable_local_knowledge=True,
                )
            )
        )

        self.assertEqual(plan.route, "multi_agent")
        self.assertEqual(plan.terminal_task_id, "synthesis")
        self.assertEqual(
            [task.agent_name for task in plan.delegation.tasks],
            [
                "research_agent",
                "knowledge_agent",
                "review_agent",
                "synthesis_agent",
            ],
        )
        self.assertEqual(
            plan.delegation.tasks[-1].dependencies,
            ("research", "knowledge", "review"),
        )

    def test_production_executor_runs_supplied_plan(self) -> None:
        class EchoAgent:
            async def run(self, question, context=None):
                return f"done: {question}"

        plan = asyncio.run(
            Planner().plan(AgentPlanningRequest(objective="hello"))
        )
        result = asyncio.run(
            Executor().execute(
                plan,
                agents={"chat_agent": EchoAgent()},
                profiles={"chat_agent": AgentProfile("chat_agent")},
            )
        )

        self.assertEqual(result.final_answer, "done: hello")

    def test_production_planner_declares_single_and_report_terminals(self) -> None:
        single = asyncio.run(
            Planner().plan(AgentPlanningRequest(objective="hello"))
        )
        browser = asyncio.run(
            Planner().plan(
                AgentPlanningRequest(
                    objective="open page",
                    enable_browser=True,
                )
            )
        )
        report = asyncio.run(
            Planner().plan(
                AgentPlanningRequest(
                    objective="analyze",
                    has_documents=True,
                    generate_report=True,
                )
            )
        )

        self.assertEqual(single.terminal_task_id, "chat")
        self.assertEqual(browser.terminal_task_id, "browser")
        self.assertEqual(browser.delegation.tasks[0].agent_name, "browser_agent")
        self.assertEqual(report.terminal_task_id, "report")
