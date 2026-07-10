import asyncio
import unittest

from qwopus_agent.agents import AgentRouter, Executor, Planner
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
        planner = Planner(skill_registry=registry)

        plan = asyncio.run(planner.plan("Analyze uploaded file"))

        self.assertEqual(plan.objective, "Analyze uploaded file")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].skill_name, "echo")

    def test_executor_only_executes_existing_plan(self) -> None:
        registry = SkillRegistry()
        registry.register(EchoSkill())
        planner = Planner(skill_registry=registry)
        executor = Executor(skill_registry=registry)
        plan = asyncio.run(planner.plan("Analyze uploaded file"))

        result = asyncio.run(executor.execute(plan))

        self.assertTrue(result.success)
        self.assertEqual(result.content, "echo: Analyze uploaded file")

    def test_router_orchestrates_planner_and_executor(self) -> None:
        registry = SkillRegistry()
        registry.register(EchoSkill())
        router = AgentRouter(
            planner=Planner(skill_registry=registry),
            executor=Executor(skill_registry=registry),
        )

        run = asyncio.run(router.run("Analyze uploaded file"))

        self.assertTrue(run.execution.success)
        self.assertEqual(run.plan.steps[0].skill_name, "echo")
