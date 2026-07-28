import asyncio
import unittest

from qwopus_agent.agents import (
    AgentRouter,
    ResearchAgent,
    SkillExecutor,
    SkillPlanner,
)
from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class ResearchEchoSkill(BaseSkill):
    name = "research_echo"
    description = "Return a sufficiently long research answer."

    async def run(self, request: SkillRequest) -> SkillResponse:
        return SkillResponse(
            success=True,
            content=(
                "This research answer is long enough and comes from a reusable skill. "
                "It includes a clear result, keeps the execution trace intact, and can be "
                "reviewed by the reflection evaluator without requiring another retry."
            ),
        )


class ResearchAgentTests(unittest.TestCase):
    def test_research_agent_reuses_router_and_reflection(self) -> None:
        registry = SkillRegistry()
        registry.register(ResearchEchoSkill())
        agent = ResearchAgent(
            router=AgentRouter(
                planner=SkillPlanner(skill_registry=registry),
                executor=SkillExecutor(skill_registry=registry),
            )
        )

        # 原因：Research Agent 应该建立在 Planner/Executor 之上，而不是另起一套执行逻辑。
        # 作用：验证研究任务能复用 Router，并产生 Reflection 结果。
        run = asyncio.run(
            agent.run("Research this topic", context={"skill_name": "research_echo"})
        )

        self.assertTrue(run.agent_run.execution.success)
        self.assertEqual(run.agent_run.plan.steps[0].skill_name, "research_echo")
        self.assertFalse(run.reflection.needs_retry)


if __name__ == "__main__":
    unittest.main()
