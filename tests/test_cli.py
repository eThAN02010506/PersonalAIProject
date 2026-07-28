import asyncio
import unittest

from qwopus_agent.agents import AgentRouter, SkillExecutor, SkillPlanner
from qwopus_agent.cli import build_parser, run_objective
from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class EchoSkill(BaseSkill):
    name = "echo"
    description = "Echo the CLI objective."

    async def run(self, request: SkillRequest) -> SkillResponse:
        return SkillResponse(success=True, content=f"echo: {request.query}")


class CLITests(unittest.TestCase):
    def test_cli_uses_production_router_and_skill_contract(self) -> None:
        registry = SkillRegistry()
        registry.register(EchoSkill())
        router = AgentRouter(SkillPlanner(registry), SkillExecutor(registry))

        # 原因：删除旧 AgentLoop 后必须证明 CLI 通过当前异步 Router/Skill 主链执行。
        # 作用：防止未来再次引入平行的同步 Agent 或 ToolRegistry。
        result = asyncio.run(
            run_objective("hello", skill_name="echo", router=router)
        )

        self.assertTrue(result.execution.success)
        self.assertEqual(result.execution.content, "echo: hello")

    def test_cli_parser_accepts_skill_and_file_context(self) -> None:
        args = build_parser().parse_args(
            ["analyze", "--skill", "document_parser", "--file", "report.pdf"]
        )

        self.assertEqual(args.objective, "analyze")
        self.assertEqual(args.skill, "document_parser")
        self.assertEqual(str(args.file), "report.pdf")


if __name__ == "__main__":
    unittest.main()
