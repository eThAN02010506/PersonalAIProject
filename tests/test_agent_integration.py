"""End-to-end Planner -> Executor -> real Skill integration tests.

These exercise the production `AgentRouter` wiring through a real skill
registry, closing the gap where Planner/Executor and the smolagents driver
were only tested in isolation.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from qwopus_agent.agents import AgentRouter, SkillExecutor, SkillPlanner
from qwopus_agent.skills import BaseSkill, SkillRegistry, SkillRequest, SkillResponse


class DocumentParserIntegrationTests(unittest.TestCase):
    """Run the production router against a real discovered document skill."""

    def test_auto_route_document_parses_real_txt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("# Title\n\nProject Alpha baseline.", encoding="utf-8")
            registry = SkillRegistry.discover()
            router = AgentRouter(
                planner=SkillPlanner(skill_registry=registry),
                executor=SkillExecutor(skill_registry=registry),
            )

            run = asyncio.run(
                router.run(
                    "解析这个文档",
                    context={"arguments": {"file_path": str(path)}},
                )
            )

        # 原因：自动路由必须真实运行 document_parser，而不是返回空计划或占位内容。
        # 作用：锁定 TXT 经 Planner 路由、Executor 执行后产出真实 Markdown 正文。
        self.assertTrue(run.execution.success)
        self.assertEqual(run.plan.steps[0].skill_name, "document_parser")
        self.assertIn("Project Alpha baseline.", run.execution.content)
        self.assertEqual(run.execution.steps[0].response.data["metadata"]["source_type"], "text")

    def test_unknown_explicit_skill_fails_planning(self) -> None:
        registry = SkillRegistry()
        router = AgentRouter(
            planner=SkillPlanner(skill_registry=registry),
            executor=SkillExecutor(skill_registry=registry),
        )
        with self.assertRaises(KeyError):
            asyncio.run(
                router.run("anything", context={"skill_name": "does_not_exist"})
            )

    def test_empty_plan_reports_failure_not_success(self) -> None:
        # 无法识别的任务类型返回空计划，Executor 必须显式失败而不是伪装成功。
        registry = SkillRegistry()
        router = AgentRouter(
            planner=SkillPlanner(skill_registry=registry),
            executor=SkillExecutor(skill_registry=registry),
        )
        run = asyncio.run(router.run("一个无法路由的任务"))
        self.assertFalse(run.execution.success)
        self.assertIn("No executable plan steps", run.execution.content)


class CustomSkillIntegrationTests(unittest.TestCase):
    """Run the router against an explicitly injected custom skill."""

    def test_router_executes_registered_custom_skill_end_to_end(self) -> None:
        class EchoSkill(BaseSkill):
            name = "echo"
            description = "Echo the query back."

            async def run(self, request: SkillRequest) -> SkillResponse:
                return SkillResponse(
                    success=True,
                    content=f"echo:{request.query}",
                )

        registry = SkillRegistry()
        registry.register(EchoSkill())
        router = AgentRouter(
            planner=SkillPlanner(skill_registry=registry),
            executor=SkillExecutor(skill_registry=registry),
        )

        run = asyncio.run(
            router.run("hello world", context={"skill_name": "echo"})
        )

        self.assertTrue(run.execution.success)
        self.assertEqual(run.execution.content, "echo:hello world")
        self.assertEqual(run.plan.steps[0].skill_name, "echo")


if __name__ == "__main__":
    unittest.main()
