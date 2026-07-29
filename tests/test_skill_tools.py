import asyncio
import sys
import types
import unittest
from dataclasses import dataclass

from qwopus_agent.integrations.skill_tools import (
    build_promoted_workflow_tools,
    build_skill_tool,
)
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse
from qwopus_agent.skills.workflow import WorkflowSpec


class FakeTool:
    def __init__(self, *args, **kwargs) -> None:
        pass


@dataclass
class EchoSkill(BaseSkill):
    name: str = "echo"
    description: str = "Echo one query through the reusable Skill contract."

    async def run(self, request: SkillRequest) -> SkillResponse:
        await asyncio.sleep(0)
        return SkillResponse(
            success=True,
            content=f"{request.query}:{request.arguments.get('count')}",
        )


class SkillToolAdapterTests(unittest.TestCase):
    def test_adapter_executes_base_skill_without_reimplementing_capability(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with unittest.mock.patch.dict(sys.modules, {"smolagents": fake_module}):
            tool = build_skill_tool(
                EchoSkill(),
                inputs={
                    "query": {"type": "string", "description": "Text to echo."},
                    "count": {"type": "integer", "description": "Count metadata."},
                },
            )

        self.assertEqual(tool.forward("hello", 2), "hello:2")
        self.assertEqual(tool.name, "echo")

    def test_adapter_bounds_output_with_tokens_instead_of_characters(self) -> None:
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with unittest.mock.patch.dict(sys.modules, {"smolagents": fake_module}):
            tool = build_skill_tool(
                EchoSkill(),
                inputs={
                    "query": {"type": "string", "description": "Text to echo."},
                    "count": {"type": "integer", "description": "Count metadata."},
                },
                max_output_tokens=2,
            )

        result = tool.forward("中文内容很长", 2)

        self.assertIn("truncated", result)

    def test_promoted_workflow_uses_only_authorized_runtime_tools(self) -> None:
        class RuntimeSearchTool:
            name = "tavily_search"
            description = "Search current sources."

            def forward(self, query: str) -> str:
                return f"evidence for {query}"

        spec = WorkflowSpec(
            name="learned_web_search",
            version="0.1.0",
            description="Validated research workflow.",
            intent_examples=("research current prices",),
            steps=({"skill_name": "web_search"},),
            source_signature="signature",
        ).sealed()
        fake_module = types.SimpleNamespace(Tool=FakeTool)
        with unittest.mock.patch.dict(sys.modules, {"smolagents": fake_module}):
            enabled = build_promoted_workflow_tools(
                (spec,),
                [RuntimeSearchTool()],
            )
            disabled = build_promoted_workflow_tools((spec,), [])

        self.assertEqual(len(enabled), 1)
        self.assertEqual(
            enabled[0].forward("current prices"),
            "evidence for current prices",
        )
        self.assertNotIn("research current prices", enabled[0].description)
        self.assertEqual(disabled, [])


if __name__ == "__main__":
    unittest.main()
