import unittest
from typing import Any

from qwopus_agent.tools import BaseTool, ToolContext, ToolRegistry, ToolResult


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo structured input."

    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        return ToolResult(success=True, content=str(arguments["text"]))


class ToolRegistryTests(unittest.TestCase):
    def test_tool_registry_registers_and_resolves_tools(self) -> None:
        registry = ToolRegistry()
        tool = EchoTool()

        registry.register(tool)

        self.assertIs(registry.get("echo"), tool)
        self.assertEqual(registry.list_names(), ["echo"])

    def test_tool_registry_rejects_duplicate_names(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())

        with self.assertRaisesRegex(ValueError, "Tool already registered"):
            registry.register(EchoTool())
