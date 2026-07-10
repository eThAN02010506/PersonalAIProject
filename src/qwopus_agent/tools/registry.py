"""Tool registry for resolving tool names at execution time."""

from __future__ import annotations

from dataclasses import dataclass, field

from qwopus_agent.tools.base import BaseTool


@dataclass
class ToolRegistry:
    """A small registry that keeps tools decoupled from the agent loop."""

    _tools: dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list_names(self) -> list[str]:
        return sorted(self._tools)
