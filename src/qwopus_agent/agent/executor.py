"""Minimal executor skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field

from qwopus_agent.agent.planner import Plan
from qwopus_agent.tools import ToolContext, ToolRegistry


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a plan."""

    success: bool
    output: str
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class Executor:
    """Executes a plan without binding the agent to concrete tools."""

    tools: ToolRegistry = field(default_factory=ToolRegistry)

    def execute(self, plan: Plan) -> ExecutionResult:
        outputs: list[str] = []
        for index, step in enumerate(plan.steps, start=1):
            if step.tool_name is None:
                outputs.append(f"{index}. {step.description}")
                continue

            tool = self.tools.get(step.tool_name)
            result = tool.run({"description": step.description}, ToolContext())
            if not result.success:
                return ExecutionResult(success=False, output=result.content)
            outputs.append(result.content)

        return ExecutionResult(success=True, output="\n".join(outputs))
