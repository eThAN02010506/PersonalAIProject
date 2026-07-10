"""Top-level agent loop skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.agent.executor import ExecutionResult, Executor
from qwopus_agent.agent.planner import Plan, Planner


@dataclass(frozen=True)
class AgentRunResult:
    """Full result returned by the agent loop."""

    objective: str
    plan: Plan
    execution: ExecutionResult


@dataclass
class AgentLoop:
    """Coordinates planning and execution."""

    planner: Planner
    executor: Executor

    def run(self, objective: str) -> AgentRunResult:
        plan = self.planner.create_plan(objective)
        execution = self.executor.execute(plan)
        return AgentRunResult(objective=objective, plan=plan, execution=execution)
