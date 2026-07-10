"""Agent Router.

Router wires Planner and Executor together. It owns orchestration, while Planner and Executor keep
their single responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qwopus_agent.agents.executor import ExecutionResult, Executor
from qwopus_agent.agents.planner import Plan, Planner


@dataclass(frozen=True)
class AgentRun:
    """Complete result of one routed agent request."""

    # Reason: Keeping both plan and execution supports future reflection and report generation.
    plan: Plan

    # Role: Actual outputs from executed skills.
    execution: ExecutionResult


@dataclass
class AgentRouter:
    """Coordinates planning followed by execution."""

    # Reason: Router depends on abstractions so CLI/UI does not contain business logic.
    planner: Planner
    executor: Executor

    async def run(self, objective: str, context: dict[str, Any] | None = None) -> AgentRun:
        """Plan and execute one user objective."""
        plan = await self.planner.plan(objective, context=context)
        execution = await self.executor.execute(plan, context=context)
        return AgentRun(plan=plan, execution=execution)
