"""Executor Agent.

Executor is responsible only for running a previously created plan. It does not decide what should
be done; it only resolves skills and executes the steps in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents.planner import Plan
from qwopus_agent.skills import SkillRegistry, SkillRequest, SkillResponse


@dataclass(frozen=True)
class StepExecution:
    """Execution result for one plan step."""

    # Reason: Reports and debugging need to connect outputs back to the skill that produced them.
    skill_name: str

    # Role: Skill response envelope.
    response: SkillResponse


@dataclass(frozen=True)
class ExecutionResult:
    """Full execution result returned by Executor."""

    # Reason: Router and UI need one boolean summary without inspecting every step.
    success: bool

    # Role: Ordered trace of skill executions.
    steps: list[StepExecution] = field(default_factory=list)

    # Role: Final merged text for quick CLI/UI display.
    content: str = ""


@dataclass
class Executor:
    """Executes plan steps through registered skills."""

    # Reason: Dependency injection allows tests to provide fake skills and production to auto-discover.
    skill_registry: SkillRegistry

    async def execute(self, plan: Plan, context: dict[str, Any] | None = None) -> ExecutionResult:
        """Execute a plan without changing the plan itself."""
        context = context or {}
        executed_steps: list[StepExecution] = []
        contents: list[str] = []

        for step in plan.steps:
            skill = self.skill_registry.get(step.skill_name)
            request = SkillRequest(
                query=step.query,
                arguments=step.arguments,
                context={"objective": plan.objective, **context},
            )
            response = await skill.run(request)
            executed_steps.append(StepExecution(skill_name=step.skill_name, response=response))
            contents.append(response.content)

            if not response.success:
                return ExecutionResult(
                    success=False,
                    steps=executed_steps,
                    content="\n".join(contents),
                )

        return ExecutionResult(
            success=True,
            steps=executed_steps,
            content="\n".join(contents),
        )
