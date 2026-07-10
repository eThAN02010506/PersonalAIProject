"""Planner Agent.

Planner is responsible only for turning a user objective into an execution plan. It never calls
skills, reads files, or performs side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.skills import SkillRegistry


@dataclass(frozen=True)
class PlanStep:
    """One planned skill call."""

    # Reason: Executor needs a stable skill key without knowing how the Planner chose it.
    skill_name: str

    # Role: Natural-language instruction passed to the selected Skill.
    query: str

    # Role: Structured inputs such as file paths, sheet names, or result limits.
    arguments: dict[str, Any] = field(default_factory=dict)

    # Role: Planner explanation useful for debugging and future reflection.
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    """A complete execution plan produced by Planner."""

    # Reason: Every plan should preserve the original user intent for reporting and audit logs.
    objective: str

    # Role: Ordered list of actions the Executor will run.
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Planner:
    """Creates plans from objectives and available skills."""

    # Reason: Dependency injection keeps Planner testable and avoids hard-coded skill knowledge.
    skill_registry: SkillRegistry

    async def plan(self, objective: str, context: dict[str, Any] | None = None) -> Plan:
        """Create a plan without executing it."""
        context = context or {}
        requested_skill = context.get("skill_name")

        if isinstance(requested_skill, str):
            return Plan(
                objective=objective,
                steps=[
                    PlanStep(
                        skill_name=requested_skill,
                        query=objective,
                        arguments=dict(context.get("arguments", {})),
                        reason="Skill was explicitly provided by caller context.",
                    )
                ],
            )

        skill_names = self.skill_registry.list_names()
        if not skill_names:
            return Plan(objective=objective, steps=[])

        selected_skill = skill_names[0]
        return Plan(
            objective=objective,
            steps=[
                PlanStep(
                    skill_name=selected_skill,
                    query=objective,
                    reason="Defaulted to the first registered skill until LLM planning is enabled.",
                )
            ],
        )
