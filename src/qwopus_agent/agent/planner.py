"""Minimal planner skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field

from qwopus_agent.llm import BaseLLM, ChatMessage


@dataclass(frozen=True)
class PlanStep:
    """One step in an agent plan."""

    description: str
    tool_name: str | None = None


@dataclass(frozen=True)
class Plan:
    """A simple ordered plan."""

    objective: str
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Planner:
    """Creates a minimal plan from a user objective."""

    llm: BaseLLM | None = None

    def create_plan(self, objective: str) -> Plan:
        if self.llm is None:
            return Plan(objective=objective, steps=[PlanStep(description=objective)])

        response = self.llm.generate(
            [
                ChatMessage(
                    role="system",
                    content="Create a concise execution plan. Return one step per line.",
                ),
                ChatMessage(role="user", content=objective),
            ],
            temperature=0.1,
        )
        steps = [
            PlanStep(description=line.strip("- ").strip())
            for line in response.content.splitlines()
            if line.strip()
        ]
        return Plan(objective=objective, steps=steps or [PlanStep(description=objective)])
