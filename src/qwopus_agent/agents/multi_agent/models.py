"""Typed contracts shared by multi-agent components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RunnableAgent(Protocol):
    """Protocol implemented by every agent managed by the supervisor."""

    async def run(self, question: str, context: dict[str, Any] | None = None) -> Any:
        """Run one delegated task and return an agent-specific result."""


@dataclass(frozen=True)
class AgentProfile:
    """Information used by a delegator when selecting an agent."""

    name: str
    description: str = ""
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class DelegatedTask:
    """One task assigned by the supervisor to a named agent."""

    task_id: str
    objective: str
    agent_name: str
    dependencies: tuple[str, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DelegationPlan:
    """Dependency-aware task plan produced by a task delegator."""

    objective: str
    tasks: tuple[DelegatedTask, ...]


@dataclass(frozen=True)
class NamedAgentRun:
    """Raw result from one named agent task."""

    name: str
    result: Any
    task_id: str = ""
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class AgentContribution:
    """Normalized agent result shared with later tasks and the arbiter."""

    task_id: str
    agent_name: str
    content: str
    success: bool
    confidence: float
    raw: Any = None
    error: str | None = None


@dataclass(frozen=True)
class DebateStatement:
    """One agent's review of all candidate conclusions."""

    agent_name: str
    round_number: int
    content: str


@dataclass(frozen=True)
class ArbitrationDecision:
    """Final decision produced after comparing agent conclusions."""

    final_answer: str
    rationale: str
    selected_agents: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiAgentRun:
    """Complete supervised multi-agent result."""

    objective: str
    runs: list[NamedAgentRun] = field(default_factory=list)
    delegation_plan: DelegationPlan | None = None
    shared_state: dict[str, Any] = field(default_factory=dict)
    debate: list[DebateStatement] = field(default_factory=list)
    decision: ArbitrationDecision | None = None
    final_answer: str = ""


class TaskDelegator(Protocol):
    """Contract for components that turn an objective into delegated tasks."""

    async def create_plan(
        self,
        objective: str,
        profiles: dict[str, AgentProfile],
        context: dict[str, Any],
    ) -> DelegationPlan:
        """Create a dependency-aware delegation plan."""


class ResultArbiter(Protocol):
    """Contract for components that resolve candidate result conflicts."""

    async def decide(
        self,
        objective: str,
        contributions: list[AgentContribution],
        debate: list[DebateStatement],
        context: dict[str, Any],
    ) -> ArbitrationDecision:
        """Return one final answer and explain the selection."""
