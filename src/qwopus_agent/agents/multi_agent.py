"""Minimal multi-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RunnableAgent(Protocol):
    """Protocol for agents that can be coordinated."""

    async def run(self, question: str, context: dict[str, Any] | None = None) -> Any:
        """Run one task and return an agent-specific result."""


@dataclass(frozen=True)
class NamedAgentRun:
    """Result from one named agent."""

    name: str

    result: Any


@dataclass(frozen=True)
class MultiAgentRun:
    """Complete result from a multi-agent run."""

    objective: str

    runs: list[NamedAgentRun] = field(default_factory=list)


@dataclass
class MultiAgentCoordinator:
    """Coordinate multiple agents without owning their internals."""

    agents: dict[str, RunnableAgent]

    async def run(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        order: list[str] | None = None,
    ) -> MultiAgentRun:
        """Run selected agents in deterministic order."""
        names = order or sorted(self.agents)
        runs: list[NamedAgentRun] = []
        for name in names:
            agent = self.agents[name]
            # 原因：多 Agent 编排层不应该知道具体 Agent 是研究、反思还是工具 Agent。
            # 作用：只按名字调度 run()，让后续可以替换不同 Agent 类型。
            result = await agent.run(objective, context=context)
            runs.append(NamedAgentRun(name=name, result=result))
        return MultiAgentRun(objective=objective, runs=runs)
