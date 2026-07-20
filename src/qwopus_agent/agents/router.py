"""Agent Router.

Router wires Planner and Executor together. It owns orchestration, while Planner and Executor keep
their single responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from qwopus_agent.agents.executor import ExecutionResult, Executor
from qwopus_agent.agents.planner import Plan, Planner


@dataclass(frozen=True)
class AgentRun:
    """Complete result of one routed agent request."""

    # Reason: Keeping both plan and execution supports future reflection and report generation.
    plan: Plan

    # Role: Actual outputs from executed skills.
    execution: ExecutionResult

    # Role: Non-fatal observer failures, such as a damaged growth catalog.
    observer_errors: tuple[str, ...] = ()


class AgentRunObserver(Protocol):
    """Receives completed Agent runs without changing Router business logic."""

    async def observe(
        self,
        objective: str,
        run: AgentRun,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Observe a completed run and optionally create a side effect."""


@dataclass
class AgentRouter:
    """Coordinates planning followed by execution."""

    # Reason: Router depends on abstractions so CLI/UI does not contain business logic.
    planner: Planner
    executor: Executor

    # Reason: Reflection and growth are post-run concerns and must remain injectable.
    # Role: Observers receive the complete trace after Planner and Executor finish.
    observers: tuple[AgentRunObserver, ...] = field(default_factory=tuple)

    async def run(self, objective: str, context: dict[str, Any] | None = None) -> AgentRun:
        """Plan and execute one user objective."""
        plan = await self.planner.plan(objective, context=context)
        execution = await self.executor.execute(plan, context=context)
        run = AgentRun(plan=plan, execution=execution)
        observer_errors: list[str] = []
        for observer in self.observers:
            try:
                # 原因：自动成长必须发生在完整成功轨迹产生之后，不能侵入 Planner/Executor。
                # 作用：Router 只发布运行事件；观察器失败也不会覆盖用户已经得到的结果。
                await observer.observe(objective, run, context=context)
            except Exception as exc:  # noqa: BLE001 - observer failures are non-fatal by design.
                observer_errors.append(f"{type(exc).__name__}: {exc}")
        return AgentRun(
            plan=plan,
            execution=execution,
            observer_errors=tuple(observer_errors),
        )
