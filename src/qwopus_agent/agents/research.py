"""Research Agent skeleton built on Planner, Executor, and Reflection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents.router import AgentRouter, AgentRun
from qwopus_agent.reflection import ReflectionResult, TaskReflectionEvaluator


@dataclass(frozen=True)
class ResearchRun:
    """One research-agent run."""

    question: str

    agent_run: AgentRun

    reflection: ReflectionResult


@dataclass
class ResearchAgent:
    """Coordinate research-style tasks through existing Agent primitives."""

    router: AgentRouter

    reflection_evaluator: TaskReflectionEvaluator = field(default_factory=TaskReflectionEvaluator)

    async def run(self, question: str, context: dict[str, Any] | None = None) -> ResearchRun:
        """Run one research question through Planner, Executor, and Reflection."""
        # 原因：Research Agent 不应该重新实现规划、执行和工具调用。
        # 作用：复用稳定的 AgentRouter，再把结果交给 Reflection 评估质量。
        agent_run = await self.router.run(question, context=context)
        reflection = self.reflection_evaluator.evaluate(
            objective=question,
            answer=agent_run.execution.content,
            success=agent_run.execution.success,
            step_names=[step.skill_name for step in agent_run.plan.steps],
        )
        return ResearchRun(question=question, agent_run=agent_run, reflection=reflection)
