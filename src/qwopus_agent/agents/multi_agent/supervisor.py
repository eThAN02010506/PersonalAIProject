"""Supervisor lifecycle for delegated multi-agent work."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents.multi_agent._utils import (
    result_confidence,
    result_content,
    result_success,
    safe_identifier,
)
from qwopus_agent.agents.multi_agent.arbitration import ConsensusArbiter
from qwopus_agent.agents.multi_agent.delegation import (
    DeterministicTaskDelegator,
    validate_plan,
)
from qwopus_agent.agents.multi_agent.models import (
    AgentContribution,
    AgentProfile,
    DebateStatement,
    DelegatedTask,
    DelegationPlan,
    MultiAgentRun,
    NamedAgentRun,
    ResultArbiter,
    RunnableAgent,
    TaskDelegator,
)
from qwopus_agent.agents.multi_agent.state import SharedAgentState


@dataclass
class MultiAgentSupervisor:
    """Delegate, coordinate, debate, and arbitrate independent agents."""

    agents: dict[str, RunnableAgent]
    profiles: dict[str, AgentProfile] = field(default_factory=dict)
    delegator: TaskDelegator = field(default_factory=DeterministicTaskDelegator)
    arbiter: ResultArbiter = field(default_factory=ConsensusArbiter)
    max_parallel: int = 4
    debate_rounds: int = 1

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("MultiAgentSupervisor requires at least one agent.")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1.")
        if self.debate_rounds < 0:
            raise ValueError("debate_rounds cannot be negative.")
        unknown_profiles = set(self.profiles) - set(self.agents)
        if unknown_profiles:
            raise ValueError(f"Profiles reference unknown agents: {sorted(unknown_profiles)}")
        self.profiles = {
            name: self.profiles.get(name, AgentProfile(name=name))
            for name in self.agents
        }

    async def run(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        order: list[str] | None = None,
    ) -> MultiAgentRun:
        """Run the complete supervised multi-agent lifecycle."""
        base_context = dict(context or {})
        supplied_plan = base_context.get("delegation_plan")
        if isinstance(supplied_plan, DelegationPlan):
            plan = supplied_plan
        elif order is not None:
            plan = self._ordered_plan(objective, order)
        else:
            plan = await self.delegator.create_plan(objective, self.profiles, base_context)
        validate_plan(plan, self.profiles)

        state = SharedAgentState(values=dict(base_context.get("shared_state", {})))
        runs, contributions = await self._execute_plan(plan, state, base_context)
        debate = await self._debate(objective, contributions, state, base_context)
        decision = await self.arbiter.decide(objective, contributions, debate, base_context)
        return MultiAgentRun(
            objective=objective,
            runs=runs,
            delegation_plan=plan,
            shared_state=await state.snapshot(),
            debate=debate,
            decision=decision,
            final_answer=decision.final_answer,
        )

    def _ordered_plan(self, objective: str, order: list[str]) -> DelegationPlan:
        tasks: list[DelegatedTask] = []
        previous_task_id: str | None = None
        for index, name in enumerate(order, start=1):
            task_id = f"task_{index}_{safe_identifier(name)}"
            tasks.append(
                DelegatedTask(
                    task_id=task_id,
                    objective=objective,
                    agent_name=name,
                    dependencies=(previous_task_id,) if previous_task_id else (),
                )
            )
            previous_task_id = task_id
        return DelegationPlan(objective=objective, tasks=tuple(tasks))

    async def _execute_plan(
        self,
        plan: DelegationPlan,
        state: SharedAgentState,
        base_context: dict[str, Any],
    ) -> tuple[list[NamedAgentRun], list[AgentContribution]]:
        remaining = {task.task_id: task for task in plan.tasks}
        completed: set[str] = set()
        failed: set[str] = set()
        runs: list[NamedAgentRun] = []
        contributions: list[AgentContribution] = []
        semaphore = asyncio.Semaphore(self.max_parallel)

        while remaining:
            blocked = [
                task
                for task in remaining.values()
                if any(dependency in failed for dependency in task.dependencies)
            ]
            for task in blocked:
                dependency_names = sorted(set(task.dependencies) & failed)
                error = f"Skipped because dependencies failed: {', '.join(dependency_names)}"
                run = NamedAgentRun(
                    name=task.agent_name,
                    result=None,
                    task_id=task.task_id,
                    success=False,
                    error=error,
                )
                contribution = AgentContribution(
                    task_id=task.task_id,
                    agent_name=task.agent_name,
                    content=error,
                    success=False,
                    confidence=0.0,
                    error=error,
                )
                runs.append(run)
                contributions.append(contribution)
                await state.publish(contribution)
                failed.add(task.task_id)
                remaining.pop(task.task_id)

            ready = [
                task
                for task in remaining.values()
                if set(task.dependencies).issubset(completed)
            ]
            if not ready and remaining:
                unresolved = ", ".join(sorted(remaining))
                raise ValueError(f"Delegation plan contains a dependency cycle: {unresolved}")
            if not ready:
                continue

            # 原因：没有依赖关系的任务彼此独立，串行调用会累加模型等待时间。
            # 作用：同一依赖波次并行执行，并限制并发量，避免压垮本地模型服务。
            wave_results = await asyncio.gather(
                *(
                    self._execute_task(task, state, base_context, semaphore)
                    for task in ready
                )
            )
            for task, (run, contribution) in zip(ready, wave_results, strict=True):
                runs.append(run)
                contributions.append(contribution)
                await state.publish(contribution)
                (completed if contribution.success else failed).add(task.task_id)
                remaining.pop(task.task_id)

        return runs, contributions

    async def _execute_task(
        self,
        task: DelegatedTask,
        state: SharedAgentState,
        base_context: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[NamedAgentRun, AgentContribution]:
        async with semaphore:
            snapshot = await state.snapshot()
            dependency_results = {
                dependency: snapshot["contributions"][dependency].content
                for dependency in task.dependencies
            }
            task_context = {
                **base_context,
                **task.context,
                "multi_agent": {
                    "phase": "execution",
                    "objective": task.objective,
                    "task_id": task.task_id,
                    "agent_name": task.agent_name,
                    "dependency_results": dependency_results,
                    "shared_state": snapshot,
                },
                "shared_agent_state": state,
            }
            try:
                result = await self.agents[task.agent_name].run(
                    task.objective,
                    context=task_context,
                )
                success = result_success(result)
                content = result_content(result)
                # 原因：失败结果的 content 可能是内部 JSON，而类型化 error 才是真实故障原因。
                # 作用：调试记录保留 Review/Synthesis 的原始异常，不让诊断被中间输出覆盖。
                error = (
                    None
                    if success
                    else getattr(result, "error", None)
                    or content
                    or "Agent returned an unsuccessful result."
                )
            except Exception as exc:  # noqa: BLE001 - isolate one Agent failure.
                result = None
                success = False
                content = str(exc)
                error = f"{type(exc).__name__}: {exc}"
            run = NamedAgentRun(
                name=task.agent_name,
                result=result,
                task_id=task.task_id,
                success=success,
                error=error,
            )
            contribution = AgentContribution(
                task_id=task.task_id,
                agent_name=task.agent_name,
                content=content,
                success=success,
                confidence=result_confidence(result) if success else 0.0,
                raw=result,
                error=error,
            )
            return run, contribution

    async def _debate(
        self,
        objective: str,
        contributions: list[AgentContribution],
        state: SharedAgentState,
        base_context: dict[str, Any],
    ) -> list[DebateStatement]:
        successful = [item for item in contributions if item.success and item.content.strip()]
        participating_agents = list(dict.fromkeys(item.agent_name for item in successful))
        if self.debate_rounds == 0 or len(participating_agents) < 2:
            return []

        statements: list[DebateStatement] = []
        semaphore = asyncio.Semaphore(self.max_parallel)
        for round_number in range(1, self.debate_rounds + 1):
            candidate_text = "\n\n".join(
                f"[{item.agent_name}] {item.content}" for item in successful
            )
            prior_text = "\n".join(
                f"[{item.agent_name}] {item.content}" for item in statements
            )
            prompt = (
                f"Review the candidate conclusions for this objective: {objective}\n\n"
                f"Candidates:\n{candidate_text}\n\n"
                f"Earlier debate:\n{prior_text or 'None'}\n\n"
                "Identify agreements, factual conflicts, and the best corrected conclusion."
            )

            async def review(
                agent_name: str,
                current_round: int = round_number,
                current_prompt: str = prompt,
            ) -> DebateStatement:
                async with semaphore:
                    debate_context = {
                        **base_context,
                        "multi_agent": {
                            "phase": "debate",
                            "round": current_round,
                            "objective": objective,
                            "candidates": successful,
                            "previous_statements": list(statements),
                            "shared_state": await state.snapshot(),
                        },
                        "shared_agent_state": state,
                    }
                    try:
                        result = await self.agents[agent_name].run(
                            current_prompt,
                            context=debate_context,
                        )
                        content = result_content(result)
                    except Exception as exc:  # noqa: BLE001 - isolate one critic failure.
                        content = f"Debate failed: {type(exc).__name__}: {exc}"
                    return DebateStatement(
                        agent_name=agent_name,
                        round_number=current_round,
                        content=content,
                    )

            # 原因：同轮评论互不依赖，所有 Agent 必须看到相同候选结果。
            # 作用：同轮并行评论、跨轮累积观点，再交给最终仲裁器。
            round_statements = await asyncio.gather(
                *(review(agent_name) for agent_name in participating_agents)
            )
            statements.extend(round_statements)
            for statement in round_statements:
                await state.record_debate(statement)
        return statements


class MultiAgentCoordinator(MultiAgentSupervisor):
    """Backward-compatible name for the full multi-agent supervisor."""
