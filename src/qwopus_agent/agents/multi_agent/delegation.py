"""Task delegation strategies for the multi-agent supervisor."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents.multi_agent._utils import (
    extract_json_object,
    json_safe,
    safe_identifier,
)
from qwopus_agent.agents.multi_agent.models import (
    AgentProfile,
    DelegatedTask,
    DelegationPlan,
    TaskDelegator,
)
from qwopus_agent.llm import BaseLLM, ChatMessage


@dataclass
class DeterministicTaskDelegator:
    """Create predictable task assignments without requiring an LLM."""

    async def create_plan(
        self,
        objective: str,
        profiles: dict[str, AgentProfile],
        context: dict[str, Any],
    ) -> DelegationPlan:
        """Use explicit assignments when supplied, otherwise assign every agent."""
        configured = context.get("delegations")
        if isinstance(configured, list):
            tasks = tuple(
                DelegatedTask(
                    task_id=str(item.get("task_id") or f"task_{index}"),
                    objective=str(item.get("objective") or objective),
                    agent_name=str(item.get("agent_name") or ""),
                    dependencies=tuple(str(value) for value in item.get("dependencies", ())),
                    context=dict(item.get("context", {})),
                )
                for index, item in enumerate(configured, start=1)
                if isinstance(item, dict)
            )
            return DelegationPlan(objective=objective, tasks=tasks)

        tasks = tuple(
            DelegatedTask(
                task_id=f"task_{index}_{safe_identifier(name)}",
                objective=objective,
                agent_name=name,
            )
            for index, name in enumerate(sorted(profiles), start=1)
        )
        return DelegationPlan(objective=objective, tasks=tasks)


@dataclass
class LLMTaskDelegator:
    """Ask any BaseLLM implementation to select agents and dependencies."""

    llm: BaseLLM
    fallback: TaskDelegator = field(default_factory=DeterministicTaskDelegator)

    async def create_plan(
        self,
        objective: str,
        profiles: dict[str, AgentProfile],
        context: dict[str, Any],
    ) -> DelegationPlan:
        """Generate a JSON plan and fall back safely when model output is invalid."""
        profile_payload = [
            {
                "name": profile.name,
                "description": profile.description,
                "capabilities": list(profile.capabilities),
            }
            for profile in profiles.values()
        ]
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a multi-agent supervisor. Return JSON only with a tasks array. "
                    "Each task needs task_id, objective, agent_name, and dependencies. "
                    "Use only the supplied agent names and dependency task IDs."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "objective": objective,
                        "agents": profile_payload,
                        "context": json_safe(context),
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
        try:
            # 原因：BaseLLM 是同步抽象，直接调用会阻塞并行 Agent 的事件循环。
            # 作用：在线程中完成模型规划，同时保持 Supervisor 的异步调度能力。
            response = await asyncio.to_thread(
                self.llm.generate,
                messages,
                temperature=0.1,
                max_tokens=1200,
            )
            payload = extract_json_object(response.content)
            tasks = tuple(
                DelegatedTask(
                    task_id=str(item["task_id"]),
                    objective=str(item["objective"]),
                    agent_name=str(item["agent_name"]),
                    dependencies=tuple(str(value) for value in item.get("dependencies", ())),
                    context=dict(item.get("context", {})),
                )
                for item in payload["tasks"]
            )
            plan = DelegationPlan(objective=objective, tasks=tasks)
            validate_plan(plan, profiles)
            return plan
        except Exception:  # noqa: BLE001 - any provider failure must use deterministic fallback.
            # 原因：任意模型适配器可能抛出 HTTP、解析或本地推理异常。
            # 作用：委派失败时退回确定性计划，不让模型实现细节中断 Supervisor。
            return await self.fallback.create_plan(objective, profiles, context)


def validate_plan(plan: DelegationPlan, profiles: dict[str, AgentProfile]) -> None:
    """Reject malformed, unsafe, or unresolvable delegation plans."""
    if not plan.tasks:
        raise ValueError("Delegation plan must contain at least one task.")
    task_ids = [task.task_id for task in plan.tasks]
    if any(not task_id for task_id in task_ids):
        raise ValueError("Every delegated task requires a task_id.")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Delegated task IDs must be unique.")
    known_ids = set(task_ids)
    for task in plan.tasks:
        if task.agent_name not in profiles:
            raise ValueError(f"Unknown delegated agent: {task.agent_name}")
        unknown_dependencies = set(task.dependencies) - known_ids
        if unknown_dependencies:
            raise ValueError(
                f"Task {task.task_id} has unknown dependencies: {sorted(unknown_dependencies)}"
            )
        if task.task_id in task.dependencies:
            raise ValueError(f"Task {task.task_id} cannot depend on itself.")
