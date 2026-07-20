"""Concurrency-safe state shared by supervised agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents.multi_agent.models import AgentContribution, DebateStatement


@dataclass
class SharedAgentState:
    """State exchanged between delegated task waves."""

    values: dict[str, Any] = field(default_factory=dict)
    contributions: dict[str, AgentContribution] = field(default_factory=dict)
    transcript: list[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def snapshot(self) -> dict[str, Any]:
        """Return a stable public snapshot for an agent invocation."""
        async with self._lock:
            return {
                "values": dict(self.values),
                "contributions": dict(self.contributions),
                "transcript": list(self.transcript),
            }

    async def publish(self, contribution: AgentContribution) -> None:
        """Publish one completed task result atomically."""
        async with self._lock:
            # 原因：并行 Agent 可能同时完成，分散写入会让共享状态只保留部分结果。
            # 作用：原子更新任务、Agent 快照和审计记录，供后续依赖任务读取。
            self.contributions[contribution.task_id] = contribution
            self.values[f"task:{contribution.task_id}"] = contribution.content
            self.values[f"agent:{contribution.agent_name}"] = contribution.content
            self.transcript.append(
                f"{contribution.task_id}:{contribution.agent_name}:"
                f"{'success' if contribution.success else 'failed'}"
            )

    async def record_debate(self, statement: DebateStatement) -> None:
        """Record a debate statement without changing task outputs."""
        async with self._lock:
            self.transcript.append(
                f"debate:{statement.round_number}:{statement.agent_name}:{statement.content}"
            )
