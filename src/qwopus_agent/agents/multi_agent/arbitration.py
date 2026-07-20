"""Conflict arbitration strategies for multi-agent results."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents.multi_agent._utils import extract_json_object, normalize_content
from qwopus_agent.agents.multi_agent.models import (
    AgentContribution,
    ArbitrationDecision,
    DebateStatement,
    ResultArbiter,
)
from qwopus_agent.llm import BaseLLM, ChatMessage


@dataclass
class ConsensusArbiter:
    """Resolve conflicts deterministically when no arbiter LLM is configured."""

    async def decide(
        self,
        objective: str,
        contributions: list[AgentContribution],
        debate: list[DebateStatement],
        context: dict[str, Any],
    ) -> ArbitrationDecision:
        """Prefer a strict content majority, then the highest-confidence result."""
        successful = [item for item in contributions if item.success and item.content.strip()]
        if not successful:
            errors = tuple(
                item.error or item.content
                for item in contributions
                if item.error or item.content
            )
            return ArbitrationDecision(
                final_answer="No agent completed the objective successfully.",
                rationale="All delegated tasks failed or returned empty content.",
                conflicts=errors,
            )

        groups: dict[str, list[AgentContribution]] = {}
        for item in successful:
            groups.setdefault(normalize_content(item.content), []).append(item)
        majority = max(groups.values(), key=len)
        if len(majority) > len(successful) / 2:
            selected = max(majority, key=lambda item: item.confidence)
            rationale = "Selected the conclusion supported by a strict agent majority."
        else:
            selected = max(
                enumerate(successful),
                key=lambda indexed: (indexed[1].confidence, indexed[0]),
            )[1]
            rationale = "No strict majority existed; selected the highest-confidence conclusion."

        distinct = [items[0].content for items in groups.values()]
        return ArbitrationDecision(
            final_answer=selected.content,
            rationale=rationale,
            selected_agents=tuple(
                item.agent_name
                for item in groups[normalize_content(selected.content)]
            ),
            conflicts=tuple(distinct) if len(distinct) > 1 else (),
        )


@dataclass
class LLMResultArbiter:
    """Use a BaseLLM to synthesize disputed conclusions into one final answer."""

    llm: BaseLLM
    fallback: ResultArbiter = field(default_factory=ConsensusArbiter)

    async def decide(
        self,
        objective: str,
        contributions: list[AgentContribution],
        debate: list[DebateStatement],
        context: dict[str, Any],
    ) -> ArbitrationDecision:
        """Synthesize candidates and fall back to deterministic arbitration on errors."""
        payload = {
            "objective": objective,
            "candidates": [
                {
                    "agent": item.agent_name,
                    "content": item.content[:6000],
                    "confidence": item.confidence,
                }
                for item in contributions
                if item.success
            ],
            "debate": [
                {
                    "agent": item.agent_name,
                    "round": item.round_number,
                    "content": item.content[:4000],
                }
                for item in debate
            ],
        }
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Resolve the candidate conclusions. Return JSON only with final_answer, "
                    "rationale, selected_agents, and conflicts. Do not expose hidden reasoning."
                ),
            ),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        try:
            response = await asyncio.to_thread(
                self.llm.generate,
                messages,
                temperature=0.1,
                max_tokens=1600,
            )
            result = extract_json_object(response.content)
            final_answer = str(result["final_answer"]).strip()
            if not final_answer:
                raise ValueError("Arbiter returned an empty final answer.")
            return ArbitrationDecision(
                final_answer=final_answer,
                rationale=str(result.get("rationale", "LLM arbitration completed.")),
                selected_agents=tuple(str(value) for value in result.get("selected_agents", ())),
                conflicts=tuple(str(value) for value in result.get("conflicts", ())),
            )
        except Exception:  # noqa: BLE001 - any provider failure must use deterministic fallback.
            # 原因：部分兼容模型会拒绝 JSON 格式或在仲裁请求中返回服务端错误。
            # 作用：保留已完成 Agent 结果，并由 ConsensusArbiter 产生最终答案。
            return await self.fallback.decide(objective, contributions, debate, context)
