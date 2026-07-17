import asyncio
import unittest
from dataclasses import dataclass
from typing import Any

from qwopus_agent.agents import MultiAgentCoordinator


@dataclass
class FakeAgent:
    label: str

    async def run(self, question: str, context: dict[str, Any] | None = None) -> str:
        return f"{self.label}: {question}: {context or {}}"


class MultiAgentCoordinatorTests(unittest.TestCase):
    def test_multi_agent_coordinator_runs_agents_in_order(self) -> None:
        coordinator = MultiAgentCoordinator(
            agents={
                "planner": FakeAgent("planner"),
                "critic": FakeAgent("critic"),
            }
        )

        # 原因：多 Agent 当前先提供可测的顺序编排，不直接实现复杂协商。
        # 作用：验证 coordinator 能按指定 order 调用多个 Agent。
        result = asyncio.run(
            coordinator.run("Research topic", context={"depth": "light"}, order=["critic", "planner"])
        )

        self.assertEqual([run.name for run in result.runs], ["critic", "planner"])
        self.assertIn("critic: Research topic", result.runs[0].result)
        self.assertIn("planner: Research topic", result.runs[1].result)


if __name__ == "__main__":
    unittest.main()
