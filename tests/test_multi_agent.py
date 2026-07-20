import asyncio
import json
import unittest
from dataclasses import dataclass, field
from typing import Any

from qwopus_agent.agents import (
    AgentContribution,
    AgentProfile,
    DelegatedTask,
    DelegationPlan,
    LLMResultArbiter,
    LLMTaskDelegator,
    MultiAgentCoordinator,
    MultiAgentSupervisor,
)
from qwopus_agent.llm import BaseLLM, ChatMessage, LLMResponse


@dataclass
class FakeAgent:
    label: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def run(self, question: str, context: dict[str, Any] | None = None) -> Any:
        self.calls.append({"question": question, "context": context or {}})
        phase = (context or {}).get("multi_agent", {}).get("phase")
        if phase == "debate":
            return f"{self.label} reviewed all candidates"
        return f"{self.label}: {question}"


@dataclass
class StaticDelegator:
    plan: DelegationPlan

    async def create_plan(
        self,
        objective: str,
        profiles: dict[str, AgentProfile],
        context: dict[str, Any],
    ) -> DelegationPlan:
        return self.plan


class FakeLLM(BaseLLM):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=f"```json\n{json.dumps(self.payload)}\n```",
            model="fake",
        )


class FailingLLM(BaseLLM):
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise RuntimeError("provider returned HTTP 500")


class MultiAgentSupervisorTests(unittest.TestCase):
    def test_compatibility_coordinator_preserves_requested_order(self) -> None:
        coordinator = MultiAgentCoordinator(
            agents={
                "planner": FakeAgent("planner"),
                "critic": FakeAgent("critic"),
            },
            debate_rounds=0,
        )

        result = asyncio.run(
            coordinator.run(
                "Research topic",
                context={"depth": "light"},
                order=["critic", "planner"],
            )
        )

        self.assertEqual([run.name for run in result.runs], ["critic", "planner"])
        self.assertIn("critic: Research topic", result.runs[0].result)
        dependency_results = result.shared_state["contributions"]
        self.assertEqual(len(dependency_results), 2)

    def test_independent_tasks_execute_in_parallel(self) -> None:
        async def scenario() -> tuple[bool, bool]:
            both_started = asyncio.Event()
            started: set[str] = set()

            @dataclass
            class GateAgent:
                name: str

                async def run(
                    self,
                    question: str,
                    context: dict[str, Any] | None = None,
                ) -> str:
                    started.add(self.name)
                    if len(started) == 2:
                        both_started.set()
                    await asyncio.wait_for(both_started.wait(), timeout=0.5)
                    return self.name

            supervisor = MultiAgentSupervisor(
                agents={"a": GateAgent("a"), "b": GateAgent("b")},
                max_parallel=2,
                debate_rounds=0,
            )
            result = await supervisor.run("parallel")
            return result.runs[0].success, result.runs[1].success

        self.assertEqual(asyncio.run(scenario()), (True, True))

    def test_dependency_task_receives_prior_result_and_shared_state(self) -> None:
        first = FakeAgent("collector")

        @dataclass
        class DependentAgent:
            received: dict[str, Any] = field(default_factory=dict)

            async def run(
                self,
                question: str,
                context: dict[str, Any] | None = None,
            ) -> str:
                self.received = (context or {})["multi_agent"]
                return "synthesis"

        second = DependentAgent()
        plan = DelegationPlan(
            objective="compose",
            tasks=(
                DelegatedTask("collect", "collect evidence", "collector"),
                DelegatedTask("write", "write answer", "writer", ("collect",)),
            ),
        )
        supervisor = MultiAgentSupervisor(
            agents={"collector": first, "writer": second},
            delegator=StaticDelegator(plan),
            debate_rounds=0,
        )

        result = asyncio.run(supervisor.run("compose"))

        self.assertEqual(
            second.received["dependency_results"],
            {"collect": "collector: collect evidence"},
        )
        self.assertIn("task:collect", second.received["shared_state"]["values"])
        self.assertEqual(result.final_answer, "synthesis")

    def test_failed_dependency_skips_downstream_task(self) -> None:
        @dataclass
        class FailingAgent:
            async def run(
                self,
                question: str,
                context: dict[str, Any] | None = None,
            ) -> str:
                raise RuntimeError("source unavailable")

        writer = FakeAgent("writer")
        plan = DelegationPlan(
            objective="compose",
            tasks=(
                DelegatedTask("collect", "collect", "collector"),
                DelegatedTask("write", "write", "writer", ("collect",)),
            ),
        )
        supervisor = MultiAgentSupervisor(
            agents={"collector": FailingAgent(), "writer": writer},
            delegator=StaticDelegator(plan),
            debate_rounds=0,
        )

        result = asyncio.run(supervisor.run("compose"))

        self.assertFalse(result.runs[0].success)
        self.assertFalse(result.runs[1].success)
        self.assertIn("dependencies failed", result.runs[1].error or "")
        self.assertEqual(writer.calls, [])

    def test_dependency_cycle_is_rejected(self) -> None:
        plan = DelegationPlan(
            objective="cycle",
            tasks=(
                DelegatedTask("a", "a", "a", ("b",)),
                DelegatedTask("b", "b", "b", ("a",)),
            ),
        )
        supervisor = MultiAgentSupervisor(
            agents={"a": FakeAgent("a"), "b": FakeAgent("b")},
            delegator=StaticDelegator(plan),
            debate_rounds=0,
        )

        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            asyncio.run(supervisor.run("cycle"))

    def test_agents_debate_and_conflict_is_arbitrated(self) -> None:
        @dataclass
        class ConfidentAgent(FakeAgent):
            confidence: float = 0.5

            async def run(
                self,
                question: str,
                context: dict[str, Any] | None = None,
            ) -> Any:
                phase = (context or {}).get("multi_agent", {}).get("phase")
                self.calls.append({"question": question, "context": context or {}})
                if phase == "debate":
                    return f"{self.label} critique"
                return {
                    "content": f"answer from {self.label}",
                    "confidence": self.confidence,
                }

        low = ConfidentAgent("low", confidence=0.2)
        high = ConfidentAgent("high", confidence=0.9)
        supervisor = MultiAgentSupervisor(
            agents={"low": low, "high": high},
            debate_rounds=1,
        )

        result = asyncio.run(supervisor.run("resolve"))

        self.assertEqual(len(result.debate), 2)
        self.assertEqual(result.final_answer, "answer from high")
        self.assertEqual(result.decision.selected_agents, ("high",))
        self.assertEqual(len(result.decision.conflicts), 2)
        self.assertEqual(len(low.calls), 2)
        self.assertEqual(len(high.calls), 2)

    def test_llm_delegator_parses_json_plan(self) -> None:
        delegator = LLMTaskDelegator(
            llm=FakeLLM(
                {
                    "tasks": [
                        {
                            "task_id": "research",
                            "objective": "find evidence",
                            "agent_name": "researcher",
                            "dependencies": [],
                        },
                        {
                            "task_id": "review",
                            "objective": "review evidence",
                            "agent_name": "critic",
                            "dependencies": ["research"],
                        },
                    ]
                }
            )
        )

        plan = asyncio.run(
            delegator.create_plan(
                "answer",
                {
                    "researcher": AgentProfile("researcher", capabilities=("web",)),
                    "critic": AgentProfile("critic", capabilities=("review",)),
                },
                {},
            )
        )

        self.assertEqual([task.agent_name for task in plan.tasks], ["researcher", "critic"])
        self.assertEqual(plan.tasks[1].dependencies, ("research",))

    def test_llm_components_fall_back_when_any_model_provider_fails(self) -> None:
        profiles = {
            "researcher": AgentProfile("researcher"),
            "reviewer": AgentProfile("reviewer"),
        }
        plan = asyncio.run(
            LLMTaskDelegator(FailingLLM()).create_plan("answer", profiles, {})
        )
        decision = asyncio.run(
            LLMResultArbiter(FailingLLM()).decide(
                "answer",
                [
                    AgentContribution(
                        task_id="result",
                        agent_name="researcher",
                        content="verified answer",
                        success=True,
                        confidence=0.9,
                    )
                ],
                [],
                {},
            )
        )

        # 原因：可插拔模型的错误类型不受 Multi-Agent 层控制。
        # 作用：验证 HTTP/本地推理异常不会丢失确定性委派和已有 Agent 结论。
        self.assertEqual([task.agent_name for task in plan.tasks], ["researcher", "reviewer"])
        self.assertEqual(decision.final_answer, "verified answer")


if __name__ == "__main__":
    unittest.main()
