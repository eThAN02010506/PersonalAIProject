"""Lightweight task reflection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReflectionResult:
    """Structured reflection output for one task."""

    needs_retry: bool

    observations: list[str] = field(default_factory=list)

    suggestions: list[str] = field(default_factory=list)


@dataclass
class TaskReflectionEvaluator:
    """Evaluate task outputs without calling an LLM."""

    min_answer_chars: int = 80

    def evaluate(
        self,
        objective: str,
        answer: str,
        success: bool,
        step_names: list[str] | None = None,
    ) -> ReflectionResult:
        """Return a small reflection result for orchestration."""
        observations: list[str] = []
        suggestions: list[str] = []
        step_names = step_names or []

        if not success:
            observations.append("Task execution failed.")
            suggestions.append("Inspect failed skill output before retrying.")
        if not answer.strip():
            observations.append("No final answer was produced.")
            suggestions.append("Ask the executor or model to produce a final answer.")
        elif len(answer.strip()) < self.min_answer_chars:
            observations.append("Final answer is very short.")
            suggestions.append(
                "Request a more complete answer when the user did not ask for brevity."
            )
        if not step_names:
            observations.append("No execution steps were recorded.")
            suggestions.append("Route the task through Planner and Executor for traceability.")

        # 原因：Reflection 当前阶段不负责自动重跑任务，只给 Router/服务层可用的判断。
        # 作用：把“是否需要重试”和“为什么”结构化，后续可接入更强反思策略。
        return ReflectionResult(
            needs_retry=bool(observations) or not success,
            observations=observations,
            suggestions=suggestions,
        )
