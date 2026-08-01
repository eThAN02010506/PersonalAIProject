"""Typed results returned by smolagents integration runners."""

from __future__ import annotations

from dataclasses import dataclass, field

from qwopus_agent.integrations import smolagents_debug

AgentDebugRun = smolagents_debug.AgentDebugRun


@dataclass(frozen=True)
class DocumentAnalysisRun:
    """Document analysis answer with a UI-visible debug trace."""

    answer: str

    debug_steps: list[str]

    tool_calls: list[str] = field(default_factory=list)

    inspected_file_names: tuple[str, ...] = ()

    debug_runs: tuple[AgentDebugRun, ...] = ()

    generation_mode: str = "model"


@dataclass(frozen=True)
class ChatAgentRun:
    """Safe structured result from one smolagents chat run."""

    answer: str
    tool_calls: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    state: str | None = None
    debug_runs: tuple[AgentDebugRun, ...] = ()
    success: bool = True
    error: str | None = None
