"""Typed application contracts for every Qwopus-Agent entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from qwopus_agent.agents.multi_agent import MultiAgentRun
    from qwopus_agent.analysis import AnalysisResult
    from qwopus_agent.reports import GeneratedReport


class ConversationTurn(BaseModel):
    """One prior user-visible chat turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class OrchestrationFile(BaseModel):
    """Framework-neutral uploaded file payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    content: bytes


class OrchestrationRequest(BaseModel):
    """Single request contract shared by CLI, UI, and future APIs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str
    history: tuple[ConversationTurn, ...] = ()
    uploaded_files: tuple[OrchestrationFile, ...] = ()
    enable_web_search: bool = False
    enable_local_knowledge: bool = False
    generate_report: bool = False
    report_title: str = "Qwopus Agent Report"
    report_basename: str = "qwopus_agent_report"


class SourceCitation(BaseModel):
    """One safe source reference extracted from a Tool result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["local", "web"]
    source: str
    page: str | None = None
    url: str | None = None


class ProcessEvent(BaseModel):
    """Safe execution event that never contains model thoughts or Tool bodies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str
    status: Literal["started", "completed", "warning", "failed"]
    agent: str | None = None
    tool: str | None = None
    message: str = ""
    duration_seconds: float | None = None


@dataclass(frozen=True)
class OrchestrationResult:
    """Unified result envelope with optional typed domain artifacts."""

    success: bool
    final_answer: str
    route: Literal["single_agent", "multi_agent"]
    citations: tuple[SourceCitation, ...] = ()
    trace: tuple[ProcessEvent, ...] = ()
    analysis_result: AnalysisResult | None = None
    report: GeneratedReport | None = None
    multi_agent_run: MultiAgentRun | None = None
