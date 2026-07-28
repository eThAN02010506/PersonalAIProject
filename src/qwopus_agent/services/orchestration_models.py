"""Typed application contracts for every Qwopus-Agent entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from qwopus_agent.agents.multi_agent import MultiAgentRun
    from qwopus_agent.analysis import AnalysisResult
    from qwopus_agent.integrations.smolagents_runtime import AgentDebugRun
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
    # 原因：开启本地知识时，全局 MiniRAG 会让一个聊天检索到其他聊天上传的文件。
    # 作用：把隔离键作为编排请求的一部分，所有知识 Tool 只能打开当前会话目录。
    conversation_id: str | None = None
    history: tuple[ConversationTurn, ...] = ()
    uploaded_files: tuple[OrchestrationFile, ...] = ()
    enable_web_search: bool = False
    enable_local_knowledge: bool = False
    include_global_knowledge: bool = False
    # 原因：不同问题对召回率和精度的要求不同，不能使用全局可变阈值。
    # 作用：把用户选择固定在单次编排请求中，并限制为索引支持的安全范围。
    min_source_relevance: float = Field(default=0.55, ge=0.25, le=0.95)
    # 原因：问题检索、章节阅读和全文总结需要不同的工具策略。
    # 作用：把用户选择作为请求数据传入 Agent，而不是由前端改写自然语言问题。
    analysis_mode: Literal["question", "section", "full"] = "question"
    # 原因：章节选择属于单次请求范围，不能写入进程级全局状态。
    # 作用：键为文档 id、值为章节 id，文档工具据此限制可读证据。
    selected_sections: dict[str, tuple[str, ...]] = Field(default_factory=dict)
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
    # 原因：本地调试台需要完整 Agent 步骤，但公开 API 与正式前端不能自动暴露这些内容。
    # 作用：把 raw debug 作为内部结果旁路传递；FastAPI response models 明确忽略该字段。
    debug_runs: tuple[AgentDebugRun, ...] = ()
