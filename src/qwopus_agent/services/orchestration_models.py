"""Typed application contracts for every Qwopus-Agent entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


InterpretationMode = Literal["precise", "contextual", "exploratory"]
TaskType = Literal[
    "answer",
    "explain",
    "how_to",
    "compare",
    "summarize",
    "analyze",
    "report",
    "continue",
]


class ContextReference(BaseModel):
    """One trusted context item used to resolve the current request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["conversation", "task", "document", "skill", "preference"]
    identifier: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ContextSnapshot(BaseModel):
    """Bounded conversation facts available to the intent resolver."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_id: str | None = None
    previous_objective: str | None = None
    open_tasks: tuple[str, ...] = ()
    document_sources: tuple[str, ...] = ()
    active_skill_names: tuple[str, ...] = ()


class AnswerContract(BaseModel):
    """Task-specific requirements that make answer depth testable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: TaskType = "answer"
    complexity: Literal["simple", "standard", "complex"] = "standard"
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed"
    response_language: str = "auto"
    output_format: str = "adaptive"
    required_facets: tuple[str, ...] = ()


class ResolvedIntent(BaseModel):
    """Operational objective produced from literal text and authorized context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_request: str = Field(min_length=1)
    operational_objective: str = Field(min_length=1)
    interpretation_mode: InterpretationMode = "contextual"
    task_type: TaskType = "answer"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    context_references: tuple[ContextReference, ...] = ()
    assumptions: tuple[str, ...] = ()
    requires_clarification: bool = False
    clarification_question: str | None = None
    answer_contract: AnswerContract = Field(default_factory=AnswerContract)

    @model_validator(mode="after")
    def validate_clarification(self) -> ResolvedIntent:
        """Require an actionable question whenever execution must pause."""
        if self.requires_clarification and not (
            self.clarification_question and self.clarification_question.strip()
        ):
            raise ValueError("A clarification question is required.")
        return self


class ConversationTaskState(BaseModel):
    """Structured state from the last successful task in one conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    last_successful_objective: str | None = None
    last_task_type: TaskType | None = None
    last_answer_contract: AnswerContract | None = None
    active_document_sources: tuple[str, ...] = ()
    last_skill_name: str | None = None
    last_skill_version: str | None = None
    open_tasks: tuple[str, ...] = ()
    updated_at: str = ""


class OrchestrationFile(BaseModel):
    """Framework-neutral uploaded or explicitly selected local file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    content: bytes | None = None
    local_path: Path | None = None

    @model_validator(mode="after")
    def validate_source(self) -> OrchestrationFile:
        """Require exactly one file source."""
        if (self.content is None) == (self.local_path is None):
            raise ValueError("Provide exactly one of content or local_path.")
        return self


class OrchestrationRequest(BaseModel):
    """Single request contract shared by CLI, UI, and future APIs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str
    # 原因：原始文字和经上下文解析的执行目标必须同时保留，不能由 Prompt 临时改写。
    # 作用：Planner 使用稳定的操作目标，最终回答仍可遵循用户原话和语言。
    resolved_intent: ResolvedIntent | None = None
    interpretation_mode: InterpretationMode = "contextual"
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
    # 原因：回答详略属于用户本轮偏好，不能靠修改模型全局 token 上限实现。
    # 作用：统一传递简洁、标准和详细三档策略，同时让简单问题仍可快速回答。
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed"
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
    # 原因：父进程只有在任务成功后才能安全更新会话状态。
    # 作用：把本轮解析结果和状态更新作为内部产物返回，不让 worker 直接写 SQLite。
    resolved_intent: ResolvedIntent | None = None
    task_state: ConversationTaskState | None = None
    # 原因：本地调试台需要完整 Agent 步骤，但公开 API 与正式前端不能自动暴露这些内容。
    # 作用：把 raw debug 作为内部结果旁路传递；FastAPI response models 明确忽略该字段。
    debug_runs: tuple[AgentDebugRun, ...] = ()
