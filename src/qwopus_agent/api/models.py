"""Typed HTTP contracts for the Qwopus-Agent frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConversationCreate(BaseModel):
    """Request for a blank conversation."""

    title: str = "New chat"


class ConversationUpdate(BaseModel):
    """Request for changing a conversation title."""

    title: str = Field(min_length=1, max_length=80)


class ConversationView(BaseModel):
    """Conversation metadata displayed in the sidebar."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: str
    updated_at: str


class MessageView(BaseModel):
    """One persisted user-visible message."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ChatStartRequest(BaseModel):
    """Start one background Agent turn."""

    content: str = Field(min_length=1)
    enable_web_search: bool = False
    enable_local_knowledge: bool = False
    # 原因：全局知识可能包含其他聊天上传的内容，不能跟随 Knowledge 默认自动授权。
    # 作用：只有用户在当前请求显式开启时，Agent 才会收到全局检索 Tool。
    include_global_knowledge: bool = False
    min_source_relevance: float = Field(default=0.55, ge=0.25, le=0.95)

    @model_validator(mode="after")
    def validate_global_permission(self) -> ChatStartRequest:
        """Reject a global permission bit without the parent knowledge capability."""
        if self.include_global_knowledge and not self.enable_local_knowledge:
            raise ValueError("Global knowledge requires local knowledge permission.")
        return self


class RunStarted(BaseModel):
    """Identifier returned while a background run is active."""

    run_id: str
    status: Literal["running"] = "running"


class RunView(BaseModel):
    """Poll response for one chat run."""

    run_id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    phase: str = "connecting"
    answer: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class AnalysisView(BaseModel):
    """Final document-analysis response and downloadable artifacts."""

    answer: str
    route: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    reports: list[dict[str, str]] = Field(default_factory=list)
    documents: list[DocumentOutlineView] = Field(default_factory=list)


class DocumentSectionView(BaseModel):
    """One selectable heading shown without exposing its full body."""

    id: str
    title: str
    level: int
    parent_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class DocumentOutlineView(BaseModel):
    """Safe document hierarchy returned to the formal frontend."""

    document_id: str
    source: str
    sections: list[DocumentSectionView] = Field(default_factory=list)


class SavedDocumentView(BaseModel):
    """One locally persisted, successfully parsed document."""

    document_id: str
    source: str
    file_type: str
    size_bytes: int
    section_count: int
    saved_at: str
    summary_available: bool


class ModelSettingsUpdate(BaseModel):
    """Request to select a remote endpoint or launch a local MLX model."""

    mode: Literal["remote", "local"]
    base_url: str | None = None
    model_path: str | None = None
    context_window_tokens: int = Field(default=32768, ge=2048)
    agent_mode: Literal["tool_calling", "code"] = "tool_calling"
    supports_structured_output: bool = False
    supports_vision: bool = False


class ModelSettingsView(BaseModel):
    """Current runtime model selection and connection state."""

    status: Literal["ok"] = "ok"
    mode: Literal["remote", "local"]
    model_online: bool
    message: str
    model: str
    base_url: str
    local_model_path: str | None = None
    context_window_tokens: int
    agent_mode: Literal["tool_calling", "code"]
    supports_structured_output: bool
    supports_vision: bool


class DebugRuntimeLogView(BaseModel):
    """Bounded runtime-log snapshot shown only by the local debug console."""

    path: str
    exists: bool
    size_bytes: int = 0
    modified_at: str | None = None
    total_lines: int = 0
    lines: list[str] = Field(default_factory=list)
    error: str | None = None


class DebugRecordSummaryView(BaseModel):
    """Small immutable run summary used by the auto-refreshing record list."""

    id: str
    timestamp: str | None = None
    source: str = "unknown"
    status: str = "unknown"
    run_id: str
    result_preview: str = ""
    trace_events: int = 0
    agent_runs: int = 0


class DebugOverviewView(BaseModel):
    """Complete read-only diagnostic snapshot for the React debug console."""

    generated_at: str
    uptime_seconds: float
    process_id: int
    python_version: str
    platform: str
    model: ModelSettingsView
    active_runs: int
    completed_runs: int
    record_count: int
    record_storage_bytes: int
    source_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    records: list[DebugRecordSummaryView] = Field(default_factory=list)
    runtime_log: DebugRuntimeLogView
