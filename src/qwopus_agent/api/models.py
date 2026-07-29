"""Typed HTTP contracts for the Qwopus-Agent frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qwopus_agent.documents.local_folder import MAX_LOCAL_FOLDER_SELECTION
from qwopus_agent.services.orchestration_models import InterpretationMode


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
    # 原因：浏览器可渲染动态站点，权限范围大于 Tavily 搜索，必须由用户单独开启。
    # 作用：HTTP 请求把 Browser 授权固定在当前聊天轮次，不形成全局状态。
    enable_browser: bool = False
    enable_local_knowledge: bool = False
    # 原因：全局知识可能包含其他聊天上传的内容，不能跟随 Knowledge 默认自动授权。
    # 作用：只有用户在当前请求显式开启时，Agent 才会收到全局检索 Tool。
    include_global_knowledge: bool = False
    min_source_relevance: float = Field(default=0.55, ge=0.25, le=0.95)
    # 原因：用户需要控制答案的信息密度，而不是被固定最少字数拖慢每次生成。
    # 作用：默认请求详细答案，并允许前端按当前问题切换详略。
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed"
    # 原因：抽象请求可以按字面、对话上下文或探索式扩展产生不同的安全执行目标。
    # 作用：前端显式控制解析半径，默认只使用当前对话内已授权的上下文。
    interpretation_mode: InterpretationMode = "contextual"

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


class SourceCoverageView(BaseModel):
    """Auditable coverage of the exact document set selected for one analysis."""

    required_sources: list[str] = Field(default_factory=list)
    covered_sources: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    complete: bool = False


class SpreadsheetSheetView(BaseModel):
    """Safe structural summary for one worksheet."""

    name: str
    kind: Literal["empty", "table", "multi_table", "form", "matrix"]
    region_count: int = Field(ge=0)
    formula_count: int = Field(ge=0)
    merged_range_count: int = Field(ge=0)
    chart_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    data_validation_count: int = Field(ge=0)
    hidden: bool = False


class SpreadsheetTableView(BaseModel):
    """Safe schema for one dataframe exposed to the pandas sandbox."""

    name: str
    source_sheet: str
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    column_names: list[str] = Field(default_factory=list)
    columns_truncated: bool = False


class SpreadsheetWorkbookView(BaseModel):
    """Workbook profile shown without raw spreadsheet cell values."""

    source: str
    sheet_count: int = Field(ge=0)
    formula_count: int = Field(ge=0)
    merged_range_count: int = Field(ge=0)
    chart_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    data_validation_count: int = Field(ge=0)
    sheets: list[SpreadsheetSheetView] = Field(default_factory=list)
    tables: list[SpreadsheetTableView] = Field(default_factory=list)


class SkillStepView(BaseModel):
    """One reviewed step without persisted argument values."""

    skill_name: str


class SkillVersionView(BaseModel):
    """Safe lifecycle record for one reusable Skill version."""

    name: str
    version: str
    description: str
    status: Literal["candidate", "active", "archived", "rejected"]
    created_at: str
    source_run_id: str | None = None
    source_model: str | None = None
    intent_examples: list[str] = Field(default_factory=list)
    steps: list[SkillStepView] = Field(default_factory=list)
    spec_valid: bool = False


class SkillCapabilityView(BaseModel):
    """One existing Skill that may be granted to the authoring model."""

    name: str
    description: str


class SkillAuthoringRequest(BaseModel):
    """Bounded authoring input submitted from the local Debug Console."""

    goal: str = Field(min_length=3, max_length=2_000)
    requested_name: str | None = Field(
        default=None,
        max_length=90,
        pattern=r"^[a-zA-Z0-9_]*$",
    )
    intent_examples: list[str] = Field(default_factory=list, max_length=8)
    allowed_skills: list[str] = Field(
        min_length=1,
        max_length=8,
    )


class SkillCandidateCheckView(BaseModel):
    """One validation check displayed before promotion."""

    name: str
    passed: bool
    detail: str


class SkillCandidateReviewView(BaseModel):
    """Full local-only review material for a generated Workflow candidate."""

    skill: SkillVersionView
    spec_json: str
    diff: str
    checks: list[SkillCandidateCheckView] = Field(default_factory=list)
    model_output: str | None = None


class SkillCandidateTestRequest(BaseModel):
    """Dry-run query used without calling real providers."""

    query: str = Field(min_length=1, max_length=2_000)


class SkillCandidateTestStepView(BaseModel):
    """One rendered step from the side-effect-free candidate dry run."""

    skill_name: str
    query: str
    argument_keys: list[str] = Field(default_factory=list)


class SkillCandidateTestView(BaseModel):
    """Dry-run result returned to the Debug Console."""

    success: bool
    output: str
    steps: list[SkillCandidateTestStepView] = Field(default_factory=list)


class AnalysisView(BaseModel):
    """Final document-analysis response and downloadable artifacts."""

    answer: str
    route: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    reports: list[dict[str, str]] = Field(default_factory=list)
    documents: list[DocumentOutlineView] = Field(default_factory=list)
    spreadsheets: list[SpreadsheetWorkbookView] = Field(default_factory=list)
    source_coverage: SourceCoverageView | None = None
    generation_mode: str | None = None


class LocalFolderScanRequest(BaseModel):
    """Local path submitted for safe server-side discovery."""

    path: str = Field(min_length=1)


class LocalFolderNodeView(BaseModel):
    """One selectable directory or file in the scanned tree."""

    name: str
    relative_path: str
    kind: Literal["directory", "file"]
    children: list[LocalFolderNodeView] = Field(default_factory=list)


class LocalFolderTreeView(BaseModel):
    """Filtered local-folder tree returned to the frontend."""

    root: str
    file_count: int
    max_selection: int
    tree: LocalFolderNodeView


class LocalFolderAnalysisRequest(BaseModel):
    """Analyze an explicit subset of a previously scanned local folder."""

    conversation_id: str = Field(min_length=1)
    root: str = Field(min_length=1)
    selected_files: list[str] = Field(
        min_length=1,
        max_length=MAX_LOCAL_FOLDER_SELECTION,
    )
    question: str = ""
    generate_report: bool = False
    analysis_mode: Literal["question", "section", "full"] = "question"
    selected_sections: dict[str, tuple[str, ...]] = Field(default_factory=dict)


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


class SavedDocumentsAttachRequest(BaseModel):
    """Explicit saved-document selection to index into one private conversation."""

    document_ids: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_document_ids(self) -> SavedDocumentsAttachRequest:
        """Reject ambiguous repeated selections before any MiniRAG mutation."""
        _validate_selected_document_ids(self.document_ids)
        return self


class SavedDocumentsAttachView(BaseModel):
    """Documents made available to the selected conversation."""

    conversation_id: str
    attached_count: int
    documents: list[SavedDocumentView]


class SavedDocumentsAnalysisRequest(BaseModel):
    """Analyze only the explicitly selected saved originals."""

    conversation_id: str = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1, max_length=100)
    question: str = ""
    generate_report: bool = False
    min_source_relevance: float = Field(default=0.55, ge=0.25, le=0.95)
    analysis_mode: Literal["question", "section", "full"] = "question"
    selected_sections: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_document_ids(self) -> SavedDocumentsAnalysisRequest:
        """Keep one deterministic analysis input per saved record."""
        _validate_selected_document_ids(self.document_ids)
        return self


def _validate_selected_document_ids(document_ids: list[str]) -> None:
    """Apply the shared non-empty and uniqueness rule to saved-document requests."""
    if any(not document_id.strip() for document_id in document_ids):
        raise ValueError("Saved document ids must not be blank.")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Saved document ids must be unique.")


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
