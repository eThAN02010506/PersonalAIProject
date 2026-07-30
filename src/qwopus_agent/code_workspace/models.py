"""Typed contracts for code inspection, proposals, approval, and rollback."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CodeChangeStatus = Literal["proposed", "applied", "rejected", "rolled_back"]
CodeChatMode = Literal["answer", "clarify", "ready"]


class CodeTreeNode(BaseModel):
    """One safe, selectable node in a Git source tree."""

    model_config = ConfigDict(frozen=True)
    name: str
    relative_path: str
    kind: Literal["directory", "file"]
    children: list[CodeTreeNode] = Field(default_factory=list)


class CodeWorkspaceTree(BaseModel):
    """Filtered source tree returned to the local administrator."""

    model_config = ConfigDict(frozen=True)
    root: str
    file_count: int
    tree: CodeTreeNode


class CodeFileView(BaseModel):
    """Bounded source file content with a conflict-detection hash."""

    model_config = ConfigDict(frozen=True)
    root: str
    path: str
    sha256: str
    content: str
    total_lines: int
    start_line: int
    end_line: int


class CodeSearchMatch(BaseModel):
    """One literal source-code search hit."""

    model_config = ConfigDict(frozen=True)
    path: str
    line: int
    column: int
    preview: str


class CodeChatMessage(BaseModel):
    """One bounded message supplied as Code Workspace conversation context."""

    model_config = ConfigDict(frozen=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class CodeContextSelection(BaseModel):
    """Internal model decision describing which repository evidence to inspect."""

    model_config = ConfigDict(frozen=True)
    candidate_paths: list[str] = Field(default_factory=list, max_length=8)
    search_queries: list[str] = Field(default_factory=list, max_length=4)


class CodeChatReply(BaseModel):
    """Grounded conversational response that may prepare a reviewed proposal."""

    model_config = ConfigDict(frozen=True)
    mode: CodeChatMode
    message: str = Field(min_length=1, max_length=12000)
    objective: str | None = Field(default=None, max_length=4000)
    selected_files: list[str] = Field(default_factory=list, max_length=8)
    inspected_files: list[str] = Field(default_factory=list, max_length=8)


class CodeWorkspaceAgentRun(BaseModel):
    """Internal result from one read-only smolagents repository exploration."""

    model_config = ConfigDict(frozen=True)
    content: str
    inspected_files: list[str] = Field(default_factory=list, max_length=8)
    tool_calls: list[str] = Field(default_factory=list, max_length=20)
    state: str | None = None


class CodeReplacement(BaseModel):
    """One exact replacement proposed by the model."""

    model_config = ConfigDict(frozen=True)
    old_text: str
    new_text: str


class CodeFileDraft(BaseModel):
    """Model-produced replacements for one explicitly selected file."""

    model_config = ConfigDict(frozen=True)
    path: str
    replacements: list[CodeReplacement] = Field(min_length=1, max_length=20)


class CodeProposalDraft(BaseModel):
    """Strict model output accepted by the proposal service."""

    model_config = ConfigDict(frozen=True)
    summary: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=2000)
    verification_plan: list[str] = Field(min_length=1, max_length=10)
    changes: list[CodeFileDraft] = Field(min_length=1, max_length=12)


class CodeFileChange(BaseModel):
    """Persistent before/after state used for approval and exact rollback."""

    model_config = ConfigDict(frozen=True)
    path: str
    before_sha256: str
    after_sha256: str
    before_content: str
    after_content: str


class CodeChangeRecord(BaseModel):
    """Private durable record; full contents never cross the HTTP boundary."""

    model_config = ConfigDict(frozen=True)
    id: str
    owner_user_id: str
    root: str
    objective: str
    summary: str
    reason: str
    verification_plan: list[str]
    unified_diff: str
    status: CodeChangeStatus
    changes: list[CodeFileChange]
    created_at: datetime
    applied_at: datetime | None = None
    rolled_back_at: datetime | None = None


class CodeChangeView(BaseModel):
    """Safe proposal metadata and diff shown in the approval UI."""

    model_config = ConfigDict(frozen=True)
    id: str
    root: str
    objective: str
    summary: str
    reason: str
    verification_plan: list[str]
    unified_diff: str
    status: CodeChangeStatus
    changed_files: list[str]
    created_at: datetime
    applied_at: datetime | None = None
    rolled_back_at: datetime | None = None


class CodeTestResult(BaseModel):
    """Bounded result from one server-defined verification command."""

    model_config = ConfigDict(frozen=True)
    command_id: str
    command: list[str]
    return_code: int
    success: bool
    timed_out: bool = False
    output: str


class CodeCommandView(BaseModel):
    """A fixed verification command available to the approval UI."""

    model_config = ConfigDict(frozen=True)
    id: str
    label: str
    description: str


def change_view(record: CodeChangeRecord) -> CodeChangeView:
    """Remove private file snapshots before returning a proposal."""
    return CodeChangeView(
        id=record.id,
        root=record.root,
        objective=record.objective,
        summary=record.summary,
        reason=record.reason,
        verification_plan=record.verification_plan,
        unified_diff=record.unified_diff,
        status=record.status,
        changed_files=[change.path for change in record.changes],
        created_at=record.created_at,
        applied_at=record.applied_at,
        rolled_back_at=record.rolled_back_at,
    )
