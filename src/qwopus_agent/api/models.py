"""Typed HTTP contracts for the Qwopus-Agent frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
