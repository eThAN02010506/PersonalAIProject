"""Base contracts for independent Agent skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillRequest(BaseModel):
    """Typed input passed from Executor to a Skill."""

    # Reason: Skills receive structured inputs so the Executor never depends on ad hoc
    # keyword arguments.
    model_config = ConfigDict(frozen=True)
    query: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class SkillResponse(BaseModel):
    """Typed output returned by every Skill."""

    # Reason: A required dependency is safer than a second, behaviorally different fallback model.
    # Role: All installed Skills now share one validated response contract.
    model_config = ConfigDict(frozen=True)
    success: bool
    content: str
    data: dict[str, Any] = Field(default_factory=dict)


class BaseSkill(ABC):
    """Abstract base class for every independently reusable Agent capability."""

    # Reason: The registry needs a stable unique key for dynamic lookup.
    name: str

    # Role: Planner-facing explanation of when this skill should be selected.
    description: str

    @abstractmethod
    async def run(self, request: SkillRequest) -> SkillResponse:
        """Execute the skill with a typed request."""
