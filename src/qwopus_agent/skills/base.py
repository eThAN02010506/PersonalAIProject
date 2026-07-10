"""Base contracts for independent Agent skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except ModuleNotFoundError:  # pragma: no cover - exercised only before project deps are installed.
    BaseModel = object  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[misc]
        return default


if BaseModel is object:

    @dataclass(frozen=True)
    class SkillRequest:
        """Fallback request model so local tests can run before dependencies are installed."""

        query: str
        arguments: dict[str, Any] = field(default_factory=dict)
        context: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class SkillResponse:
        """Fallback response model so the skill contract remains importable."""

        success: bool
        content: str
        data: dict[str, Any] = field(default_factory=dict)

else:

    class SkillRequest(BaseModel):
        """Typed input passed from Executor to a Skill."""

        # Reason: Skills must receive structured inputs so the Executor never depends on ad hoc kwargs.
        model_config = ConfigDict(frozen=True)

        # Role: Natural-language task or search query for the skill.
        query: str

        # Role: Structured parameters such as file paths, sheet names, limits, or filters.
        arguments: dict[str, Any] = Field(default_factory=dict)

        # Role: Runtime metadata such as task id, user preferences, or trace ids.
        context: dict[str, Any] = Field(default_factory=dict)

    class SkillResponse(BaseModel):
        """Typed output returned by every Skill."""

        # Reason: All skills need the same success envelope for clean Executor orchestration.
        model_config = ConfigDict(frozen=True)

        # Role: Indicates whether the skill completed its task.
        success: bool

        # Role: Human-readable result summary that can be passed back to the Agent.
        content: str

        # Role: Structured artifacts for downstream report generation or memory insertion.
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
