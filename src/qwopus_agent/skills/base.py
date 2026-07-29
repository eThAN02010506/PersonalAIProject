"""Base contracts for independent Agent skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

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

    # 原因：新增普通 Skill 应由 Registry 自动暴露给 Agent，不应再修改中央工具清单。
    # 作用：默认 query-only Tool 可零注册接入；敏感 Skill 通过覆盖 permission 保持显式授权。
    agent_tool_permission: ClassVar[str | None] = "always"
    agent_tool_inputs: ClassVar[Mapping[str, dict[str, Any]]] = {
        "query": {
            "type": "string",
            "description": "The current user objective for this skill.",
        }
    }
    agent_tool_name: ClassVar[str | None] = None

    # Reason: The registry needs a stable unique key for dynamic lookup.
    name: str

    # Role: Planner-facing explanation of when this skill should be selected.
    description: str

    @abstractmethod
    async def run(self, request: SkillRequest) -> SkillResponse:
        """Execute the skill with a typed request."""
