"""Declarative workflow skills learned from successful Agent runs."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse

if TYPE_CHECKING:
    from qwopus_agent.skills.registry import SkillRegistry


class WorkflowStep(BaseModel):
    """One existing Skill call inside a learned workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_]+$")
    query_template: str = "{query}"
    arguments: dict[str, Any] = Field(default_factory=dict)


class WorkflowSpec(BaseModel):
    """Versioned and integrity-protected learned workflow definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1)
    # 原因：工具序列只说明“怎么做”，不足以识别用户用模糊说法表达的“要做什么”。
    # 作用：保存已验证任务的脱敏示例，Planner 可据此匹配已晋升工作流。
    intent_examples: tuple[str, ...] = ()
    steps: tuple[WorkflowStep, ...] = Field(min_length=1)
    source_signature: str = Field(min_length=1)
    checksum: str = ""

    def calculate_checksum(self) -> str:
        """Calculate a deterministic integrity digest excluding the digest itself."""
        payload = self.model_dump(mode="json", exclude={"checksum"})
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def sealed(self) -> WorkflowSpec:
        """Return a copy carrying its calculated checksum."""
        return self.model_copy(update={"checksum": self.calculate_checksum()})

    def checksum_is_valid(self) -> bool:
        """Verify that persisted workflow content was not modified."""
        return bool(self.checksum) and self.checksum == self.calculate_checksum()


class WorkflowSkill(BaseSkill):
    """Execute a validated WorkflowSpec through the normal Skill Registry."""

    def __init__(self, spec: WorkflowSpec, registry: SkillRegistry) -> None:
        if not spec.checksum_is_valid():
            raise ValueError(f"Workflow checksum is invalid: {spec.name}@{spec.version}")
        self.spec = spec
        self.registry = registry
        self.name = spec.name
        self.description = spec.description

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Run workflow steps in order and stop on the first failed Skill."""
        stack = [str(value) for value in request.context.get("workflow_stack", [])]
        if self.name in stack:
            return SkillResponse(
                success=False,
                content=f"Recursive workflow call blocked: {self.name}",
            )

        traces: list[dict[str, Any]] = []
        contents: list[str] = []
        for step in self.spec.steps:
            nested_request = SkillRequest(
                query=step.query_template.replace("{query}", request.query),
                arguments={**step.arguments, **request.arguments},
                context={
                    **request.context,
                    "workflow_stack": [*stack, self.name],
                    "workflow_name": self.name,
                    "workflow_version": self.spec.version,
                },
            )
            # 原因：成长 Skill 只能组合已注册能力，不能绕过 Registry 执行任意代码。
            # 作用：复用统一权限与错误边界，并让每个底层 Skill 仍可独立测试。
            response = await self.registry.execute(step.skill_name, nested_request)
            traces.append(
                {
                    "skill_name": step.skill_name,
                    "success": response.success,
                    "content": response.content,
                    "data": response.data,
                }
            )
            contents.append(response.content)
            if not response.success:
                return SkillResponse(
                    success=False,
                    content="\n".join(contents),
                    data={"workflow": self.name, "steps": traces},
                )

        return SkillResponse(
            success=True,
            content="\n".join(contents),
            data={"workflow": self.name, "version": self.spec.version, "steps": traces},
        )
