"""Read-only source-tree Skill for an explicitly selected Git workspace."""

from __future__ import annotations

from qwopus_agent.code_workspace.security import scan_code_workspace
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class CodeTreeSkill(BaseSkill):
    name = "code_tree"
    description = "List safe source files in a local Git workspace without reading contents."
    agent_tool_permission = "code_read"

    async def run(self, request: SkillRequest) -> SkillResponse:
        tree = scan_code_workspace(str(request.arguments.get("root", "")))
        return SkillResponse(success=True, content=tree.model_dump_json(), data=tree.model_dump())


def create_skill() -> BaseSkill:
    return CodeTreeSkill()
