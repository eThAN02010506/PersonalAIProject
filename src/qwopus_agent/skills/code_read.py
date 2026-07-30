"""Bounded read-only source-file Skill."""

from __future__ import annotations

from qwopus_agent.code_workspace.security import read_code_file
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class CodeReadSkill(BaseSkill):
    name = "code_read"
    description = "Read a bounded line range from one selected UTF-8 source file."
    agent_tool_permission = "code_read"

    async def run(self, request: SkillRequest) -> SkillResponse:
        view = read_code_file(
            str(request.arguments.get("root", "")),
            str(request.arguments.get("path", "")),
            start_line=int(request.arguments.get("start_line", 1)),
            end_line=int(request.arguments.get("end_line", 600)),
        )
        return SkillResponse(
            success=True,
            content=view.content,
            data=view.model_dump(exclude={"content"}),
        )


def create_skill() -> BaseSkill:
    return CodeReadSkill()
