"""Allowlisted source verification Skill."""

from __future__ import annotations

from qwopus_agent.code_workspace.commands import run_code_command
from qwopus_agent.code_workspace.security import CodeWorkspaceError, resolve_git_workspace
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class CodeTestSkill(BaseSkill):
    name = "code_test"
    description = "Run one server-defined verification command without shell interpolation."
    agent_tool_permission = "code_execute"

    async def run(self, request: SkillRequest) -> SkillResponse:
        if request.context.get("allow_execution") is not True:
            raise CodeWorkspaceError("Code execution requires explicit approval.")
        result = run_code_command(
            resolve_git_workspace(str(request.arguments.get("root", ""))),
            str(request.arguments.get("command_id", "")),
        )
        return SkillResponse(
            success=result.success,
            content=result.output,
            data=result.model_dump(exclude={"output"}),
        )


def create_skill() -> BaseSkill:
    return CodeTestSkill()
