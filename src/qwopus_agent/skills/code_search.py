"""Literal source search Skill with bounded output."""

from __future__ import annotations

import json

from qwopus_agent.code_workspace.security import search_code_workspace
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class CodeSearchSkill(BaseSkill):
    name = "code_search"
    description = "Search eligible source files for a literal query without shell access."
    agent_tool_permission = "code_read"

    async def run(self, request: SkillRequest) -> SkillResponse:
        matches = search_code_workspace(
            str(request.arguments.get("root", "")),
            request.query,
            limit=int(request.arguments.get("limit", 100)),
        )
        data = [match.model_dump() for match in matches]
        return SkillResponse(
            success=True,
            # 原因：smolagents Tool 只能看到 SkillResponse.content，不能读取内部 data。
            # 作用：把有界匹配路径和行号作为 Observation 返回，Agent 才能继续调用 code_read。
            content=json.dumps({"matches": data}, ensure_ascii=False),
            data={"matches": data},
        )


def create_skill() -> BaseSkill:
    return CodeSearchSkill()
