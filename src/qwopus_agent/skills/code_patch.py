"""In-memory code patch preview Skill; it intentionally has no write operation."""

from __future__ import annotations

from qwopus_agent.code_workspace.models import CodeProposalDraft
from qwopus_agent.code_workspace.patching import apply_exact_replacements, build_git_diff
from qwopus_agent.code_workspace.security import read_complete_code_file, resolve_git_workspace
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


class CodePatchSkill(BaseSkill):
    name = "code_patch"
    description = "Validate exact source replacements and return a diff without writing files."
    # 原因：普通聊天 Agent 不应自行越过用户的 Diff 审批步骤。
    # 作用：Skill 可独立测试和由专用服务复用，但不会被自动包装为 Agent Tool。
    agent_tool_permission = None

    async def run(self, request: SkillRequest) -> SkillResponse:
        root = resolve_git_workspace(str(request.arguments.get("root", "")))
        draft = CodeProposalDraft.model_validate(request.arguments.get("proposal", {}))
        changes: list[tuple[str, str, str]] = []
        for file_draft in draft.changes:
            before = read_complete_code_file(root, file_draft.path)
            after = apply_exact_replacements(before, draft, file_draft.path)
            changes.append((file_draft.path, before, after))
        diff = build_git_diff(changes)
        return SkillResponse(success=True, content=diff, data={"changed_files": len(changes)})


def create_skill() -> BaseSkill:
    return CodePatchSkill()
