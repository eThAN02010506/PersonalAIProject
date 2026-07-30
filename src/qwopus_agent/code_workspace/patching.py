"""Pure proposal validation and Git-compatible diff generation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from qwopus_agent.code_workspace.models import CodeProposalDraft
from qwopus_agent.code_workspace.security import CodeWorkspaceError

MAX_REPLACEMENT_CHARS = 64 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_proposal_response(content: str) -> CodeProposalDraft:
    """Extract and validate the first complete JSON object from a model response."""
    return parse_json_model_response(
        content,
        CodeProposalDraft,
        error_message="The model did not return a valid code-change proposal.",
    )


def parse_json_model_response(
    content: str,
    model_type: type[ModelT],
    *,
    error_message: str,
) -> ModelT:
    """Extract the first JSON object satisfying one explicit Pydantic contract."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
            return model_type.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            continue
    raise CodeWorkspaceError(error_message)


def apply_exact_replacements(content: str, draft: CodeProposalDraft, path: str) -> str:
    """Apply unique snippets in memory; no proposal function writes to disk."""
    file_drafts = [change for change in draft.changes if change.path == path]
    if len(file_drafts) != 1:
        raise CodeWorkspaceError(f"Proposal must contain exactly one change for {path}.")
    updated = content
    for replacement in file_drafts[0].replacements:
        if not replacement.old_text:
            raise CodeWorkspaceError("Replacement old_text cannot be empty.")
        if (
            len(replacement.old_text) > MAX_REPLACEMENT_CHARS
            or len(replacement.new_text) > MAX_REPLACEMENT_CHARS
        ):
            raise CodeWorkspaceError("One replacement exceeds the 64 KiB safety limit.")
        if updated.count(replacement.old_text) != 1:
            # 原因：模糊或重复片段会让弱模型把修改落到错误函数。
            # 作用：只接受唯一精确匹配，无法确定位置时整个提案失败且不会写盘。
            raise CodeWorkspaceError(
                f"Replacement target in {path} must occur exactly once."
            )
        updated = updated.replace(replacement.old_text, replacement.new_text, 1)
    if updated == content:
        raise CodeWorkspaceError(f"Proposal does not change {path}.")
    return updated


def build_git_diff(changes: list[tuple[str, str, str]]) -> str:
    """Use Git itself to create an accurate text diff, including no-newline markers."""
    with tempfile.TemporaryDirectory(prefix="qwopus-code-diff-") as temporary_directory:
        root = Path(temporary_directory)
        before_root = root / "before"
        after_root = root / "after"
        for relative_path, before, after in changes:
            before_path = before_root / relative_path
            after_path = after_root / relative_path
            before_path.parent.mkdir(parents=True, exist_ok=True)
            after_path.parent.mkdir(parents=True, exist_ok=True)
            before_path.write_text(before, encoding="utf-8")
            after_path.write_text(after, encoding="utf-8")
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--no-ext-diff",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "--",
                "before",
                "after",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    if result.returncode not in {0, 1}:
        raise CodeWorkspaceError("Git could not generate the proposal diff.")
    return _strip_snapshot_prefixes(result.stdout)


def check_git_diff(root: Path, unified_diff: str) -> None:
    """Ask Git to validate patch applicability without changing the worktree."""
    result = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", "--whitespace=nowarn", "-"],
        input=unified_diff,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Patch is no longer applicable."
        raise CodeWorkspaceError(detail[:1000])


def _strip_snapshot_prefixes(diff: str) -> str:
    lines: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git a/before/"):
            line = line.replace("a/before/", "a/", 1).replace("b/after/", "b/", 1)
        elif line.startswith("--- a/before/"):
            line = line.replace("--- a/before/", "--- a/", 1)
        elif line.startswith("+++ b/after/"):
            line = line.replace("+++ b/after/", "+++ b/", 1)
        lines.append(line)
    return "".join(lines)
