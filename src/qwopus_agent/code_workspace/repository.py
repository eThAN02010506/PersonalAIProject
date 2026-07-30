"""Atomic local persistence for reviewable source-code proposals."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

from qwopus_agent.code_workspace.models import CodeChangeRecord
from qwopus_agent.code_workspace.security import CodeWorkspaceError

DEFAULT_CODE_CHANGE_DIRECTORY = Path("storage/code_changes")


class CodeChangeRepository:
    """Store private proposal snapshots separately from conversations and Skills."""

    def __init__(self, directory: Path = DEFAULT_CODE_CHANGE_DIRECTORY) -> None:
        self.directory = directory
        self._lock = RLock()

    def save(self, record: CodeChangeRecord) -> None:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{record.id}.json"
            temporary = self.directory / f".{record.id}.tmp"
            temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            # 原因：记录内含修改前源码，其他本机账号不应通过文件权限绕过 API。
            # 作用：仅让启动 Qwopus-Agent 的主机账号读写提案快照。
            os.chmod(temporary, 0o600)
            temporary.replace(path)

    def get(self, change_id: str, owner_user_id: str) -> CodeChangeRecord:
        if not change_id.isalnum() or len(change_id) > 64:
            raise CodeWorkspaceError("Invalid code change id.")
        path = self.directory / f"{change_id}.json"
        try:
            record = CodeChangeRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CodeWorkspaceError("Code change was not found.") from exc
        if record.owner_user_id != owner_user_id:
            raise CodeWorkspaceError("Code change was not found.")
        return record

    def list_for_user(self, owner_user_id: str, limit: int = 50) -> list[CodeChangeRecord]:
        if not self.directory.is_dir():
            return []
        records: list[CodeChangeRecord] = []
        for path in sorted(self.directory.glob("*.json"), reverse=True):
            try:
                record = CodeChangeRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.owner_user_id == owner_user_id:
                records.append(record)
                if len(records) >= limit:
                    break
        return records
