"""Generated report download route."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from qwopus_agent.api.auth import current_user
from qwopus_agent.api.repository import ConversationRepository


def build_report_router(
    report_directory: Path,
    repository: ConversationRepository,
) -> APIRouter:
    """Build a report router confined to one configured storage directory."""
    router = APIRouter()

    @router.get("/api/reports/{filename}")
    def report(filename: str, request: Request) -> FileResponse:
        user = current_user(request)
        path = report_directory / Path(filename).name
        if (
            not path.is_file()
            or not repository.can_access_report(path.name, user.id)
        ):
            # 原因：仅检查磁盘文件名会让其他账号通过猜测名称下载报告。
            # 作用：物理存在和 ACL 访问失败统一为 404，不泄漏报告是否存在。
            raise HTTPException(status_code=404, detail="Report not found.")
        return FileResponse(path, filename=path.name)

    return router
