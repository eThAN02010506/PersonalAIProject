"""Generated report download route."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def build_report_router(report_directory: Path) -> APIRouter:
    """Build a report router confined to one configured storage directory."""
    router = APIRouter()

    @router.get("/api/reports/{filename}")
    def report(filename: str) -> FileResponse:
        path = report_directory / Path(filename).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Report not found.")
        return FileResponse(path, filename=path.name)

    return router
