"""FastAPI entry point for the primary Qwopus-Agent web application."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qwopus_agent.api.routes import (
    build_analysis_router,
    build_conversation_router,
    build_model_router,
    build_report_router,
)
from qwopus_agent.utils.debug_store import DEFAULT_DEBUG_DIRECTORY

if TYPE_CHECKING:
    from qwopus_agent.memory import MiniRAG

from .model_runtime import RuntimeModelController
from .repository import ConversationRepository
from .runs import ChatRunRegistry

REPORT_DIRECTORY = Path("storage/reports")
FRONTEND_DIRECTORY = Path("frontend/dist")


def create_app(
    repository: ConversationRepository | None = None,
    minirag: MiniRAG | None = None,
    model_runtime: RuntimeModelController | None = None,
    debug_directory: Path | None = None,
) -> FastAPI:
    """Build an independently testable API application."""
    repo = repository or ConversationRepository()
    debug_path = debug_directory if debug_directory is not None else DEFAULT_DEBUG_DIRECTORY
    runs = ChatRunRegistry(repo, debug_directory=debug_path)
    memory = minirag
    memory_lock = Lock()
    runtime = model_runtime or RuntimeModelController()

    def get_minirag() -> MiniRAG:
        nonlocal memory
        if memory is not None:
            return memory
        with memory_lock:
            if memory is None:
                # 原因：普通聊天和 API 文档不需要加载 embedding/Torch，且并发请求不能重复初始化。
                # 作用：首次文档分析时恢复同一个持久化 MiniRAG，后续请求复用其索引和图谱。
                from qwopus_agent.memory import MiniRAG

                memory = MiniRAG()
        return memory

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        repo.initialize()
        try:
            yield
        finally:
            # 原因：本地 MLX 由 FastAPI 启动后不应在应用退出时成为孤儿进程。
            # 作用：只终止本控制器拥有的子进程，不影响用户手工启动的模型服务。
            runtime.close()

    api = FastAPI(title="Qwopus-Agent API", version="0.1.0", lifespan=lifespan)
    api.state.repository = repo
    api.state.runs = runs
    api.state.model_runtime = runtime
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 原因：入口工厂只负责依赖装配，路由业务边界不应共同累积在一个函数中。
    # 作用：每组 Router 可独立测试，create_app 的复杂度不随功能数量线性增长。
    api.include_router(build_model_router(runtime))
    api.include_router(build_conversation_router(repo, runs, runtime))
    api.include_router(build_analysis_router(runtime, get_minirag, debug_path))
    api.include_router(build_report_router(REPORT_DIRECTORY))

    # 原因：生产模式需要一个进程同时提供 API 和已构建前端，调试时 Vite 仍可独立热更新。
    # 作用：存在 dist 时托管 SPA；没有构建前端时 API 测试和 OpenAPI 仍可单独运行。
    if FRONTEND_DIRECTORY.is_dir():
        assets = FRONTEND_DIRECTORY / "assets"
        if assets.is_dir():
            api.mount("/assets", StaticFiles(directory=assets), name="assets")

        @api.get("/{full_path:path}", include_in_schema=False)
        def frontend(full_path: str) -> FileResponse:
            candidate = FRONTEND_DIRECTORY / full_path
            return FileResponse(
                candidate if candidate.is_file() else FRONTEND_DIRECTORY / "index.html"
            )

    return api


app = create_app()
