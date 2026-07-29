"""FastAPI entry point for the primary Qwopus-Agent web application."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path, PurePosixPath

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import Scope

from qwopus_agent.api.routes import (
    build_analysis_router,
    build_conversation_router,
    build_debug_router,
    build_document_router,
    build_local_folder_router,
    build_model_router,
    build_report_router,
)
from qwopus_agent.documents import DocumentStore
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.utils.debug_store import DEFAULT_DEBUG_DIRECTORY
from qwopus_agent.utils.logging_config import (
    DEFAULT_RUNTIME_LOG_PATH,
    configure_runtime_logging,
)

from .model_runtime import RuntimeModelController
from .repository import ConversationRepository
from .runs import ChatRunRegistry

REPORT_DIRECTORY = Path("storage/reports")
FRONTEND_DIRECTORY = Path("frontend/dist")


class SPAStaticFiles(StaticFiles):
    """Serve built frontend files and fall back to index.html for client routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            # 原因：SPA fallback 只能处理客户端路由，不能把明确的越界路径伪装成正常页面。
            # 作用：保留 StaticFiles 的 404 安全语义，同时不影响 /debug 等合法 React 路由。
            raise HTTPException(status_code=404)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
        else:
            if response.status_code != 404:
                return response

        # 原因：手写 Path 拼接会绕过 Starlette 的 common-path 与符号链接检查。
        # 作用：未知前端路由仍返回 SPA 入口，文件访问继续由 StaticFiles 限定在 dist 内。
        return await super().get_response("index.html", scope)


def create_app(
    repository: ConversationRepository | None = None,
    knowledge_manager: ConversationKnowledgeManager | None = None,
    model_runtime: RuntimeModelController | None = None,
    debug_directory: Path | None = None,
    runtime_log_path: Path = DEFAULT_RUNTIME_LOG_PATH,
    document_store: DocumentStore | None = None,
    debug_allow_lan: bool | None = None,
) -> FastAPI:
    """Build an independently testable API application."""
    repo = repository or ConversationRepository()
    debug_path = debug_directory if debug_directory is not None else DEFAULT_DEBUG_DIRECTORY
    knowledge = knowledge_manager or ConversationKnowledgeManager()
    runs = ChatRunRegistry(
        repo,
        debug_directory=debug_path,
        knowledge_root=knowledge.root,
    )
    runtime = model_runtime or RuntimeModelController()
    documents = document_store or DocumentStore()
    started_at = time.monotonic()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        repo.initialize()
        reaper = asyncio.create_task(_reap_chat_runs(runs))
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
            # 原因：关闭浏览器或停止 API 时可能仍有等待模型的后台 Agent worker。
            # 作用：应用退出前统一终止 registry 拥有的任务并释放 multiprocessing 队列。
            runs.close()
            # 原因：本地 MLX 由 FastAPI 启动后不应在应用退出时成为孤儿进程。
            # 作用：只终止本控制器拥有的子进程，不影响用户手工启动的模型服务。
            runtime.close()

    api = FastAPI(title="Qwopus-Agent API", version="0.1.0", lifespan=lifespan)
    api.state.repository = repo
    api.state.runs = runs
    api.state.model_runtime = runtime
    api.state.knowledge_manager = knowledge
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
    api.include_router(build_conversation_router(repo, runs, runtime, knowledge))
    api.include_router(build_analysis_router(runtime, repo, knowledge, debug_path))
    api.include_router(
        build_document_router(
            documents,
            repo,
            knowledge,
            runtime,
            debug_path,
        )
    )
    api.include_router(build_local_folder_router(runtime, repo, debug_path))
    api.include_router(build_report_router(REPORT_DIRECTORY))
    api.include_router(
        build_debug_router(
            runtime,
            runs,
            debug_path,
            runtime_log_path,
            started_at=started_at,
            allow_lan=debug_allow_lan,
        )
    )

    # 原因：生产模式需要一个进程同时提供 API 和已构建前端，调试时 Vite 仍可独立热更新。
    # 作用：根挂载放在 API Router 之后，既保留 API 优先级，也让 StaticFiles 统一约束路径。
    if FRONTEND_DIRECTORY.is_dir():
        api.mount(
            "/",
            SPAStaticFiles(
                directory=FRONTEND_DIRECTORY,
                check_dir=True,
                follow_symlink=False,
            ),
            name="frontend",
        )

    return api


async def _reap_chat_runs(runs: ChatRunRegistry) -> None:
    """Collect abandoned terminal workers without blocking the event loop."""
    while True:
        await asyncio.sleep(5)
        # 原因：用户可能关闭所有 Debug 页面并放弃原 run_id，不能依赖后续 HTTP 轮询收割。
        # 作用：固定低频维护任务在线程池执行同步 Queue 检查，其他 API 请求保持可响应。
        await asyncio.to_thread(runs.reap)


configure_runtime_logging()
app = create_app()
