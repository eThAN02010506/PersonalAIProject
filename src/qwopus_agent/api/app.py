"""FastAPI entry point for the primary Qwopus-Agent web application."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import Scope

from qwopus_agent.api.auth import AccountAuthMiddleware, AuthService
from qwopus_agent.api.routes import (
    build_analysis_router,
    build_auth_router,
    build_code_workspace_router,
    build_conversation_router,
    build_debug_router,
    build_document_router,
    build_local_folder_router,
    build_model_router,
    build_report_router,
    build_skill_authoring_router,
    build_skill_router,
    build_web_search_settings_router,
)
from qwopus_agent.code_workspace.repository import CodeChangeRepository
from qwopus_agent.documents import DocumentStore
from qwopus_agent.integrations.smolagents_code_workspace import (
    run_smolagents_code_workspace_chat,
)
from qwopus_agent.integrations.tavily_credentials import TavilyCredentialStore
from qwopus_agent.llm import LLMRegistry, create_default_llm_registry
from qwopus_agent.memory import ConversationKnowledgeManager
from qwopus_agent.services.code_workspace_service import CodeWorkspaceService
from qwopus_agent.services.skill_authoring_service import SkillAuthoringService
from qwopus_agent.services.skill_growth_service import SkillGrowthService
from qwopus_agent.skills import SkillCatalog, SkillRegistry
from qwopus_agent.utils.debug_store import DEFAULT_DEBUG_DIRECTORY
from qwopus_agent.utils.logging_config import (
    DEFAULT_RUNTIME_LOG_PATH,
    configure_runtime_logging,
)

from .lan_auth import LanAuthConfig, LanAuthMiddleware
from .model_runtime import RuntimeModelController
from .repository import ConversationRepository
from .runs import ChatRunRegistry

if TYPE_CHECKING:
    from qwopus_agent.memory.minirag import MiniRAG

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
    report_directory: Path | None = None,
    lan_auth: LanAuthConfig | None = None,
    tavily_credentials: TavilyCredentialStore | None = None,
    code_workspace_service: CodeWorkspaceService | None = None,
) -> FastAPI:
    """Build an independently testable API application."""
    repo = repository or ConversationRepository()
    debug_path = debug_directory if debug_directory is not None else DEFAULT_DEBUG_DIRECTORY
    runtime = model_runtime or RuntimeModelController()
    llm_registry = create_default_llm_registry()
    knowledge = knowledge_manager or _build_default_knowledge_manager(
        runtime,
        llm_registry,
    )
    skill_root = repo.database_path.parent / "skills"
    skill_catalog = SkillCatalog(storage_path=skill_root / "catalog.json")
    skill_registry = SkillRegistry.discover(
        catalog=skill_catalog,
        workflow_root=skill_root / "workflows",
    )
    skill_growth = SkillGrowthService(
        registry=skill_registry,
        catalog=skill_catalog,
        workflow_root=skill_root / "workflows",
        history_path=skill_root / "growth_history.json",
    )
    skill_authoring = SkillAuthoringService(
        growth=skill_growth,
        # 原因：用户可在运行时切换模型地址和模型 ID，作者服务不能缓存旧适配器。
        # 作用：每次生成都从 RuntimeModelController 获取当前在线模型并走 BaseLLM。
        llm_factory=lambda: llm_registry.create_from_settings(
            runtime.require_online_settings()
        ),
    )
    documents = document_store or DocumentStore()
    runs = ChatRunRegistry(
        repo,
        debug_directory=debug_path,
        knowledge_root=knowledge.root,
        knowledge_manager=knowledge,
        skill_catalog=skill_catalog,
        skill_growth=skill_growth,
        document_store=documents,
    )
    reports = report_directory or REPORT_DIRECTORY
    auth = AuthService(repo)
    web_search_credentials = tavily_credentials or TavilyCredentialStore()
    code_workspace = code_workspace_service or CodeWorkspaceService(
        CodeChangeRepository(repo.database_path.parent / "code_changes"),
        # 原因：源码提案必须跟随管理员当前选择的模型，不能缓存启动时模型名称。
        # 作用：Gemma、Qwen、Qwopus 或其他 OpenAI Compatible 模型可直接生成同一合同。
        llm_factory=lambda: llm_registry.create_from_settings(
            runtime.require_online_settings()
        ),
        # 原因：Code Workspace 原先绕过 smolagents，抽象需求只能经过固定两段 Prompt。
        # 作用：每轮使用当前在线模型执行受控 code_search/code_read 循环，写入仍留在审批服务。
        code_chat_runner=lambda root, transcript, eligible_paths, selected_files: (
            run_smolagents_code_workspace_chat(
                root,
                transcript,
                eligible_paths,
                selected_files,
                settings=runtime.require_online_settings(),
            )
        ),
    )
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
    api.state.skill_catalog = skill_catalog
    api.state.skill_registry = skill_registry
    api.state.skill_growth = skill_growth
    api.state.skill_authoring = skill_authoring
    api.state.auth = auth
    api.state.tavily_credentials = web_search_credentials
    api.state.code_workspace = code_workspace
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 原因：LAN Basic Auth 只保护网络入口，无法区分同一局域网中的实际使用者。
    # 作用：账号中间件为每个 API 请求解析服务端会话，并默认拒绝未登录的私有资源。
    api.add_middleware(AccountAuthMiddleware, auth=auth)
    # 原因：认证若只挂在 /api，React、OpenAPI 和 Debug 静态入口仍能从 LAN 直接读取。
    # 作用：统一保护完整 ASGI 应用；环回客户端由中间件自动免认证。
    api.add_middleware(
        LanAuthMiddleware,
        config=lan_auth or LanAuthConfig.from_environment(),
    )
    # 原因：入口工厂只负责依赖装配，路由业务边界不应共同累积在一个函数中。
    # 作用：每组 Router 可独立测试，create_app 的复杂度不随功能数量线性增长。
    api.include_router(
        build_auth_router(
            auth,
            repo,
            documents,
            knowledge,
            reports,
            runs,
        )
    )
    api.include_router(build_model_router(runtime))
    api.include_router(
        build_conversation_router(
            repo,
            runs,
            runtime,
            knowledge,
            web_search_credentials,
        )
    )
    api.include_router(
        build_web_search_settings_router(web_search_credentials)
    )
    api.include_router(
        build_analysis_router(
            runtime,
            repo,
            knowledge,
            debug_path,
            documents,
        )
    )
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
    api.include_router(build_code_workspace_router(code_workspace, debug_path))
    api.include_router(build_report_router(reports, repo))
    api.include_router(build_skill_router(skill_growth))
    api.include_router(
        build_skill_authoring_router(skill_authoring, repo)
    )
    api.include_router(
        build_debug_router(
            runtime,
            runs,
            debug_path,
            runtime_log_path,
            started_at=started_at,
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


def _build_default_knowledge_manager(
    runtime: RuntimeModelController,
    llm_registry: LLMRegistry,
) -> ConversationKnowledgeManager:
    """Compose evidence-bound LLM graph extraction at the application boundary."""

    def create_memory(storage_path: Path) -> MiniRAG:
        from qwopus_agent.memory.graph_extraction import (
            CompositeGraphExtractor,
            LLMGraphExtractor,
            RuleBasedGraphExtractor,
        )
        from qwopus_agent.memory.minirag import MiniRAG

        # 原因：仅使用规则语法时，普通 PDF/DOCX 文本不会产生实体关系，图谱只在
        # 人工写入 [[A]] -[relation]-> [[B]] 时有效。
        # 作用：规则抽取提供确定性保底，LLM 抽取补充自然语言关系；每次调用都读取
        # RuntimeModelController 当前设置，因此切换 Gemma/Qwen/Qwopus 后无需重建适配器。
        graph_extractor = CompositeGraphExtractor(
            extractors=(
                RuleBasedGraphExtractor(),
                LLMGraphExtractor(
                    llm_factory=lambda: llm_registry.create_from_settings(
                        runtime.require_online_settings()
                    )
                ),
            )
        )
        return MiniRAG(
            storage_path=storage_path,
            graph_extractor=graph_extractor,
        )

    return ConversationKnowledgeManager(factory=create_memory)


async def _reap_chat_runs(runs: ChatRunRegistry) -> None:
    """Collect abandoned terminal workers without blocking the event loop."""
    while True:
        await asyncio.sleep(5)
        # 原因：用户可能关闭所有 Debug 页面并放弃原 run_id，不能依赖后续 HTTP 轮询收割。
        # 作用：固定低频维护任务在线程池执行同步 Queue 检查，其他 API 请求保持可响应。
        await asyncio.to_thread(runs.reap)


configure_runtime_logging()
app = create_app()
