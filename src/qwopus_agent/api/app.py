"""FastAPI entry point for the primary Qwopus-Agent web application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.orchestration_models import OrchestrationFile, OrchestrationRequest
from qwopus_agent.utils.debug_store import DEFAULT_DEBUG_DIRECTORY, append_debug_record

if TYPE_CHECKING:
    from qwopus_agent.memory import MiniRAG

from .model_runtime import ModelRuntimeError, RuntimeModelController, RuntimeModelStatus
from .models import (
    AnalysisView,
    ChatStartRequest,
    ConversationCreate,
    ConversationUpdate,
    ConversationView,
    MessageView,
    ModelSettingsUpdate,
    ModelSettingsView,
    RunStarted,
    RunView,
)
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
    async def lifespan(_app: FastAPI):
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

    @api.get("/api/health", response_model=ModelSettingsView)
    async def health() -> ModelSettingsView:
        return _model_settings_view(await asyncio.to_thread(runtime.status))

    @api.get("/api/model-settings", response_model=ModelSettingsView)
    async def model_settings() -> ModelSettingsView:
        return _model_settings_view(await asyncio.to_thread(runtime.status))

    @api.put("/api/model-settings", response_model=ModelSettingsView)
    async def update_model_settings(payload: ModelSettingsUpdate) -> ModelSettingsView:
        try:
            if payload.mode == "remote":
                if not payload.base_url:
                    raise ModelRuntimeError("Model address is required for remote mode.")
                status = await asyncio.to_thread(runtime.configure_remote, payload.base_url)
            else:
                if not payload.model_path:
                    raise ModelRuntimeError("Model path is required for local mode.")
                status = await asyncio.to_thread(runtime.configure_local, payload.model_path)
        except ModelRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _model_settings_view(status)

    @api.get("/api/conversations", response_model=list[ConversationView])
    def conversations() -> list[ConversationView]:
        return [ConversationView.model_validate(item) for item in repo.list_conversations()]

    @api.post("/api/conversations", response_model=ConversationView, status_code=201)
    def create_conversation(payload: ConversationCreate) -> ConversationView:
        return ConversationView.model_validate(repo.create_conversation(payload.title))

    @api.patch("/api/conversations/{conversation_id}", response_model=ConversationView)
    def rename_conversation(
        conversation_id: str,
        payload: ConversationUpdate,
    ) -> ConversationView:
        record = repo.rename_conversation(conversation_id, payload.title)
        if record is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return ConversationView.model_validate(record)

    @api.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str) -> None:
        if not repo.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")

    @api.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageView])
    def messages(conversation_id: str) -> list[MessageView]:
        if repo.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return [MessageView.model_validate(item) for item in repo.list_messages(conversation_id)]

    @api.post("/api/conversations/{conversation_id}/runs", response_model=RunStarted)
    def start_run(conversation_id: str, payload: ChatStartRequest) -> RunStarted:
        conversation = repo.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        if conversation.title in {"New chat", "新对话"}:
            repo.rename_conversation(conversation_id, _conversation_title(payload.content))
        run_id = runs.start(
            conversation_id,
            payload.content,
            runtime.current_settings(),
            enable_web_search=payload.enable_web_search,
            enable_local_knowledge=payload.enable_local_knowledge,
        )
        return RunStarted(run_id=run_id)

    @api.get("/api/runs/{run_id}", response_model=RunView)
    def poll_run(run_id: str) -> RunView:
        result = runs.poll(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return result

    @api.delete("/api/runs/{run_id}", response_model=RunView)
    def cancel_run(run_id: str) -> RunView:
        result = runs.cancel(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return result

    @api.post("/api/analysis", response_model=AnalysisView)
    async def analyze(
        files: Annotated[list[UploadFile], File()],
        question: Annotated[str, Form()] = "",
        generate_report: Annotated[bool, Form()] = False,
    ) -> AnalysisView:
        # 原因：生成式中包含 await 时会产生 async_generator，不能直接交给 tuple()。
        # 作用：逐个读取上传字节，确保 MinerU/MiniRAG 编排收到完整且有序的文件集合。
        uploads_list: list[OrchestrationFile] = []
        for file in files:
            uploads_list.append(
                OrchestrationFile(name=file.filename or "upload", content=await file.read())
            )
        uploads = tuple(uploads_list)
        if not uploads:
            raise HTTPException(status_code=400, detail="At least one file is required.")
        request = OrchestrationRequest(
            objective=question,
            uploaded_files=uploads,
            generate_report=generate_report,
            report_title="Qwopus Analysis Report",
            report_basename="qwopus_web_analysis",
        )
        orchestrator = AgentOrchestrator(
            runtime.current_settings(),
            minirag=get_minirag(),
        )
        result = await asyncio.to_thread(orchestrator.run_sync, request)
        # 原因：文档分析是同步 API 路径，不经过 ChatRunRegistry 的完成回调。
        # 作用：把 MinerU/MiniRAG/Agent 的内部步骤写给只读 Console，同时不进入 AnalysisView。
        append_debug_record(
            source="document",
            status="completed" if result.success else "failed",
            result=result.final_answer,
            trace=result.trace,
            debug_runs=result.debug_runs,
            directory=debug_path,
        )
        if not result.success:
            raise HTTPException(status_code=500, detail=result.final_answer)
        reports = []
        if result.report is not None:
            reports = [
                {
                    "kind": artifact.kind,
                    "name": artifact.path.name,
                    "url": f"/api/reports/{artifact.path.name}",
                }
                for artifact in result.report.artifacts
            ]
        return AnalysisView(
            answer=result.final_answer,
            route=result.route,
            citations=[item.model_dump(mode="json") for item in result.citations],
            trace=[item.model_dump(mode="json") for item in result.trace],
            reports=reports,
        )

    @api.get("/api/reports/{filename}")
    def report(filename: str) -> FileResponse:
        path = REPORT_DIRECTORY / Path(filename).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Report not found.")
        return FileResponse(path, filename=path.name)

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


def _conversation_title(content: str) -> str:
    title = " ".join(content.split())
    return title if len(title) <= 48 else f"{title[:47]}…"


def _model_settings_view(status: RuntimeModelStatus) -> ModelSettingsView:
    return ModelSettingsView(
        mode=status.mode,
        model_online=status.online,
        message=status.message,
        model=status.settings.model_id,
        base_url=status.settings.base_url,
        local_model_path=status.local_model_path,
    )


app = create_app()
