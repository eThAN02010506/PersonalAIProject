"""Conversation persistence and background Agent run routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from qwopus_agent.api.auth import current_user
from qwopus_agent.api.model_runtime import ModelRuntimeError, RuntimeModelController
from qwopus_agent.api.models import (
    ChatStartRequest,
    ConversationCreate,
    ConversationMemberView,
    ConversationShareCreate,
    ConversationUpdate,
    ConversationView,
    MessageView,
    RunStarted,
    RunView,
)
from qwopus_agent.api.repository import ConversationRecord, ConversationRepository
from qwopus_agent.api.runs import ChatRunRegistry
from qwopus_agent.memory import ConversationKnowledgeManager


def build_conversation_router(
    repository: ConversationRepository,
    runs: ChatRunRegistry,
    runtime: RuntimeModelController,
    knowledge: ConversationKnowledgeManager,
) -> APIRouter:
    """Build routes around one injected conversation repository and run registry."""
    router = APIRouter()
    endpoints = _ConversationEndpoints(repository, runs, runtime, knowledge)
    router.add_api_route(
        "/api/conversations",
        endpoints.conversations,
        methods=["GET"],
        response_model=list[ConversationView],
    )
    router.add_api_route(
        "/api/conversations",
        endpoints.create_conversation,
        methods=["POST"],
        response_model=ConversationView,
        status_code=201,
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}",
        endpoints.rename_conversation,
        methods=["PATCH"],
        response_model=ConversationView,
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}",
        endpoints.delete_conversation,
        methods=["DELETE"],
        status_code=204,
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/messages",
        endpoints.messages,
        methods=["GET"],
        response_model=list[MessageView],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/members",
        endpoints.members,
        methods=["GET"],
        response_model=list[ConversationMemberView],
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/members",
        endpoints.share_conversation,
        methods=["POST"],
        response_model=ConversationMemberView,
        status_code=201,
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/members/{user_id}",
        endpoints.unshare_conversation,
        methods=["DELETE"],
        status_code=204,
    )
    router.add_api_route(
        "/api/conversations/{conversation_id}/runs",
        endpoints.start_run,
        methods=["POST"],
        response_model=RunStarted,
    )
    router.add_api_route(
        "/api/runs/{run_id}",
        endpoints.poll_run,
        methods=["GET"],
        response_model=RunView,
    )
    router.add_api_route(
        "/api/runs/{run_id}",
        endpoints.cancel_run,
        methods=["DELETE"],
        response_model=RunView,
    )
    return router


class _ConversationEndpoints:
    """Bound HTTP handlers sharing explicit repository and runtime dependencies."""

    def __init__(
        self,
        repository: ConversationRepository,
        runs: ChatRunRegistry,
        runtime: RuntimeModelController,
        knowledge: ConversationKnowledgeManager,
    ) -> None:
        self.repository = repository
        self.runs = runs
        self.runtime = runtime
        self.knowledge = knowledge

    def conversations(self, request: Request) -> list[ConversationView]:
        user = current_user(request)
        return [
            ConversationView.model_validate(item)
            for item in self.repository.list_conversations_for_user(user.id)
        ]

    def create_conversation(
        self,
        payload: ConversationCreate,
        request: Request,
    ) -> ConversationView:
        user = current_user(request)
        created = self.repository.create_conversation(
            payload.title,
            owner_user_id=user.id,
        )
        scoped = self.repository.get_conversation_for_user(created.id, user.id)
        if scoped is None:
            raise RuntimeError("Created conversation is not accessible to its owner.")
        return ConversationView.model_validate(scoped)

    def rename_conversation(
        self,
        conversation_id: str,
        payload: ConversationUpdate,
        request: Request,
    ) -> ConversationView:
        user = current_user(request)
        self._owned_conversation(conversation_id, user.id)
        record = self.repository.rename_conversation(conversation_id, payload.title)
        if record is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        scoped = self.repository.get_conversation_for_user(conversation_id, user.id)
        if scoped is None:
            raise RuntimeError("Renamed conversation is not accessible to its owner.")
        return ConversationView.model_validate(scoped)

    def delete_conversation(self, conversation_id: str, request: Request) -> None:
        user = current_user(request)
        self._owned_conversation(conversation_id, user.id)
        self.runs.cancel_conversation(conversation_id)
        # 原因：会话记录删除后，其私有向量与图谱目录不应变成不可见的孤立数据。
        # 作用：显式删除聊天时同步清理该 conversation_id 的知识库，不影响其他聊天。
        self.knowledge.delete(conversation_id, global_scope=user.id)
        self.repository.delete_conversation(conversation_id)

    def messages(
        self,
        conversation_id: str,
        request: Request,
    ) -> list[MessageView]:
        user = current_user(request)
        self._accessible_conversation(conversation_id, user.id)
        return [
            MessageView.model_validate(item)
            for item in self.repository.list_messages(conversation_id)
        ]

    def members(
        self,
        conversation_id: str,
        request: Request,
    ) -> list[ConversationMemberView]:
        user = current_user(request)
        self._accessible_conversation(conversation_id, user.id)
        return [
            ConversationMemberView.model_validate(member)
            for member in self.repository.list_conversation_members(conversation_id)
        ]

    def share_conversation(
        self,
        conversation_id: str,
        payload: ConversationShareCreate,
        request: Request,
    ) -> ConversationMemberView:
        user = current_user(request)
        self._owned_conversation(conversation_id, user.id)
        member = self.repository.add_conversation_member(
            conversation_id,
            payload.username.strip(),
        )
        if member is None:
            raise HTTPException(
                status_code=404,
                detail="Active account not found or already owns this conversation.",
            )
        return ConversationMemberView.model_validate(member)

    def unshare_conversation(
        self,
        conversation_id: str,
        user_id: str,
        request: Request,
    ) -> None:
        user = current_user(request)
        self._owned_conversation(conversation_id, user.id)
        if not self.repository.is_conversation_member(conversation_id, user_id):
            raise HTTPException(status_code=404, detail="Shared member not found.")
        self.runs.cancel_user_conversation(conversation_id, user_id)
        if not self.repository.remove_conversation_member(conversation_id, user_id):
            raise HTTPException(status_code=404, detail="Shared member not found.")

    def start_run(
        self,
        conversation_id: str,
        payload: ChatStartRequest,
        request: Request,
    ) -> RunStarted:
        user = current_user(request)
        conversation = self._accessible_conversation(conversation_id, user.id)
        if conversation.is_owner and conversation.title in {"New chat", "新对话"}:
            self.repository.rename_conversation(
                conversation_id,
                _conversation_title(payload.content),
            )
        prepared = self.runs.prepare(
            conversation_id,
            payload.content,
            response_detail=payload.response_detail,
            interpretation_mode=payload.interpretation_mode,
        )
        if prepared.resolved_intent.requires_clarification:
            # 原因：缺少指代对象时调用模型只会放大猜测，并无谓占用远程或本地推理资源。
            # 作用：澄清作为正常完成的聊天轮次持久化，前端沿用同一轮询协议显示问题。
            return RunStarted(
                run_id=self.runs.complete_clarification(
                    conversation_id,
                    prepared,
                    user_id=user.id,
                    username=user.username,
                )
            )
        try:
            settings = self.runtime.require_online_settings()
        except ModelRuntimeError as exc:
            # 原因：MiniRAG 可用不代表知识 Agent 能在模型离线时生成最终自然语言答案。
            # 作用：不启动注定失败的 worker，并让正式前端显示可操作的 503 错误。
            raise HTTPException(
                status_code=503,
                detail=f"Model service is unavailable. {exc}",
            ) from exc
        run_id = self.runs.start(
            conversation_id,
            payload.content,
            settings,
            enable_web_search=payload.enable_web_search,
            enable_browser=payload.enable_browser,
            enable_local_knowledge=payload.enable_local_knowledge,
            include_global_knowledge=payload.include_global_knowledge,
            min_source_relevance=payload.min_source_relevance,
            max_evidence_sources=payload.max_evidence_sources,
            response_detail=payload.response_detail,
            interpretation_mode=payload.interpretation_mode,
            prepared=prepared,
            user_id=user.id,
            username=user.username,
            global_knowledge_path=self.knowledge.global_storage_path_for(user.id),
        )
        return RunStarted(run_id=run_id)

    def poll_run(self, run_id: str, request: Request) -> RunView:
        user = current_user(request)
        conversation_id = self.runs.conversation_id_for(run_id)
        if (
            conversation_id is None
            or self.repository.get_conversation_for_user(conversation_id, user.id) is None
        ):
            raise HTTPException(status_code=404, detail="Run not found.")
        result = self.runs.poll(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return result

    def cancel_run(self, run_id: str, request: Request) -> RunView:
        user = current_user(request)
        conversation_id = self.runs.conversation_id_for(run_id)
        if (
            conversation_id is None
            or self.repository.get_conversation_for_user(conversation_id, user.id) is None
        ):
            raise HTTPException(status_code=404, detail="Run not found.")
        result = self.runs.cancel(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return result

    def _accessible_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> ConversationRecord:
        conversation = self.repository.get_conversation_for_user(
            conversation_id,
            user_id,
        )
        if conversation is None:
            # 原因：403 会确认其他账号的会话 ID 确实存在，便于枚举私有资源。
            # 作用：不存在和无权访问统一返回 404，授权事实不泄漏给请求者。
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return conversation

    def _owned_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> None:
        if not self.repository.is_conversation_owner(conversation_id, user_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")


def _conversation_title(content: str) -> str:
    title = " ".join(content.split())
    return title if len(title) <= 48 else f"{title[:47]}…"
