"""Conversation persistence and background Agent run routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from qwopus_agent.api.model_runtime import RuntimeModelController
from qwopus_agent.api.models import (
    ChatStartRequest,
    ConversationCreate,
    ConversationUpdate,
    ConversationView,
    MessageView,
    RunStarted,
    RunView,
)
from qwopus_agent.api.repository import ConversationRepository
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

    def conversations(self) -> list[ConversationView]:
        return [
            ConversationView.model_validate(item)
            for item in self.repository.list_conversations()
        ]

    def create_conversation(self, payload: ConversationCreate) -> ConversationView:
        return ConversationView.model_validate(
            self.repository.create_conversation(payload.title)
        )

    def rename_conversation(
        self,
        conversation_id: str,
        payload: ConversationUpdate,
    ) -> ConversationView:
        record = self.repository.rename_conversation(conversation_id, payload.title)
        if record is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return ConversationView.model_validate(record)

    def delete_conversation(self, conversation_id: str) -> None:
        if self.repository.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        # 原因：会话记录删除后，其私有向量与图谱目录不应变成不可见的孤立数据。
        # 作用：显式删除聊天时同步清理该 conversation_id 的知识库，不影响其他聊天。
        self.knowledge.delete(conversation_id)
        self.repository.delete_conversation(conversation_id)

    def messages(self, conversation_id: str) -> list[MessageView]:
        if self.repository.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return [
            MessageView.model_validate(item)
            for item in self.repository.list_messages(conversation_id)
        ]

    def start_run(
        self,
        conversation_id: str,
        payload: ChatStartRequest,
    ) -> RunStarted:
        conversation = self.repository.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        if conversation.title in {"New chat", "新对话"}:
            self.repository.rename_conversation(
                conversation_id,
                _conversation_title(payload.content),
            )
        run_id = self.runs.start(
            conversation_id,
            payload.content,
            self.runtime.current_settings(),
            enable_web_search=payload.enable_web_search,
            enable_local_knowledge=payload.enable_local_knowledge,
            include_global_knowledge=payload.include_global_knowledge,
            min_source_relevance=payload.min_source_relevance,
        )
        return RunStarted(run_id=run_id)

    def poll_run(self, run_id: str) -> RunView:
        result = self.runs.poll(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return result

    def cancel_run(self, run_id: str) -> RunView:
        result = self.runs.cancel(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return result


def _conversation_title(content: str) -> str:
    title = " ".join(content.split())
    return title if len(title) <= 48 else f"{title[:47]}…"
