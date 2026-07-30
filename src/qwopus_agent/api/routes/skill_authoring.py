"""Model-assisted Workflow Skill authoring routes for approved Debug clients."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from qwopus_agent.api.auth import require_admin
from qwopus_agent.api.debug_access import require_debug_client
from qwopus_agent.api.models import (
    SkillAuthoringRequest,
    SkillCandidateCheckView,
    SkillCandidateReviewView,
    SkillCandidateTestRequest,
    SkillCandidateTestStepView,
    SkillCandidateTestView,
    SkillCapabilityView,
    SkillFromRunsRequest,
    SkillSourceConversationView,
    SkillSourceRunView,
)
from qwopus_agent.api.repository import ConversationRepository
from qwopus_agent.api.routes.skills import skill_version_view
from qwopus_agent.services.skill_authoring_service import (
    CandidateReview,
    ConversationSkillRun,
    SkillAuthoringService,
)


def build_skill_authoring_router(
    authoring: SkillAuthoringService,
    repository: ConversationRepository,
) -> APIRouter:
    """Build candidate-only authoring routes behind the Debug network boundary."""
    router = APIRouter()
    @router.get(
        "/api/debug/skills/source-conversations",
        response_model=list[SkillSourceConversationView],
    )
    async def source_conversations(
        request: Request,
    ) -> list[SkillSourceConversationView]:
        require_admin(request)
        require_debug_client(request)
        conversations = await asyncio.to_thread(
            repository.list_conversations_with_reusable_runs
        )
        return [
            SkillSourceConversationView(
                id=conversation.id,
                title=conversation.title,
                owner_username=conversation.owner_username,
                updated_at=conversation.updated_at,
            )
            for conversation in conversations
        ]

    @router.get(
        "/api/debug/skills/source-conversations/{conversation_id}/runs",
        response_model=list[SkillSourceRunView],
    )
    async def source_runs(
        request: Request,
        conversation_id: str,
    ) -> list[SkillSourceRunView]:
        require_admin(request)
        require_debug_client(request)
        if repository.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        runs = await asyncio.to_thread(
            repository.list_conversation_runs,
            conversation_id,
        )
        messages = {
            message.id: message
            for message in await asyncio.to_thread(
                repository.list_messages,
                conversation_id,
            )
        }
        return [
            SkillSourceRunView(
                run_id=run.run_id,
                conversation_id=run.conversation_id,
                objective=run.objective,
                operational_objective=run.operational_objective,
                model_id=run.model_id,
                reusable_skills=list(run.reusable_skills),
                answer_preview=(
                    messages[run.assistant_message_id].content[:500]
                    if run.assistant_message_id in messages
                    else ""
                ),
                created_at=run.created_at,
            )
            for run in runs
            if run.status == "completed" and run.reusable_skills
        ]

    @router.get(
        "/api/debug/skills/capabilities",
        response_model=list[SkillCapabilityView],
    )
    async def skill_capabilities(request: Request) -> list[SkillCapabilityView]:
        require_admin(request)
        require_debug_client(request)
        return [
            SkillCapabilityView(
                name=capability.name,
                description=capability.description,
            )
            for capability in authoring.capabilities()
        ]

    @router.post(
        "/api/debug/skills/generate",
        response_model=SkillCandidateReviewView,
    )
    async def generate_skill_candidate(
        request: Request,
        payload: SkillAuthoringRequest,
    ) -> SkillCandidateReviewView:
        require_admin(request)
        require_debug_client(request)
        try:
            # 原因：BaseLLM 使用同步 HTTP，直接调用会阻塞 FastAPI 事件循环和 Debug 轮询。
            # 作用：模型生成在工作线程中完成，候选仍由同进程 Catalog 原子持久化。
            review = await asyncio.to_thread(
                authoring.generate_candidate,
                goal=payload.goal,
                requested_name=payload.requested_name,
                intent_examples=payload.intent_examples,
                allowed_skills=payload.allowed_skills,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _candidate_review_view(authoring, review)

    @router.post(
        "/api/debug/skills/from-runs",
        response_model=SkillCandidateReviewView,
    )
    async def generate_skill_candidate_from_runs(
        request: Request,
        payload: SkillFromRunsRequest,
    ) -> SkillCandidateReviewView:
        require_admin(request)
        require_debug_client(request)
        persisted = await asyncio.to_thread(
            repository.list_conversation_runs,
            payload.conversation_id,
        )
        by_id = {
            run.run_id: run
            for run in persisted
            if run.status == "completed" and run.reusable_skills
        }
        selected = [by_id[run_id] for run_id in payload.run_ids if run_id in by_id]
        if len(selected) != len(payload.run_ids):
            raise HTTPException(
                status_code=422,
                detail="Every selected run must be reusable and belong to this conversation.",
            )
        try:
            review = await asyncio.to_thread(
                authoring.generate_candidate_from_runs,
                (
                    ConversationSkillRun(
                        run_id=run.run_id,
                        objective=run.objective,
                        operational_objective=run.operational_objective,
                        model_id=run.model_id,
                        reusable_skills=run.reusable_skills,
                    )
                    for run in selected
                ),
                requested_name=payload.requested_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _candidate_review_view(authoring, review)

    @router.get(
        "/api/debug/skills/{name}/{version}",
        response_model=SkillCandidateReviewView,
    )
    async def review_skill_candidate(
        request: Request,
        name: str,
        version: str,
    ) -> SkillCandidateReviewView:
        require_admin(request)
        require_debug_client(request)
        try:
            review = await asyncio.to_thread(
                authoring.review_candidate,
                name,
                version,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _candidate_review_view(authoring, review)

    @router.post(
        "/api/debug/skills/{name}/{version}/test",
        response_model=SkillCandidateTestView,
    )
    async def test_skill_candidate(
        request: Request,
        name: str,
        version: str,
        payload: SkillCandidateTestRequest,
    ) -> SkillCandidateTestView:
        require_admin(request)
        require_debug_client(request)
        try:
            result = await authoring.test_candidate(
                name,
                version,
                payload.query,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SkillCandidateTestView(
            success=result.success,
            output=result.output,
            steps=[
                SkillCandidateTestStepView(
                    skill_name=step.skill_name,
                    query=step.query,
                    argument_keys=list(step.argument_keys),
                )
                for step in result.steps
            ],
        )

    return router


def _candidate_review_view(
    authoring: SkillAuthoringService,
    review: CandidateReview,
) -> SkillCandidateReviewView:
    """Map complete review evidence without exposing candidate filesystem paths."""
    return SkillCandidateReviewView(
        skill=skill_version_view(authoring.growth, review.manifest),
        spec_json=review.spec_json,
        diff=review.diff,
        checks=[
            SkillCandidateCheckView(
                name=check.name,
                passed=check.passed,
                detail=check.detail,
            )
            for check in review.checks
        ],
        model_output=review.model_output,
    )
