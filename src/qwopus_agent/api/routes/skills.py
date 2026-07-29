"""Review and lifecycle routes for learned reusable Skills."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from qwopus_agent.api.models import SkillStepView, SkillVersionView
from qwopus_agent.services.skill_growth_service import SkillGrowthService
from qwopus_agent.skills import SkillManifest


def build_skill_router(growth: SkillGrowthService) -> APIRouter:
    """Build explicit candidate review, promotion, rejection, and rollback routes."""
    router = APIRouter()

    @router.get("/api/skills", response_model=list[SkillVersionView])
    def list_skills() -> list[SkillVersionView]:
        return [skill_version_view(growth, manifest) for manifest in growth.catalog.list()]

    @router.post(
        "/api/skills/{name}/{version}/promote",
        response_model=SkillVersionView,
    )
    def promote_skill(name: str, version: str) -> SkillVersionView:
        return _run_action(growth, growth.promote, name, version)

    @router.post(
        "/api/skills/{name}/{version}/reject",
        response_model=SkillVersionView,
    )
    def reject_skill(name: str, version: str) -> SkillVersionView:
        return _run_action(growth, growth.reject, name, version)

    @router.post(
        "/api/skills/{name}/{version}/rollback",
        response_model=SkillVersionView,
    )
    def rollback_skill(name: str, version: str) -> SkillVersionView:
        return _run_action(growth, growth.rollback, name, version)

    return router


def _run_action(
    growth: SkillGrowthService,
    action: Callable[[str, str], SkillManifest],
    name: str,
    version: str,
) -> SkillVersionView:
    """Normalize lifecycle failures without exposing filesystem details."""
    try:
        manifest = action(name, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return skill_version_view(growth, manifest)


def skill_version_view(
    growth: SkillGrowthService,
    manifest: SkillManifest,
) -> SkillVersionView:
    """Map a manifest and confined spec to the public review contract."""
    spec = growth.spec_for(manifest)
    return SkillVersionView(
        name=manifest.name,
        version=manifest.version,
        description=manifest.description,
        status=manifest.status,
        created_at=manifest.created_at,
        source_run_id=manifest.source_run_id,
        source_model=manifest.source_model,
        intent_examples=list(spec.intent_examples) if spec is not None else [],
        steps=(
            [SkillStepView(skill_name=step.skill_name) for step in spec.steps]
            if spec is not None
            else []
        ),
        spec_valid=spec is not None,
    )
