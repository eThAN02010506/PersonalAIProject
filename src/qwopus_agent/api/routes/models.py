"""Model runtime status and configuration routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from qwopus_agent.api.model_runtime import (
    ModelRuntimeError,
    RuntimeModelController,
    RuntimeModelStatus,
)
from qwopus_agent.api.models import ModelSettingsUpdate, ModelSettingsView


def build_model_router(runtime: RuntimeModelController) -> APIRouter:
    """Build routes that depend only on the model runtime controller."""
    router = APIRouter()

    @router.get("/api/health", response_model=ModelSettingsView)
    async def health() -> ModelSettingsView:
        return _model_settings_view(await asyncio.to_thread(runtime.status))

    @router.get("/api/model-settings", response_model=ModelSettingsView)
    async def model_settings() -> ModelSettingsView:
        return _model_settings_view(await asyncio.to_thread(runtime.status))

    @router.put("/api/model-settings", response_model=ModelSettingsView)
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

    return router


def _model_settings_view(status: RuntimeModelStatus) -> ModelSettingsView:
    return ModelSettingsView(
        mode=status.mode,
        model_online=status.online,
        message=status.message,
        model=status.settings.model_id,
        base_url=status.settings.base_url,
        local_model_path=status.local_model_path,
    )
