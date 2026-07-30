"""Model runtime status and configuration routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from qwopus_agent.api.auth import require_admin
from qwopus_agent.api.model_runtime import (
    ModelRuntimeError,
    RuntimeModelController,
    RuntimeModelStatus,
)
from qwopus_agent.api.models import ModelSettingsUpdate, ModelSettingsView
from qwopus_agent.llm import ModelCapabilities


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
    async def update_model_settings(
        payload: ModelSettingsUpdate,
        request: Request,
    ) -> ModelSettingsView:
        require_admin(request)
        capabilities = ModelCapabilities(
            context_window_tokens=payload.context_window_tokens,
            agent_mode=payload.agent_mode,
            supports_structured_output=payload.supports_structured_output,
            supports_vision=payload.supports_vision,
        )
        try:
            if payload.mode == "remote":
                if not payload.base_url:
                    raise ModelRuntimeError("Model address is required for remote mode.")
                status = await asyncio.to_thread(
                    runtime.configure_remote,
                    payload.base_url,
                    capabilities,
                    timeout_seconds=payload.request_timeout_seconds,
                    max_retries=payload.max_retries,
                    run_timeout_seconds=payload.run_timeout_seconds,
                )
            else:
                if not payload.model_path:
                    raise ModelRuntimeError("Model path is required for local mode.")
                status = await asyncio.to_thread(
                    runtime.configure_local,
                    payload.model_path,
                    capabilities,
                    timeout_seconds=payload.request_timeout_seconds,
                    max_retries=payload.max_retries,
                    run_timeout_seconds=payload.run_timeout_seconds,
                )
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
        context_window_tokens=status.settings.capabilities.context_window_tokens,
        agent_mode=status.settings.capabilities.agent_mode,
        supports_structured_output=(
            status.settings.capabilities.supports_structured_output
        ),
        supports_vision=status.settings.capabilities.supports_vision,
        request_timeout_seconds=status.settings.timeout_seconds,
        max_retries=status.settings.max_retries,
        run_timeout_seconds=status.settings.run_timeout_seconds,
    )
