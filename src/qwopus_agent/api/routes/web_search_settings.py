"""Host-wide Tavily settings routes."""

from __future__ import annotations

import asyncio
import json
from urllib.error import HTTPError, URLError

from fastapi import APIRouter, HTTPException, Request

from qwopus_agent.api.auth import current_user, require_admin
from qwopus_agent.api.models import (
    TavilyConnectionTestView,
    TavilyKeyTestRequest,
    TavilyKeyUpdate,
    WebSearchSettingsView,
)
from qwopus_agent.integrations.tavily import TavilySearchConfig, TavilySearchProvider
from qwopus_agent.integrations.tavily_credentials import (
    TavilyCredentialError,
    TavilyCredentialStatus,
    TavilyCredentialStore,
)


def build_web_search_settings_router(
    credentials: TavilyCredentialStore,
) -> APIRouter:
    """Build credential routes around one injected host-local store."""
    router = APIRouter()

    @router.get(
        "/api/web-search-settings",
        response_model=WebSearchSettingsView,
    )
    async def settings(request: Request) -> WebSearchSettingsView:
        user = current_user(request)
        status = await asyncio.to_thread(credentials.status)
        return _settings_view(status, is_admin=user.role == "admin")

    @router.put(
        "/api/web-search-settings",
        response_model=WebSearchSettingsView,
    )
    async def update(
        payload: TavilyKeyUpdate,
        request: Request,
    ) -> WebSearchSettingsView:
        require_admin(request)
        try:
            status = await asyncio.to_thread(
                credentials.save,
                payload.api_key.get_secret_value(),
            )
        except TavilyCredentialError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _settings_view(status, is_admin=True)

    @router.delete(
        "/api/web-search-settings",
        response_model=WebSearchSettingsView,
    )
    async def delete(request: Request) -> WebSearchSettingsView:
        require_admin(request)
        try:
            status = await asyncio.to_thread(credentials.delete)
        except TavilyCredentialError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _settings_view(status, is_admin=True)

    @router.post(
        "/api/web-search-settings/test",
        response_model=TavilyConnectionTestView,
    )
    async def test(
        payload: TavilyKeyTestRequest,
        request: Request,
    ) -> TavilyConnectionTestView:
        require_admin(request)
        try:
            api_key = (
                payload.api_key.get_secret_value()
                if payload.api_key is not None
                else credentials.resolve()
            )
        except TavilyCredentialError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not api_key:
            raise HTTPException(status_code=400, detail="Tavily API key is not configured.")
        return await asyncio.to_thread(_test_connection, api_key)

    return router


def _test_connection(api_key: str) -> TavilyConnectionTestView:
    """Issue one bounded query without returning its evidence or credential."""
    provider = TavilySearchProvider(
        TavilySearchConfig(
            api_key=api_key,
            max_results=1,
            timeout_seconds=10,
        )
    )
    try:
        provider.search("Qwopus Agent Tavily connectivity test")
    except HTTPError as exc:
        message = f"Tavily rejected the request (HTTP {exc.code})."
        return TavilyConnectionTestView(success=False, message=message)
    except (TimeoutError, URLError):
        return TavilyConnectionTestView(
            success=False,
            message="Could not reach Tavily within 10 seconds.",
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return TavilyConnectionTestView(
            success=False,
            message="Tavily returned an invalid or unavailable response.",
        )
    return TavilyConnectionTestView(
        success=True,
        message="Tavily search is ready.",
    )


def _settings_view(
    status: TavilyCredentialStatus,
    *,
    is_admin: bool,
) -> WebSearchSettingsView:
    """Hide source and key identity from non-administrator accounts."""
    return WebSearchSettingsView(
        configured=status.configured,
        source=status.source if is_admin else None,
        masked_key=status.masked_key if is_admin else None,
        can_manage=is_admin,
        message=(
            "Tavily search is ready."
            if status.configured
            else "Tavily search has not been configured by the host administrator."
        ),
    )
