"""Focused HTTP route factories for the FastAPI application."""

from qwopus_agent.api.routes.analysis import build_analysis_router
from qwopus_agent.api.routes.conversations import build_conversation_router
from qwopus_agent.api.routes.models import build_model_router
from qwopus_agent.api.routes.reports import build_report_router

__all__ = [
    "build_analysis_router",
    "build_conversation_router",
    "build_model_router",
    "build_report_router",
]
