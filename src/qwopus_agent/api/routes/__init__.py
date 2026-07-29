"""Focused HTTP route factories for the FastAPI application."""

from qwopus_agent.api.routes.analysis import build_analysis_router
from qwopus_agent.api.routes.conversations import build_conversation_router
from qwopus_agent.api.routes.debug import build_debug_router
from qwopus_agent.api.routes.documents import build_document_router
from qwopus_agent.api.routes.local_folders import build_local_folder_router
from qwopus_agent.api.routes.models import build_model_router
from qwopus_agent.api.routes.reports import build_report_router
from qwopus_agent.api.routes.skill_authoring import build_skill_authoring_router
from qwopus_agent.api.routes.skills import build_skill_router

__all__ = [
    "build_analysis_router",
    "build_conversation_router",
    "build_debug_router",
    "build_document_router",
    "build_local_folder_router",
    "build_model_router",
    "build_report_router",
    "build_skill_authoring_router",
    "build_skill_router",
]
