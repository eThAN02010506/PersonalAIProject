"""Read-only routes for locally persisted parsed documents."""

from fastapi import APIRouter

from qwopus_agent.api.models import SavedDocumentView
from qwopus_agent.documents import DocumentStore


def build_document_router(document_store: DocumentStore) -> APIRouter:
    """Expose a safe inventory of documents that completed local persistence."""
    router = APIRouter()

    @router.get("/api/documents", response_model=list[SavedDocumentView])
    def list_documents() -> list[SavedDocumentView]:
        return [
            SavedDocumentView.model_validate(document, from_attributes=True)
            for document in document_store.list_documents()
        ]

    return router
