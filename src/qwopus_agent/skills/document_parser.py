"""Document parser skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class DocumentParserSkill(BaseSkill):
    """Convert supported documents into Markdown before indexing."""

    # Reason: All document types should enter memory through one normalized Markdown pipeline.
    name: str = "document_parser"

    # Role: Parses PDF, DOCX, Markdown, and TXT into Markdown text.
    description: str = "Convert PDF, DOCX, Markdown, and TXT documents into Markdown."

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Validate document type before concrete parsers are added."""
        file_path = request.arguments.get("file_path")
        if not file_path:
            return SkillResponse(
                success=False,
                content="document_parser requires arguments.file_path.",
            )

        suffix = Path(str(file_path)).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            return SkillResponse(
                success=False,
                content=f"Unsupported document type: {suffix or '<none>'}.",
                data={"supported_extensions": sorted(SUPPORTED_EXTENSIONS)},
            )

        return SkillResponse(
            success=True,
            content="Document parser is registered and ready for Markdown conversion.",
            data={"file_path": str(file_path), "target_format": "markdown"},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return DocumentParserSkill()
