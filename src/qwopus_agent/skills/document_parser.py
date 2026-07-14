"""Document parser skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qwopus_agent.documents import parse_document
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
        """Parse a supported document into Markdown."""
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

        path = Path(str(file_path))
        if not path.exists():
            return SkillResponse(
                success=False,
                content=f"Document file does not exist: {path}",
            )

        try:
            # 原因：Skill 不能只做占位，Executor 需要拿到真实 Markdown 才能交给 MiniRAG 或 LLM。
            # 作用：复用 documents.parser 中的 MinerU/fallback 解析链路，保持解析逻辑只有一份。
            parsed = parse_document(path)
        except Exception as exc:
            return SkillResponse(
                success=False,
                content=f"Document parsing failed: {exc}",
                data={"file_path": str(path)},
            )

        return SkillResponse(
            success=True,
            content=parsed.markdown,
            data={
                "file_path": str(path),
                "markdown": parsed.markdown,
                "metadata": parsed.metadata,
            },
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return DocumentParserSkill()
