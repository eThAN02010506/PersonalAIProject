"""Excel schema inspection skill."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class ExcelSchemaSkill(BaseSkill):
    """Inspect workbook structure before any LLM analysis."""

    # Reason: Planner needs a stable capability name for schema-only Excel inspection.
    name: str = "excel_schema"

    # Role: Explains that this skill summarizes workbook structure without analyzing all rows.
    description: str = "Inspect Excel schema, sheet names, columns, dtypes, and safe sample rows."

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Return a safe placeholder until the concrete Excel parser is added."""
        file_path = request.arguments.get("file_path")
        if not file_path:
            return SkillResponse(
                success=False,
                content="excel_schema requires arguments.file_path.",
            )

        return SkillResponse(
            success=True,
            content="Excel schema inspection is registered and ready for parser implementation.",
            data={"file_path": str(file_path)},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return ExcelSchemaSkill()
