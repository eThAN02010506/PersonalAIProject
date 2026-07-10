"""Excel analysis skill."""

from __future__ import annotations

from dataclasses import dataclass

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class ExcelAnalysisSkill(BaseSkill):
    """Analyze Excel files through local code execution, not raw LLM ingestion."""

    # Reason: Planner needs a distinct skill for analysis after schema inspection.
    name: str = "excel_analysis"

    # Role: Runs local pandas-style analysis from schema and samples, returning only computed results.
    description: str = "Generate and execute local dataframe analysis without sending full Excel data."

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Return a safe placeholder until the pandas execution sandbox is added."""
        if "schema" not in request.arguments:
            return SkillResponse(
                success=False,
                content="excel_analysis requires arguments.schema from excel_schema.",
            )

        return SkillResponse(
            success=True,
            content="Excel analysis skill is registered and ready for local code execution.",
            data={"query": request.query},
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return ExcelAnalysisSkill()
