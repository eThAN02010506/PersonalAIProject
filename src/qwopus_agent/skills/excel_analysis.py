"""Excel analysis skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qwopus_agent.analysis import analyze_uploaded_file

from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class ExcelAnalysisSkill(BaseSkill):
    """Analyze Excel files through local code execution, not raw LLM ingestion."""

    # Reason: Planner needs a distinct skill for analysis after schema inspection.
    name: str = "excel_analysis"

    # Role: Runs local pandas-style analysis from schema and samples, returning only computed results.
    description: str = "Generate and execute local dataframe analysis without sending full Excel data."

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Run the existing local spreadsheet analysis pipeline."""
        file_path = request.arguments.get("file_path")
        if not file_path:
            return SkillResponse(
                success=False,
                content="excel_analysis requires arguments.file_path.",
            )

        path = Path(str(file_path))
        if not path.exists():
            return SkillResponse(
                success=False,
                content=f"Spreadsheet file does not exist: {path}",
            )

        try:
            # 原因：第三步只要求把现有能力 Skill 化，暂时不引入 pandas 代码沙箱。
            # 作用：复用当前安全分析流程，返回 schema/sample/statistics 等本地计算结果。
            result = analyze_uploaded_file(path, user_question=request.query)
        except Exception as exc:
            return SkillResponse(
                success=False,
                content=f"Excel analysis failed: {exc}",
                data={"file_path": str(path)},
            )

        return SkillResponse(
            success=True,
            content=result.markdown_summary,
            data={
                "file_path": str(path),
                "metadata": result.metadata,
                "markdown_document": result.markdown_document,
                "table_names": sorted(result.tables),
            },
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return ExcelAnalysisSkill()
