"""Excel schema inspection skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse


@dataclass
class ExcelSchemaSkill(BaseSkill):
    """Inspect workbook structure before any LLM analysis."""

    # 原因：Excel Skill 接收本地路径，必须复用上传服务的文件授权边界。
    # 作用：自动发现保留，但普通聊天不会获得任意工作簿读取能力。
    agent_tool_permission: ClassVar[str | None] = "documents"

    # Reason: Planner needs a stable capability name for schema-only Excel inspection.
    name: str = "excel_schema"

    # Role: Explains that this skill summarizes workbook structure without analyzing all rows.
    description: str = "Inspect Excel schema, sheet names, columns, dtypes, and safe sample rows."

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Inspect workbook schema and safe sample rows."""
        file_path = request.arguments.get("file_path")
        if not file_path:
            return SkillResponse(
                success=False,
                content="excel_schema requires arguments.file_path.",
            )

        path = Path(str(file_path))
        if not path.exists():
            return SkillResponse(
                success=False,
                content=f"Spreadsheet file does not exist: {path}",
            )

        try:
            # 原因：模型做 Excel 分析前只需要字段结构和少量样例，不应该看到整表。
            # 作用：复用本地 Excel 读取与表头识别逻辑，返回可控 schema/sample。
            spreadsheet = read_spreadsheet(path)
        except Exception as exc:
            return SkillResponse(
                success=False,
                content=f"Excel schema inspection failed: {exc}",
                data={"file_path": str(path)},
            )

        sheets: dict[str, Any] = {}
        content_lines = [f"# Excel Schema: {path.name}"]
        for table_name, dataframe in spreadsheet.analysis_frames().items():
            source_sheet = table_name.split("::", maxsplit=1)[0]
            column_names = [str(column) for column in dataframe.columns]
            sheet_data = {
                "source_sheet": source_sheet,
                "rows": int(len(dataframe)),
                "columns": int(len(dataframe.columns)),
                "column_names": column_names,
                "dtypes": {
                    str(column): str(dtype)
                    for column, dtype in dataframe.dtypes.items()
                },
                "sample_rows": dataframe.head(3).fillna("").to_dict(orient="records"),
                "metadata": spreadsheet.metadata.get(source_sheet, {}),
            }
            sheets[table_name] = sheet_data
            # 原因：SkillResponse.content 要给 Planner/LLM 一个简短可读摘要。
            # 作用：把结构信息压缩成 Markdown，避免 UI 或 Agent 解析复杂对象。
            heading = "Sheet" if table_name == source_sheet else "Table"
            content_lines.extend(
                [
                    "",
                    f"## {heading}: {table_name}",
                    f"- Rows: {sheet_data['rows']}",
                    f"- Columns: {sheet_data['columns']}",
                    f"- Column names: {', '.join(column_names)}",
                ]
            )

        return SkillResponse(
            success=True,
            content="\n".join(content_lines),
            data={
                "file_path": str(path),
                "sheets": sheets,
                "workbook_profile": (
                    spreadsheet.profile.model_dump(mode="json")
                    if spreadsheet.profile is not None
                    else None
                ),
            },
        )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return ExcelSchemaSkill()
