"""smolagents adapters for schema-first spreadsheet analysis."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.analysis.pandas_sandbox import execute_pandas_code
from qwopus_agent.utils.token_budget import (
    TokenBudgetManager,
    estimate_tokens,
    truncate_to_tokens,
)


def build_excel_schema_tool(
    spreadsheet_contexts: Mapping[str, str],
    *,
    budget_manager: TokenBudgetManager | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Expose spreadsheet schema, samples, and local summaries only."""
    Tool = _smolagents_tool_class()
    contexts = {
        str(file_name): str(context)
        for file_name, context in spreadsheet_contexts.items()
        if str(context).strip()
    }
    if not contexts:
        raise ValueError("spreadsheet_contexts must not be empty.")
    budget = budget_manager or TokenBudgetManager()
    output_budget = max_tokens or budget.observation_budget
    available_files = ", ".join(contexts)

    class ExcelSchemaTool(Tool):  # type: ignore[misc, valid-type]
        name = "excel_schema"
        description = (
            "Inspect spreadsheet sheet names, columns, data types, sample rows, "
            "and local summaries. Call this before excel_analysis. It never returns "
            f"the full spreadsheet. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            }
        }
        output_type = "string"

        def forward(self, file_name: str) -> str:
            context = str(_lookup_file_value(contexts, file_name))
            # 原因：LLM 只需要 schema、样本和本地统计来设计分析代码。
            # 作用：严格阻止整份 Excel 数据通过 Tool 进入模型上下文。
            if estimate_tokens(context) <= output_budget:
                return context
            return (
                f"{truncate_to_tokens(context, output_budget)}\n\n"
                "[Spreadsheet schema context truncated by the tool.]"
            )

    return ExcelSchemaTool()


def build_excel_analysis_tool(spreadsheets: Mapping[str, str | Path]) -> Any:
    """Execute Agent-generated pandas code in the existing local sandbox."""
    Tool = _smolagents_tool_class()
    spreadsheet_paths = {
        str(file_name): Path(file_path) for file_name, file_path in spreadsheets.items()
    }
    if not spreadsheet_paths:
        raise ValueError("spreadsheets must not be empty.")
    available_files = ", ".join(spreadsheet_paths)

    class ExcelAnalysisTool(Tool):  # type: ignore[misc, valid-type]
        name = "excel_analysis"
        description = (
            "Execute restricted pandas code against an uploaded spreadsheet locally. "
            "Call excel_schema first. The code may use only dfs and pd and must "
            "assign its final value "
            f"to result. Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            },
            "code": {
                "type": "string",
                "description": "Restricted pandas code that assigns the computed answer to result.",
            },
        }
        output_type = "string"

        def forward(self, file_name: str, code: str) -> str:
            path = _lookup_file_value(spreadsheet_paths, file_name)
            if not path.exists():
                raise FileNotFoundError(f"Spreadsheet does not exist: {path}")
            spreadsheet = read_spreadsheet(path)
            # 原因：模型负责提出分析代码，但不能直接运行任意 Python 或读取本机文件。
            # 作用：在 AST 受限沙箱内针对本地 DataFrame 执行，只把计算结果返回 Agent。
            execution = execute_pandas_code(code, spreadsheet.sheets)
            return str(execution.markdown)

    return ExcelAnalysisTool()


def _smolagents_tool_class() -> Any:
    """Load Tool lazily so schema inspection remains usable without smolagents."""
    try:
        from smolagents import Tool
    except ModuleNotFoundError as exc:
        raise RuntimeError("smolagents is required to build Agent tools.") from exc
    return Tool


def _lookup_file_value(values: Mapping[str, Any], file_name: str) -> Any:
    """Resolve an exact Agent-provided file name without allowing arbitrary paths."""
    normalized_name = file_name.strip()
    if normalized_name not in values:
        available_files = ", ".join(values)
        raise ValueError(
            f"Unknown file_name: {normalized_name}. Available files: {available_files}."
        )
    return values[normalized_name]
