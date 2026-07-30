"""smolagents adapters for schema-first spreadsheet analysis."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.analysis.pandas_sandbox import (
    PANDAS_SANDBOX_CODE_GUIDANCE,
    execute_pandas_code,
)
from qwopus_agent.integrations.skill_tools import build_skill_tool
from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.excel_modeling import ExcelModelingSkill
from qwopus_agent.skills.excel_statistics import ExcelStatisticsSkill
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
            f"Call excel_schema first. {PANDAS_SANDBOX_CODE_GUIDANCE} "
            "Prefer a DataFrame with explicit metric "
            "and group labels so the bounded output is a verifiable Markdown table. "
            f"Available files: {available_files}."
        )
        inputs = {
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            },
            "code": {
                "type": "string",
                "description": PANDAS_SANDBOX_CODE_GUIDANCE,
            },
        }
        output_type = "string"

        def forward(self, file_name: str, code: str) -> str:
            path = _lookup_file_value(spreadsheet_paths, file_name)
            if not path.exists():
                raise FileNotFoundError(f"Spreadsheet does not exist: {path}")
            spreadsheet = read_spreadsheet(path)
            # 原因：模型负责提出分析代码，但不能直接运行任意 Python 或读取本机文件。
            # 作用：在 AST 受限沙箱内针对所有检测到的表区域执行，只把计算结果返回 Agent。
            try:
                execution = execute_pandas_code(code, spreadsheet.analysis_frames())
            except ValueError as exc:
                # 原因：只返回 AST 错误时，弱模型会原样重复同一段普通 pandas 脚本。
                # 作用：把可模仿的安全合同放入 Tool error，下一步可自行修正而不放宽沙箱。
                raise ValueError(f"{exc}\n{PANDAS_SANDBOX_CODE_GUIDANCE}") from exc
            return str(execution.markdown)

    return ExcelAnalysisTool()


def build_excel_statistics_tool(spreadsheets: Mapping[str, str | Path]) -> Any:
    """Expose common deterministic statistics for only the approved spreadsheets."""
    spreadsheet_paths = {
        str(file_name): Path(file_path) for file_name, file_path in spreadsheets.items()
    }
    if not spreadsheet_paths:
        raise ValueError("spreadsheets must not be empty.")
    available_files = ", ".join(spreadsheet_paths)

    def request_factory(values: Mapping[str, Any]) -> SkillRequest:
        file_name = str(values.get("file_name", "")).strip()
        path = _lookup_file_value(spreadsheet_paths, file_name)
        arguments = {
            key: value
            for key, value in values.items()
            if key != "file_name"
        }
        arguments["file_path"] = str(path)
        return SkillRequest(query="", arguments=arguments)

    # 原因：正式 Agent 只能使用本轮上传文件名，不能让通用 Skill 接收任意本地路径。
    # 作用：复用 Skill 业务逻辑，同时由 Tool 适配器完成批准文件名到路径的注入。
    return build_skill_tool(
        ExcelStatisticsSkill(),
        tool_name="excel_statistics",
        description=(
            "Run a reviewed local statistical method after excel_schema. Supported methods: "
            "describe, frequency, missing, iqr_outliers, zscore_outliers, group_summary, "
            "correlation, "
            "mean_confidence_interval, one_sample_t_test, and two_sample_t_test. "
            "Use describe for an R summary()-style numeric profile and frequency for "
            "R table()-style categorical counts. "
            "Use exact table and column names from excel_schema. For abstract outlier questions, "
            "prefer one interpretable business metric. Select multiple value columns only when "
            "they are repeated measurements of that same metric and unit, such as years; never "
            "mix counts, percentages, currencies, totals, or hierarchy levels in one row mean. "
            "If rows mix entities and aggregate groups, use the scope_* arguments to retain only "
            "entities classified by a metadata table before calculating statistics. "
            f"Available files: {available_files}."
        ),
        inputs={
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            },
            "table_name": {
                "type": "string",
                "description": "Exact sheet or table name from excel_schema.",
            },
            "method": {
                "type": "string",
                "description": "One supported statistical method name.",
            },
            "value_columns": {
                "type": "array",
                "description": "Exact comparable value column names.",
                "items": {"type": "string"},
            },
            "label_columns": {
                "type": "array",
                "description": "Exact identifier columns to return with each result row.",
                "items": {"type": "string"},
                "nullable": True,
            },
            "group_column": {
                "type": "string",
                "description": (
                    "Exact grouping column for group_summary or two_sample_t_test, otherwise null."
                ),
                "nullable": True,
            },
            "group_values": {
                "type": "array",
                "description": (
                    "Exactly two group labels for two_sample_t_test; null when the column "
                    "already contains exactly two groups or for other methods."
                ),
                "items": {"type": "string"},
                "nullable": True,
            },
            "confidence_level": {
                "type": "number",
                "description": (
                    "Confidence level between 0 and 1 for intervals and t-tests; normally 0.95."
                ),
                "nullable": True,
            },
            "hypothesized_mean": {
                "type": "number",
                "description": (
                    "Population mean in the one_sample_t_test null hypothesis, otherwise null."
                ),
                "nullable": True,
            },
            "scope_table_name": {
                "type": "string",
                "description": (
                    "Metadata table used to define a comparable population, otherwise null."
                ),
                "nullable": True,
            },
            "scope_data_key": {
                "type": "string",
                "description": "Key column in table_name matched to the scope table.",
                "nullable": True,
            },
            "scope_lookup_key": {
                "type": "string",
                "description": "Matching key column in scope_table_name.",
                "nullable": True,
            },
            "scope_required_columns": {
                "type": "array",
                "description": (
                    "Scope-table columns that must all be non-null, such as Region."
                ),
                "items": {"type": "string"},
                "nullable": True,
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum result rows, normally 20.",
                "nullable": True,
            },
            "threshold": {
                "type": "number",
                "description": "IQR multiplier (normally 1.5) or Z-score cutoff.",
                "nullable": True,
            },
        },
        request_factory=request_factory,
    )


def build_excel_modeling_tool(spreadsheets: Mapping[str, str | Path]) -> Any:
    """Expose reviewed regression and ANOVA models for approved spreadsheets."""
    spreadsheet_paths = {
        str(file_name): Path(file_path) for file_name, file_path in spreadsheets.items()
    }
    if not spreadsheet_paths:
        raise ValueError("spreadsheets must not be empty.")
    available_files = ", ".join(spreadsheet_paths)

    def request_factory(values: Mapping[str, Any]) -> SkillRequest:
        file_name = str(values.get("file_name", "")).strip()
        path = _lookup_file_value(spreadsheet_paths, file_name)
        arguments = {key: value for key, value in values.items() if key != "file_name"}
        arguments["file_path"] = str(path)
        return SkillRequest(query="", arguments=arguments)

    # 原因：回归和 ANOVA 的参数合同不同于描述统计，独立 Tool 可避免弱模型混用字段。
    # 作用：模型只选择获准文件、表和列，模型拟合与检验始终在本地 Skill 内完成。
    return build_skill_tool(
        ExcelModelingSkill(),
        tool_name="excel_modeling",
        description=(
            "Fit reviewed local spreadsheet models after excel_schema. Use linear_regression "
            "for an R summary(lm())-style OLS result, or one_way_anova for group means, "
            "ANOVA, effect sizes, variance diagnostics, and optional Tukey HSD. "
            f"Available files: {available_files}."
        ),
        inputs={
            "file_name": {
                "type": "string",
                "description": "Exact uploaded spreadsheet file name.",
            },
            "table_name": {
                "type": "string",
                "description": "Exact sheet or table name from excel_schema.",
            },
            "method": {
                "type": "string",
                "description": "linear_regression or one_way_anova.",
            },
            "outcome_column": {
                "type": "string",
                "description": "Exact numeric response column.",
            },
            "predictor_columns": {
                "type": "array",
                "description": (
                    "Numeric or categorical predictors for linear_regression, otherwise null."
                ),
                "items": {"type": "string"},
                "nullable": True,
            },
            "group_column": {
                "type": "string",
                "description": "Categorical group for one_way_anova, otherwise null.",
                "nullable": True,
            },
            "confidence_level": {
                "type": "number",
                "description": "Confidence level between 0 and 1; normally 0.95.",
                "nullable": True,
            },
            "include_posthoc": {
                "type": "boolean",
                "description": "Whether ANOVA should include Tukey HSD; normally true.",
                "nullable": True,
            },
        },
        request_factory=request_factory,
    )


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
