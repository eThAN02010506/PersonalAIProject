"""Verify model self-computed spreadsheet values against local re-computation.

When a weak model writes numeric statistics (mean, IQR outlier count, p-value,
R-squared) in its final answer without calling the local Skill tools, the
runtime currently fail-closes.  This module lets the runtime re-compute the
required method locally and accept the answer when the model's claimed prose
value matches the local result within tolerance.  When the local recompute
succeeds but the claimed value is missing or mismatched, the runtime degrades
to showing the verified local table instead of failing.  Only a failed local
recompute stays fail-closed: the final tables always come from the local Skill.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.integrations import smolagents_debug
from qwopus_agent.skills import SkillRequest
from qwopus_agent.skills.base import BaseSkill
from qwopus_agent.skills.excel_modeling import ExcelModelingSkill
from qwopus_agent.skills.excel_statistics import ExcelStatisticsSkill

# 原因：本地 Skill 保留 6 位小数，模型叙述通常只报 3-4 位。
# 作用：绝对容差吸收模型舍入，相对容差处理量纲差异。
VERIFY_ABS_TOL = 1e-2
VERIFY_REL_TOL = 1e-3


def local_verify_missing_spreadsheet_methods(
    steps: list[dict[str, Any]],
    *,
    spreadsheet_paths: dict[str, Path],
    spreadsheet_names: list[str],
    user_question: str,
    missing_tools: set[str],
    required_spreadsheet_methods: tuple[tuple[str, str], ...],
    debug_steps: list[str],
    narrative: str = "",
) -> str:
    """Recompute each still-missing required method and verify the model's prose value.

    Returns a replacement narrative sentence when at least one method verified
    or degraded to a local table; appends the synthetic tool step and discards
    the missing tool so the local table reaches the final answer.  Returns
    ``""`` when nothing could be recomputed.
    """
    if not spreadsheet_paths:
        return ""
    model_narrative = narrative or _extract_narrative(steps)
    if not model_narrative:
        return ""
    for tool_name, method in required_spreadsheet_methods:
        if tool_name not in missing_tools:
            continue
        verified = verify_one_method(
            (tool_name, method),
            spreadsheet_names=spreadsheet_names,
            spreadsheet_paths=spreadsheet_paths,
            user_question=user_question,
            narrative=model_narrative,
            debug_steps=debug_steps,
            step_number=len(steps),
            steps=steps,
        )
        if verified is None:
            continue
        synthetic_step, prose = verified
        steps.append(synthetic_step)
        missing_tools.discard(tool_name)
        if prose:
            return prose
    return ""


def verify_one_method(
    method: tuple[str, str],
    *,
    spreadsheet_names: list[str],
    spreadsheet_paths: dict[str, Path],
    user_question: str,
    narrative: str,
    debug_steps: list[str],
    step_number: int = 0,
    steps: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Recompute one required method and accept when locally recomputable.

    Returns ``(synthetic_step, prose)`` when the local recompute succeeds:
    with a verified prose when the model's claimed value matches, or a
    degraded prose that shows the local table without citing the model's value
    when the claim is missing or mismatched.  Returns ``None`` only when the
    local recompute itself fails (no trusted table can be provided).
    """
    tool_name, method_name = method
    if method_name not in _VERIFIABLE_METHODS:
        return None
    schema_targets = _schema_targets(steps or [])
    for spreadsheet_name in spreadsheet_names:
        path = spreadsheet_paths.get(spreadsheet_name)
        if path is None:
            continue
        try:
            response = _run_local_method(
                tool_name,
                method_name,
                path,
                user_question,
                schema_targets=schema_targets,
            )
        except (OSError, ValueError, TypeError) as exc:
            debug_steps.append(
                f"本地校验复算失败：{spreadsheet_name} / {method_name}: {exc}"
            )
            continue
        if not response.success or not response.data:
            debug_steps.append(
                f"本地校验复算无结果：{spreadsheet_name} / {method_name}"
            )
            continue
        local_values = _local_comparison_values(method_name, response.data)
        if not local_values:
            continue
        synthetic_step = {
            "step_number": step_number + 1,
            "observations": response.content,
            "tool_calls": [
                {
                    "function": {
                        "name": tool_name,
                        "arguments": {
                            "file_name": spreadsheet_name,
                            **(_method_arguments(method_name, response.data)),
                        },
                    }
                }
            ],
        }
        claimed = _extract_claimed_value(method_name, narrative)
        if claimed is not None and _values_match(local_values, claimed):
            prose = (
                f"已在 {spreadsheet_name} 上复核 {method_name}："
                f"{_prose_claim(method_name, claimed)}。"
            )
            debug_steps.append(
                f"本地校验通过：{spreadsheet_name} / {method_name} / "
                "自算值与本地复算一致"
            )
        else:
            # 原因：模型自算值缺失或与本地不符，但本地复算已成功。
            # 作用：降级为展示本地核验表，不再引用模型自算值，避免用户拿到报错。
            prose = (
                f"已在 {spreadsheet_name} 上按 {method_name} 重新计算，"
                "核验表见下方。"
            )
            debug_steps.append(
                f"本地校验降级：{spreadsheet_name} / {method_name} / "
                "自算值缺失或不符，改用本地核验表"
            )
        return synthetic_step, prose
    return None


def _schema_targets(steps: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Parse exact table names and column names from excel_schema observations.

    原因：模型自算前通常已调用 excel_schema，其输出含确切表名和列名，
    比从问题文字猜测可靠得多。
    作用：优先用这些 schema 信息推导本地复算的目标表与列。
    """
    targets: dict[str, list[str]] = {}
    for observation in smolagents_debug.extract_tool_observations(
        steps, "excel_schema"
    ):
        table_names = re.findall(r"(?m)^##\s+Sheet:\s*([^\r\n]+)", observation)
        column_matches = re.findall(
            r"(?m)^-+\s*Column names?:\s*([^\r\n]+)", observation
        )
        columns: list[str] = []
        for line in column_matches:
            columns.extend(
                column.strip()
                for column in line.split(",")
                if column.strip()
            )
        for table_name in table_names:
            table_name = table_name.strip()
            targets.setdefault(table_name, [])
            for column in columns:
                if column not in targets[table_name]:
                    targets[table_name].append(column)
    return targets


def _run_local_method(
    tool_name: str,
    method_name: str,
    path: Path,
    user_question: str,
    schema_targets: dict[str, list[str]] | None = None,
) -> Any:
    """Run the local Skill once and return its SkillResponse."""
    skill: BaseSkill = (
        ExcelStatisticsSkill()
        if tool_name == "excel_statistics"
        else ExcelModelingSkill()
    )
    frames = read_spreadsheet(path).analysis_frames()
    table_name = _choose_table(frames, user_question, schema_targets or {})
    arguments: dict[str, Any] = {
        "file_path": str(path),
        "table_name": table_name,
        "method": method_name,
    }
    arguments.update(
        _derive_arguments(
            method_name,
            frames[table_name],
            user_question,
            schema_targets or {},
        )
    )
    return asyncio.run(
        skill.run(
            SkillRequest(
                query=user_question,
                arguments=arguments,
            )
        )
    )


def _choose_table(
    frames: dict[str, Any],
    user_question: str,
    schema_targets: dict[str, list[str]] | None = None,
) -> str:
    """Pick the frame whose columns best match the user question or schema."""
    if len(frames) == 1:
        return next(iter(frames))
    lowered = user_question.casefold()
    schema_targets = schema_targets or {}
    # 原因：schema 输出含模型已看过的确切表名和列名，比问题文字匹配可靠。
    # 作用：优先选 schema 表名或列名命中的表，回退到问题文字匹配。
    for table_name in schema_targets:
        if table_name.casefold() in lowered:
            for candidate in frames:
                if candidate.casefold() in table_name.casefold():
                    return candidate
            return next(iter(frames))
    schema_columns = {
        column.casefold()
        for columns in schema_targets.values()
        for column in columns
    }
    scored = [
        (
            sum(
                1
                for column in frame.columns
                if str(column).casefold() in lowered
                or str(column).casefold() in schema_columns
            ),
            name,
        )
        for name, frame in frames.items()
    ]
    best_name, best_score = max(
        ((name, score) for score, name in scored),
        key=lambda item: (item[1], len(item[0])),
    )
    return best_name if best_score > 0 else next(iter(frames))


def _derive_arguments(
    method_name: str,
    frame: Any,
    user_question: str,
    schema_targets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Map a required method to concrete Skill arguments from the selected frame."""
    lowered = user_question.casefold()
    schema_columns = [
        column
        for columns in (schema_targets or {}).values()
        for column in columns
    ]
    numeric_columns = [
        str(column)
        for column in frame.select_dtypes(include="number").columns
    ]
    non_numeric_columns = [
        str(column)
        for column in frame.columns
        if str(column) not in numeric_columns
    ]
    # 优先使用 schema 中出现的列名，避免同名列或缩写歧义。
    schema_numeric = [
        column for column in numeric_columns if column in schema_columns
    ]
    if schema_numeric:
        numeric_columns = schema_numeric
    schema_non_numeric = [
        column
        for column in non_numeric_columns
        if column in schema_columns
    ]
    if schema_non_numeric:
        non_numeric_columns = schema_non_numeric
    if method_name == "linear_regression":
        outcome = next(
            (
                column
                for column in numeric_columns
                if column.casefold() in lowered
            ),
            numeric_columns[-1] if numeric_columns else "",
        )
        predictors = [
            column for column in numeric_columns if column != outcome
        ]
        return {
            "outcome_column": outcome,
            "predictor_columns": predictors,
        }
    if method_name == "one_way_anova":
        group = next(
            (
                column
                for column in non_numeric_columns
                if column.casefold() in lowered
            ),
            non_numeric_columns[0] if non_numeric_columns else "",
        )
        outcome = next(
            (
                column
                for column in numeric_columns
                if column.casefold() in lowered
            ),
            numeric_columns[0] if numeric_columns else "",
        )
        return {"outcome_column": outcome, "group_column": group}
    named = [
        column
        for column in numeric_columns
        if column.casefold() in lowered
    ]
    value_columns = named or numeric_columns
    if method_name in {"iqr_outliers", "zscore_outliers"}:
        return {"value_columns": value_columns, "label_columns": []}
    if method_name == "group_summary":
        return {
            "group_column": (
                non_numeric_columns[0] if non_numeric_columns else ""
            ),
            "value_columns": value_columns,
        }
    if method_name == "crosstab":
        return {"category_columns": non_numeric_columns[:2]}
    if method_name == "chi_square_independence":
        return {"category_columns": non_numeric_columns[:2]}
    if method_name == "two_sample_t_test":
        return {
            "value_columns": value_columns[:1],
            "group_column": (
                non_numeric_columns[0] if non_numeric_columns else ""
            ),
        }
    if method_name in {"one_sample_t_test"}:
        return {"value_columns": value_columns[:1], "hypothesized_mean": 0.0}
    if method_name == "correlation":
        return {"value_columns": value_columns}
    return {"value_columns": value_columns}


def _method_arguments(
    method_name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the tool-call arguments from the recompute response data."""
    rows = data.get("rows")
    if rows:
        return {"method": method_name, "value_columns": _value_columns_from_rows(rows)}
    return {"method": method_name}


def _value_columns_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Best-effort value-column recovery from a statistics result row set."""
    for row in rows:
        value = row.get("column")
        if isinstance(value, str) and value:
            return [value]
    return []


# Methods whose headline value a model can plausibly self-report.  The tuple is
# (comparison key, prose keyword regex) used to locate the claimed number.
_VERIFIABLE_METHODS = {
    "describe": "mean",
    "iqr_outliers": "outlier_count",
    "zscore_outliers": "outlier_count",
    "missing": "missing",
    "quantiles": "median",
    "normality_test": "p_value",
    "chi_square_independence": "p_value",
    "linear_regression": "r_squared",
    "one_way_anova": "p_value",
    "one_sample_t_test": "p_value",
    "mean_confidence_interval": "ci_lower",
}

_COMPARISON_EXTRACTORS = {
    "outlier_count": re.compile(
        r"outlier[_ ]?count\s*(?::?|=|is)\s*(\d+)", re.I
    ),
    "mean": re.compile(
        r"(?:平均|均值|mean)\D{0,20}?(-?\d[\d,]*(?:\.\d+)?)", re.I
    ),
    "median": re.compile(
        r"(?:中位|median)\D{0,20}?(-?\d[\d,]*(?:\.\d+)?)", re.I
    ),
    "missing": re.compile(r"(?:缺失|missing)\D{0,20}?(\d+)", re.I),
    "p_value": re.compile(
        r"(?:p[- ]?value|p值|p 值)\D{0,12}?(\d*\.?\d+(?:e-?\d+)?)", re.I
    ),
    "r_squared": re.compile(
        r"(?:r.?squared|R²|r²)\D{0,12}?(\d[\d.]*)", re.I
    ),
    "ci_lower": re.compile(r"ci[-_ ]?lower\D{0,12}?(-?\d[\d,.]*)", re.I),
}


def _extract_narrative(steps: list[dict[str, Any]]) -> str:
    """Join non-tool final prose from the steps for number extraction."""
    parts: list[str] = []
    for step in steps:
        observations = step.get("observations")
        if isinstance(observations, str) and observations.strip():
            parts.append(observations)
    return "\n".join(parts)


def _local_comparison_values(
    method_name: str,
    data: dict[str, Any],
) -> tuple[float, ...] | None:
    """Extract the local numeric value(s) to compare against for a method.

    统计方法的 headline 值(mean/p_value/ci_lower/r_squared/outlier_count)通常
    在结果表 rows 里,建模方法的 r_squared/f_p_value 在 named tables 里。依次
    从 rows、named tables、details 收集。
    """
    key = _VERIFIABLE_METHODS[method_name]
    details = data.get("details")
    rows = data.get("rows")
    if isinstance(rows, list):
        row_values = [
            float(value)
            for row in rows
            if isinstance(row, dict)
            and isinstance((value := row.get(key)), (int, float))
            and float(value) == float(value)
        ]
        if row_values:
            return tuple(row_values)
    tables = data.get("tables")
    if isinstance(tables, dict):
        table_values: list[float] = []
        for table in tables.values():
            if not isinstance(table, list):
                continue
            for row in table:
                if isinstance(row, dict) and isinstance(
                    (value := row.get(key)), (int, float)
                ) and float(value) == float(value):
                    table_values.append(float(value))
        if table_values:
            return tuple(table_values)
    if isinstance(details, dict):
        value = details.get(key)
        if isinstance(value, (int, float)):
            return (float(value),)
    return None


def _extract_claimed_value(
    method_name: str,
    narrative: str,
) -> tuple[float, ...] | None:
    """Extract the model's claimed numeric value(s) from its prose."""
    key = _VERIFIABLE_METHODS[method_name]
    pattern = _COMPARISON_EXTRACTORS[key]
    matches = [
        float(re.sub(r"[,\s]", "", match.group(1)))
        for match in pattern.finditer(narrative)
    ]
    return tuple(matches) if matches else None


def _values_match(
    local: tuple[float, ...],
    claimed: tuple[float, ...],
) -> bool:
    """Accept when any local value matches any claimed value within tolerance."""
    return any(
        _close_enough(local_value, claimed_value)
        for local_value in local
        for claimed_value in claimed
    )


def _close_enough(local: float, claimed: float) -> bool:
    if int(local) == int(claimed) and float(local).is_integer() and float(
        claimed
    ).is_integer():
        return int(local) == int(claimed)
    difference = abs(claimed - local)
    return difference <= VERIFY_ABS_TOL or difference <= VERIFY_REL_TOL * max(
        abs(local), 1.0
    )


def _prose_claim(method_name: str, claimed: tuple[float, ...]) -> str:
    """Render a stable human sentence describing the verified claimed value."""
    key = _VERIFIABLE_METHODS[method_name]
    if key == "outlier_count":
        return f"离群点数量 {int(claimed[0])}"
    return f"{key} = {claimed[0]:g}"
