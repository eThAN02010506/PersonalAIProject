"""Verify model self-computed spreadsheet values against local re-computation.

When a weak model writes numeric statistics (mean, IQR outlier count, p-value,
R-squared) in its final answer without calling the local Skill tools, the
runtime currently fail-closes.  This module lets the runtime re-compute the
required method locally and accept the answer when the model's claimed prose
value matches the local result within tolerance.  Mismatches and missing
numbers stay fail-closed: the final tables always come from the local Skill.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from qwopus_agent.analysis.excel_processing import read_spreadsheet
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

    Returns a replacement narrative sentence when at least one method verified;
    appends the synthetic tool step and discards the missing tool so the local
    table reaches the final answer.  Returns ``""`` when nothing verified.
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
) -> tuple[dict[str, Any], str] | None:
    """Recompute one required method and compare against the model's prose claim.

    Returns ``(synthetic_step, prose)`` when the local value matches a number the
    model stated, else ``None`` (fail-closed).  The prose is a short sentence the
    runner can substitute as the final answer narrative.
    """
    tool_name, method_name = method
    if method_name not in _VERIFIABLE_METHODS:
        return None
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
            )
        except (OSError, ValueError, TypeError) as exc:
            debug_steps.append(
                f"本地校验复算失败：{spreadsheet_name} / {method_name}: {exc}"
            )
            continue
        if not response.success or not response.data:
            continue
        local_values = _local_comparison_values(method_name, response.data)
        if not local_values:
            continue
        claimed = _extract_claimed_value(method_name, narrative)
        if claimed is None or not _values_match(local_values, claimed):
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
        prose = (
            f"已在 {spreadsheet_name} 上复核 {method_name}："
            f"{_prose_claim(method_name, claimed)}。"
        )
        debug_steps.append(
            f"本地校验通过：{spreadsheet_name} / {method_name} / 自算值与本地复算一致"
        )
        return synthetic_step, prose
    return None


def _run_local_method(
    tool_name: str,
    method_name: str,
    path: Path,
    user_question: str,
) -> Any:
    """Run the local Skill once and return its SkillResponse."""
    skill: BaseSkill = (
        ExcelStatisticsSkill()
        if tool_name == "excel_statistics"
        else ExcelModelingSkill()
    )
    frames = read_spreadsheet(path).analysis_frames()
    table_name = _choose_table(frames, user_question)
    arguments: dict[str, Any] = {
        "file_path": str(path),
        "table_name": table_name,
        "method": method_name,
    }
    arguments.update(_derive_arguments(method_name, frames[table_name], user_question))
    return asyncio.run(
        skill.run(
            SkillRequest(
                query=user_question,
                arguments=arguments,
            )
        )
    )


def _choose_table(frames: dict[str, Any], user_question: str) -> str:
    """Pick the frame whose columns best match the user question."""
    if len(frames) == 1:
        return next(iter(frames))
    lowered = user_question.casefold()
    scored = [
        (sum(1 for column in frame.columns if str(column).casefold() in lowered), name)
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
) -> dict[str, Any]:
    """Map a required method to concrete Skill arguments from the selected frame."""
    lowered = user_question.casefold()
    numeric_columns = [
        str(column)
        for column in frame.select_dtypes(include="number").columns
    ]
    non_numeric_columns = [
        str(column)
        for column in frame.columns
        if str(column) not in numeric_columns
    ]
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
        return {"group_column": group}
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
    "one_way_anova": "f_p_value",
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
    "f_p_value": re.compile(
        r"(?:f[- ]?p|p[- ]?value|p值)\D{0,12}?(\d*\.?\d+(?:e-?\d+)?)",
        re.I,
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
    """Extract the local numeric value(s) to compare against for a method."""
    details = data.get("details")
    if not isinstance(details, dict):
        return None
    key = _VERIFIABLE_METHODS[method_name]
    if key == "outlier_count":
        value = details.get("outlier_count")
        return (float(value),) if isinstance(value, (int, float)) else None
    if key == "p_value":
        value = details.get("p_value")
        return (float(value),) if isinstance(value, (int, float)) else None
    if key == "f_p_value":
        value = details.get("f_p_value")
        return (float(value),) if isinstance(value, (int, float)) else None
    if key == "r_squared":
        value = details.get("r_squared")
        return (float(value),) if isinstance(value, (int, float)) else None
    if key == "missing":
        value = details.get("missing_count") or details.get("missing")
        return (float(value),) if isinstance(value, (int, float)) else None
    if key == "ci_lower":
        value = details.get("ci_lower")
        return (float(value),) if isinstance(value, (int, float)) else None
    if key in {"mean", "median"}:
        rows = data.get("rows")
        if not isinstance(rows, list):
            return None
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return tuple(values) if values else None
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
