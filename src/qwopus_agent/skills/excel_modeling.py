"""Deterministic regression and ANOVA models for spreadsheet analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.analysis.markdown_tables import dataframe_to_markdown
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse

SUPPORTED_METHODS = {"linear_regression", "one_way_anova"}


@dataclass
class ExcelModelingSkill(BaseSkill):
    """Fit reviewed statistical models without model-generated Python code."""

    agent_tool_permission: ClassVar[str | None] = "documents"
    name: str = "excel_modeling"
    description: str = (
        "Fit deterministic spreadsheet models: R summary(lm())-style linear regression "
        "or one-way ANOVA with optional Tukey post-hoc comparisons."
    )

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Validate one workbook model request and return auditable result tables."""
        try:
            path = Path(str(request.arguments["file_path"]))
            table_name = str(request.arguments["table_name"]).strip()
            method = str(request.arguments["method"]).strip()
            outcome_column = str(request.arguments["outcome_column"]).strip()
            predictor_columns = _column_list(
                request.arguments.get("predictor_columns")
            )
            group_column = str(request.arguments.get("group_column") or "").strip()
            confidence_level_argument = request.arguments.get("confidence_level")
            confidence_level = (
                0.95
                if confidence_level_argument is None
                else float(confidence_level_argument)
            )
            include_posthoc = bool(
                request.arguments.get("include_posthoc")
                if request.arguments.get("include_posthoc") is not None
                else True
            )
        except (KeyError, TypeError, ValueError) as exc:
            return SkillResponse(
                success=False,
                content=f"Invalid excel_modeling arguments: {exc}",
            )
        if method not in SUPPORTED_METHODS:
            return SkillResponse(
                success=False,
                content=(
                    f"Unsupported modeling method: {method}. "
                    f"Choose one of: {', '.join(sorted(SUPPORTED_METHODS))}."
                ),
            )
        if not path.is_file():
            return SkillResponse(
                success=False,
                content=f"Spreadsheet file does not exist: {path}",
            )
        if not 0.0 < confidence_level < 1.0:
            return SkillResponse(
                success=False,
                content="confidence_level must be greater than 0 and less than 1.",
            )

        try:
            frames = read_spreadsheet(path).analysis_frames()
            dataframe = frames[table_name]
            if method == "linear_regression":
                tables, details = _linear_regression(
                    dataframe,
                    outcome_column=outcome_column,
                    predictor_columns=predictor_columns,
                    confidence_level=confidence_level,
                )
            else:
                tables, details = _one_way_anova(
                    dataframe,
                    outcome_column=outcome_column,
                    group_column=group_column,
                    confidence_level=confidence_level,
                    include_posthoc=include_posthoc,
                )
        except KeyError as exc:
            available = ", ".join(frames) if "frames" in locals() else ""
            return SkillResponse(
                success=False,
                content=f"Unknown table or column: {exc}. Available tables: {available}.",
            )
        except (TypeError, ValueError) as exc:
            return SkillResponse(success=False, content=f"Modeling failed: {exc}")

        sections = [
            f"## Modeling result: {method}",
            f"- Table: {table_name}",
            *[f"- {key}: {value}" for key, value in details.items()],
        ]
        for name, table in tables.items():
            sections.extend(["", f"### {name}", dataframe_to_markdown(table)])
        # 原因：回归和 ANOVA 都包含多张互补结果表，单一摘要会丢失检验依据。
        # 作用：把整体统计、系数或组统计、ANOVA 和事后比较完整交给 Agent。
        return SkillResponse(
            success=True,
            content="\n".join(sections),
            data={
                "file_path": str(path),
                "table_name": table_name,
                "method": method,
                "details": details,
                "tables": {
                    name: table.to_dict(orient="records")
                    for name, table in tables.items()
                },
            },
        )


def _linear_regression(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    predictor_columns: list[str],
    confidence_level: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit one OLS model with numeric or categorical predictors."""
    if not predictor_columns:
        raise ValueError("linear_regression requires predictor_columns.")
    if outcome_column in predictor_columns:
        raise ValueError("outcome_column cannot also be a predictor.")
    required = [outcome_column, *predictor_columns]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise KeyError(", ".join(missing))

    model_data = dataframe[required].dropna().copy()
    outcome = pd.to_numeric(model_data[outcome_column], errors="coerce")
    model_data = model_data.loc[outcome.notna()]
    outcome = outcome.loc[outcome.notna()].astype(float)
    design_parts: list[pd.DataFrame] = []
    for column in predictor_columns:
        raw = model_data[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            design_parts.append(pd.DataFrame({column: numeric.astype(float)}))
            continue
        text = raw.astype("string")
        categories = list(dict.fromkeys(text))
        encoded = pd.get_dummies(
            pd.Categorical(text, categories=categories),
            prefix=column,
            drop_first=True,
            dtype=float,
        )
        # 原因：Categorical 编码会生成新的 RangeIndex，而 Excel 数据帧保留工作表行号。
        # 作用：恢复原索引后再与数值预测量按行拼接，避免错位产生 NaN 并使 OLS 失败。
        encoded.index = raw.index
        if encoded.empty:
            raise ValueError(f"categorical predictor {column} has fewer than two levels.")
        design_parts.append(encoded)
    design = sm.add_constant(pd.concat(design_parts, axis=1), has_constant="add")
    if len(outcome) <= design.shape[1]:
        raise ValueError("linear_regression needs more complete rows than coefficients.")
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
        raise ValueError("linear_regression predictors are perfectly collinear.")

    fitted = sm.OLS(outcome, design).fit()
    intervals = fitted.conf_int(alpha=1.0 - confidence_level)
    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.to_numpy(),
            "standard_error": fitted.bse.to_numpy(),
            "t_statistic": fitted.tvalues.to_numpy(),
            "p_value": fitted.pvalues.to_numpy(),
            "ci_lower": intervals.iloc[:, 0].to_numpy(),
            "ci_upper": intervals.iloc[:, 1].to_numpy(),
        }
    ).round(6)
    residuals = pd.Series(fitted.resid)
    quantiles = residuals.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    model_summary = pd.DataFrame(
        [
            {
                "observations": int(fitted.nobs),
                "df_model": float(fitted.df_model),
                "df_residual": float(fitted.df_resid),
                "r_squared": float(fitted.rsquared),
                "adjusted_r_squared": float(fitted.rsquared_adj),
                "residual_standard_error": float(fitted.mse_resid**0.5),
                "f_statistic": float(fitted.fvalue),
                "f_p_value": float(fitted.f_pvalue),
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
            }
        ]
    ).round(6)
    residual_summary = pd.DataFrame(
        [
            {
                "minimum": float(quantiles.loc[0.0]),
                "q1": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "q3": float(quantiles.loc[0.75]),
                "maximum": float(quantiles.loc[1.0]),
            }
        ]
    ).round(6)
    return {
        "Model summary": model_summary,
        "Coefficients": coefficients,
        "Residual summary": residual_summary,
    }, {
        "outcome": outcome_column,
        "predictors": ", ".join(predictor_columns),
        "confidence_level": confidence_level,
        "method": "ordinary least squares with an intercept",
        "categorical_rule": "first observed level is the reference category",
    }


def _one_way_anova(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    group_column: str,
    confidence_level: float,
    include_posthoc: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Test one numeric outcome across independent categorical groups."""
    if not group_column:
        raise ValueError("one_way_anova requires group_column.")
    missing = [
        column
        for column in [outcome_column, group_column]
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(", ".join(missing))
    clean = dataframe[[outcome_column, group_column]].dropna().copy()
    clean[outcome_column] = pd.to_numeric(clean[outcome_column], errors="coerce")
    clean = clean.dropna(subset=[outcome_column])
    grouped = list(clean.groupby(group_column, sort=False))
    if not 2 <= len(grouped) <= 20:
        raise ValueError("one_way_anova requires between 2 and 20 non-empty groups.")
    samples = [group[outcome_column].astype(float) for _, group in grouped]
    if any(len(sample) < 2 for sample in samples):
        raise ValueError("one_way_anova requires at least two observations per group.")

    total_count = sum(len(sample) for sample in samples)
    grand_mean = float(clean[outcome_column].mean())
    sum_squares_between = sum(
        len(sample) * (float(sample.mean()) - grand_mean) ** 2
        for sample in samples
    )
    sum_squares_within = sum(
        float(((sample - sample.mean()) ** 2).sum())
        for sample in samples
    )
    sum_squares_total = sum_squares_between + sum_squares_within
    df_between = len(samples) - 1
    df_within = total_count - len(samples)
    mean_square_between = sum_squares_between / df_between
    mean_square_within = sum_squares_within / df_within
    if mean_square_within == 0 or sum_squares_total == 0:
        raise ValueError(
            "one_way_anova is undefined when the outcome has no within-group variance."
        )
    f_statistic = mean_square_between / mean_square_within
    p_value = float(stats.f.sf(f_statistic, df_between, df_within))
    eta_squared = sum_squares_between / sum_squares_total
    omega_squared = (
        (sum_squares_between - df_between * mean_square_within)
        / (sum_squares_total + mean_square_within)
    )
    levene = stats.levene(*[sample.to_numpy() for sample in samples], center="median")

    group_rows = []
    for group_name, group in grouped:
        sample = group[outcome_column].astype(float)
        count = len(sample)
        standard_deviation = float(sample.std(ddof=1))
        standard_error = standard_deviation / count**0.5
        t_critical = float(
            stats.t.ppf((1.0 + confidence_level) / 2.0, count - 1)
        )
        group_rows.append(
            {
                "group": str(group_name),
                "count": count,
                "mean": float(sample.mean()),
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "ci_lower": float(sample.mean()) - t_critical * standard_error,
                "ci_upper": float(sample.mean()) + t_critical * standard_error,
            }
        )
    anova_table = pd.DataFrame(
        [
            {
                "source": "between_groups",
                "sum_squares": sum_squares_between,
                "degrees_of_freedom": df_between,
                "mean_square": mean_square_between,
                "f_statistic": f_statistic,
                "p_value": p_value,
            },
            {
                "source": "within_groups",
                "sum_squares": sum_squares_within,
                "degrees_of_freedom": df_within,
                "mean_square": mean_square_within,
                "f_statistic": float("nan"),
                "p_value": float("nan"),
            },
            {
                "source": "total",
                "sum_squares": sum_squares_total,
                "degrees_of_freedom": total_count - 1,
                "mean_square": float("nan"),
                "f_statistic": float("nan"),
                "p_value": float("nan"),
            },
        ]
    ).round(6)
    tables = {
        "Group summary": pd.DataFrame(group_rows).round(6),
        "ANOVA": anova_table,
    }
    if include_posthoc and len(samples) >= 3:
        tukey = stats.tukey_hsd(*[sample.to_numpy() for sample in samples])
        interval = tukey.confidence_interval(confidence_level)
        posthoc_rows = []
        for first_index, second_index in combinations(range(len(grouped)), 2):
            posthoc_rows.append(
                {
                    "group_1": str(grouped[first_index][0]),
                    "group_2": str(grouped[second_index][0]),
                    "mean_difference": float(
                        tukey.statistic[first_index, second_index]
                    ),
                    "ci_lower": float(interval.low[first_index, second_index]),
                    "ci_upper": float(interval.high[first_index, second_index]),
                    "adjusted_p_value": float(
                        tukey.pvalue[first_index, second_index]
                    ),
                    "significant": bool(
                        tukey.pvalue[first_index, second_index]
                        < 1.0 - confidence_level
                    ),
                }
            )
        tables["Tukey HSD"] = pd.DataFrame(posthoc_rows).round(6)
    # 原因：ANOVA 显著只说明至少一个均值不同，不能指出具体差异或实际影响大小。
    # 作用：同时返回 eta²/omega²、方差齐性检查和可选 Tukey，防止 Agent 过度解释 p 值。
    return tables, {
        "outcome": outcome_column,
        "group": group_column,
        "confidence_level": confidence_level,
        "eta_squared": round(eta_squared, 6),
        "omega_squared": round(omega_squared, 6),
        "levene_statistic": round(float(levene.statistic), 6),
        "levene_p_value": round(float(levene.pvalue), 6),
        "assumptions": (
            "independent observations, approximately normal residuals, "
            "and similar within-group variances"
        ),
    }


def _column_list(value: Any) -> list[str]:
    """Normalize a JSON array into unique non-empty column names."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("predictor_columns must be a JSON array of strings.")
    return list(
        dict.fromkeys(
            str(column).strip()
            for column in value
            if str(column).strip()
        )
    )


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return ExcelModelingSkill()
