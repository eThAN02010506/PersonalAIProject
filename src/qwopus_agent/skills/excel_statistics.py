"""Deterministic statistical methods for spreadsheet analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from scipy import stats

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.analysis.markdown_tables import dataframe_to_markdown
from qwopus_agent.skills.base import BaseSkill, SkillRequest, SkillResponse

SUPPORTED_METHODS = {
    "chi_square_independence",
    "correlation",
    "covariance",
    "crosstab",
    "date_extract",
    "deduplicate",
    "describe",
    "frequency",
    "group_summary",
    "iqr_outliers",
    "kruskal_wallis",
    "lookup",
    "mann_whitney_u",
    "mean_confidence_interval",
    "missing",
    "normality_test",
    "one_sample_t_test",
    "pivot",
    "quantiles",
    "rank",
    "two_sample_t_test",
    "wilcoxon_signed_rank",
    "zscore_outliers",
}
CROSSTAB_MAX_DISPLAY_CATEGORIES = 20


@dataclass
class ExcelStatisticsSkill(BaseSkill):
    """Run reusable statistics without asking the model to generate pandas code."""

    agent_tool_permission: ClassVar[str | None] = "documents"
    name: str = "excel_statistics"
    description: str = (
        "Run deterministic spreadsheet statistics: describe, missing, IQR outliers, "
        "Z-score outliers, frequency tables, grouped summaries, correlations, "
        "covariance, quantiles, normality checks, crosstabs, chi-square tests, "
        "lookup, confidence intervals, t-tests, or non-parametric tests. Also "
        "handles data-shaping tasks: pivot, date_extract, deduplicate, and rank. "
        "Prefer this skill for common statistical questions; use excel_analysis "
        "only for custom computations."
    )

    async def run(self, request: SkillRequest) -> SkillResponse:
        """Validate one approved workbook request and return a computed Markdown table."""
        try:
            path = Path(str(request.arguments["file_path"]))
            table_name = str(request.arguments["table_name"]).strip()
            method = str(request.arguments["method"]).strip()
            value_columns = _column_list(request.arguments.get("value_columns"))
            label_columns = _column_list(request.arguments.get("label_columns"))
            group_column = _optional_text_argument(
                request.arguments.get("group_column")
            )
            group_values = _column_list(request.arguments.get("group_values"))
            category_columns = _column_list(request.arguments.get("category_columns"))
            lookup_value = _optional_text_argument(
                request.arguments.get("lookup_value")
            )
            confidence_level_argument = request.arguments.get("confidence_level")
            confidence_level = (
                0.95
                if confidence_level_argument is None
                else float(confidence_level_argument)
            )
            hypothesized_mean_argument = request.arguments.get("hypothesized_mean")
            hypothesized_mean = (
                float(hypothesized_mean_argument)
                if hypothesized_mean_argument is not None
                else None
            )
            scope_table_name = _optional_text_argument(
                request.arguments.get("scope_table_name")
            )
            scope_data_key = _optional_text_argument(
                request.arguments.get("scope_data_key")
            )
            scope_lookup_key = _optional_text_argument(
                request.arguments.get("scope_lookup_key")
            )
            scope_required_columns = _column_list(
                request.arguments.get("scope_required_columns")
            )
            row_column = _optional_text_argument(
                request.arguments.get("row_column")
            )
            column_column = _optional_text_argument(
                request.arguments.get("column_column")
            )
            value_column = _optional_text_argument(
                request.arguments.get("value_column")
            )
            agg = _optional_text_argument(request.arguments.get("agg")) or "sum"
            date_column = _optional_text_argument(
                request.arguments.get("date_column")
            )
            keep = _optional_text_argument(request.arguments.get("keep")) or "first"
            rank_method = (
                _optional_text_argument(request.arguments.get("rank_method"))
                or "rank"
            )
            parts = _column_list(request.arguments.get("parts"))
            columns = _column_list(request.arguments.get("columns"))
            into_bins_argument = request.arguments.get("into_bins")
            into_bins = (
                4
                if into_bins_argument is None
                else int(into_bins_argument)
            )
            top_n = max(1, min(int(request.arguments.get("top_n") or 20), 100))
            threshold_argument = request.arguments.get("threshold")
            threshold = float(
                threshold_argument
                if threshold_argument is not None
                else (3.0 if method == "zscore_outliers" else 1.5)
            )
        except (KeyError, TypeError, ValueError) as exc:
            return SkillResponse(
                success=False,
                content=f"Invalid excel_statistics arguments: {exc}",
            )

        if method not in SUPPORTED_METHODS:
            return SkillResponse(
                success=False,
                content=(
                    f"Unsupported statistical method: {method}. "
                    f"Choose one of: {', '.join(sorted(SUPPORTED_METHODS))}."
                ),
            )
        if not path.is_file():
            return SkillResponse(
                success=False,
                content=f"Spreadsheet file does not exist: {path}",
            )

        try:
            frames = read_spreadsheet(path).analysis_frames()
            dataframe = frames[table_name]
            auto_scope = False
            if (
                method in {"iqr_outliers", "zscore_outliers"}
                and not scope_table_name
                and (
                    scope_candidate := _comparison_scope_candidate(
                        frames,
                        table_name=table_name,
                        dataframe=dataframe,
                        label_columns=label_columns,
                    )
                )
            ):
                scope_table_name = scope_candidate["scope_table_name"]
                scope_data_key = scope_candidate["scope_data_key"]
                scope_lookup_key = scope_candidate["scope_lookup_key"]
                scope_required_columns = scope_candidate["scope_required_columns"]
                auto_scope = True
            dataframe, scope_details = _apply_comparison_scope(
                frames,
                dataframe,
                scope_table_name=scope_table_name,
                scope_data_key=scope_data_key,
                scope_lookup_key=scope_lookup_key,
                scope_required_columns=scope_required_columns,
            )
            if auto_scope:
                scope_details = {
                    "scope_mode": "auto-detected from workbook classification metadata",
                    **scope_details,
                }
            result, details = _run_statistical_method(
                dataframe,
                method=method,
                value_columns=value_columns,
                label_columns=label_columns,
                group_column=group_column,
                group_values=group_values,
                category_columns=category_columns,
                lookup_value=lookup_value,
                top_n=top_n,
                threshold=threshold,
                confidence_level=confidence_level,
                hypothesized_mean=hypothesized_mean,
                row_column=row_column,
                column_column=column_column,
                value_column=value_column,
                agg=agg,
                date_column=date_column,
                parts=parts,
                columns=columns,
                keep=keep,
                rank_method=rank_method,
                into_bins=into_bins,
            )
            details = {**scope_details, **details}
        except KeyError as exc:
            available = ", ".join(frames) if "frames" in locals() else ""
            return SkillResponse(
                success=False,
                content=f"Unknown table or column: {exc}. Available tables: {available}.",
            )
        except (TypeError, ValueError) as exc:
            return SkillResponse(success=False, content=f"Statistics failed: {exc}")

        # 原因：最终回答必须能核对真实本地计算，不能只返回模型对数值的转述。
        # 作用：把方法、口径和有界结果表作为一个完整 Observation 交给 Agent。
        content = "\n".join(
            [
                f"## Statistical result: {method}",
                f"- Table: {table_name}",
                f"- Value columns: {', '.join(value_columns) if value_columns else '<auto/none>'}",
                *[f"- {key}: {value}" for key, value in details.items()],
                "",
                dataframe_to_markdown(result.head(top_n)),
            ]
        )
        return SkillResponse(
            success=True,
            content=content,
            data={
                "file_path": str(path),
                "table_name": table_name,
                "method": method,
                "details": details,
                "rows": result.to_dict(orient="records"),
            },
        )


def _run_statistical_method(
    dataframe: pd.DataFrame,
    *,
    method: str,
    value_columns: list[str],
    label_columns: list[str],
    group_column: str,
    group_values: list[str],
    category_columns: list[str],
    lookup_value: str,
    top_n: int,
    threshold: float,
    confidence_level: float,
    hypothesized_mean: float | None,
    row_column: str,
    column_column: str,
    value_column: str,
    agg: str,
    date_column: str,
    parts: list[str],
    columns: list[str],
    keep: str,
    rank_method: str,
    into_bins: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch one validated common method against an in-memory dataframe."""
    if method == "lookup":
        return _lookup_rows(dataframe, lookup_value=lookup_value, top_n=top_n)

    # 原因：pivot/date_extract/deduplicate/rank 用各自的参数（不一定有 value_columns），
    # 必须先于下方 `if not value_columns:` 的自动选取，否则因无 value_columns 提前报错。
    # 作用：把数据整理方法从“按数值列统计”的公共路径中分离。
    if method == "pivot":
        return _pivot_table(
            dataframe,
            row_column=row_column,
            column_column=column_column,
            value_column=value_column,
            agg=agg,
            top_n=top_n,
        )
    if method == "date_extract":
        return _date_extract(
            dataframe,
            date_column=date_column,
            parts=parts,
            top_n=top_n,
        )
    if method == "deduplicate":
        return _deduplicate_rows(
            dataframe,
            columns=columns,
            keep=keep,
            top_n=top_n,
        )
    if method == "rank":
        return _rank_rows(
            dataframe,
            value_column=value_column,
            rank_method=rank_method,
            into_bins=into_bins,
            label_columns=label_columns,
            top_n=top_n,
        )

    if method in {"crosstab", "chi_square_independence"}:
        return _categorical_association(
            dataframe,
            method=method,
            category_columns=category_columns,
            top_n=top_n,
        )

    if not value_columns:
        # 原因：describe 和 frequency 是通用 profile 方法，用户不指定列时应自动选取。
        # 作用：describe 自动选所有数值列，frequency 自动选所有非数值列。
        if method in {"describe", "quantiles", "normality_test"}:
            value_columns = [
                str(c) for c in dataframe.columns
                if pd.api.types.is_numeric_dtype(dataframe[c])
            ]
        elif method == "frequency":
            value_columns = [
                str(c) for c in dataframe.columns
                if not pd.api.types.is_numeric_dtype(dataframe[c])
            ]
        if not value_columns:
            raise ValueError("value_columns must contain at least one column.")
    requested_columns = [*label_columns, *value_columns]
    if group_column:
        requested_columns.append(group_column)
    missing_columns = [
        column for column in dict.fromkeys(requested_columns)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(", ".join(missing_columns))
    _validate_comparable_units(value_columns, method)

    if method == "missing":
        missing = dataframe[value_columns].isna().sum()
        result = pd.DataFrame(
            {
                "column": value_columns,
                "missing": [int(missing[column]) for column in value_columns],
                "missing_percent": [
                    round(float(missing[column]) / max(len(dataframe), 1) * 100, 3)
                    for column in value_columns
                ],
            }
        )
        return result, {"row_count": int(len(dataframe))}

    if method == "frequency":
        rows: list[dict[str, Any]] = []
        for column in value_columns:
            displayed = dataframe[column].astype("string").fillna("<missing>")
            counts = displayed.value_counts(dropna=False)
            rows.extend(
                {
                    "column": column,
                    "value": str(value),
                    "count": int(count),
                    "percent": round(float(count) / max(len(dataframe), 1) * 100, 4),
                }
                for value, count in counts.items()
            )
        result = pd.DataFrame(rows)
        return result.head(top_n), {
            "row_count": int(len(dataframe)),
            "rule": "R table()-style counts and percentages, including missing values",
        }

    values = dataframe[value_columns].apply(pd.to_numeric, errors="coerce")
    if values.notna().sum().sum() == 0:
        raise ValueError("selected value columns contain no numeric observations.")

    if method == "describe":
        rows = []
        for column in value_columns:
            sample = values[column].dropna()
            count = int(len(sample))
            standard_deviation = float(sample.std(ddof=1)) if count >= 2 else float("nan")
            mean = float(sample.mean()) if count else float("nan")
            quartiles = sample.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "column": column,
                    "count": count,
                    "missing": int(values[column].isna().sum()),
                    "missing_percent": (
                        float(values[column].isna().sum()) / max(len(values), 1) * 100
                    ),
                    "sum": float(sample.sum()),
                    "mean": mean,
                    "median": float(quartiles.loc[0.5]) if count else float("nan"),
                    "standard_deviation": standard_deviation,
                    "variance": (
                        float(sample.var(ddof=1)) if count >= 2 else float("nan")
                    ),
                    "standard_error": (
                        standard_deviation / count**0.5
                        if count >= 2
                        else float("nan")
                    ),
                    "minimum": float(sample.min()) if count else float("nan"),
                    "q1": float(quartiles.loc[0.25]) if count else float("nan"),
                    "q3": float(quartiles.loc[0.75]) if count else float("nan"),
                    "maximum": float(sample.max()) if count else float("nan"),
                    "range": (
                        float(sample.max() - sample.min())
                        if count
                        else float("nan")
                    ),
                    "iqr": (
                        float(quartiles.loc[0.75] - quartiles.loc[0.25])
                        if count
                        else float("nan")
                    ),
                    "coefficient_of_variation_percent": (
                        standard_deviation / abs(mean) * 100
                        if count >= 2 and mean != 0
                        else float("nan")
                    ),
                    "skewness": float(sample.skew()) if count >= 3 else float("nan"),
                    "excess_kurtosis": (
                        float(sample.kurt()) if count >= 4 else float("nan")
                    ),
                    "unique": int(sample.nunique()),
                }
            )
        return pd.DataFrame(rows).round(6), {
            "method": "R summary()-style numeric profile with dispersion and shape",
            "unit_rule": "Each column is summarized separately.",
        }

    if method == "quantiles":
        return _quantile_summary(values), {
            "method": "selected percentile summary",
            "unit_rule": "Each column is summarized separately.",
        }

    if method == "correlation":
        result = values.corr().reset_index(names="column")
        return result.round(4), {"method": "Pearson correlation"}

    if method == "covariance":
        result = values.cov().reset_index(names="column")
        return result.round(6), {"method": "sample covariance matrix"}

    if method == "normality_test":
        return _normality_tests(values), {
            "method": "D'Agostino-Pearson normality test; Shapiro-Wilk for small samples",
        }

    if method == "group_summary":
        if not group_column:
            raise ValueError("group_summary requires group_column.")
        grouped = pd.concat([dataframe[[group_column]], values], axis=1)
        result = grouped.groupby(group_column, dropna=False)[value_columns].agg(
            ["count", "mean", "median", "min", "max"]
        )
        result.columns = [
            f"{column}_{statistic}" for column, statistic in result.columns
        ]
        return result.reset_index().round(4), {"group_column": group_column}

    if method == "mean_confidence_interval":
        result = _mean_confidence_intervals(
            values,
            confidence_level=confidence_level,
        )
        return result, {
            "method": "Student-t interval for a population mean",
            "confidence_level": confidence_level,
        }

    if method == "one_sample_t_test":
        if hypothesized_mean is None:
            raise ValueError("one_sample_t_test requires hypothesized_mean.")
        result = _one_sample_t_tests(
            values,
            hypothesized_mean=hypothesized_mean,
            confidence_level=confidence_level,
        )
        return result, {
            "method": "two-sided one-sample Student t-test",
            "null_hypothesis": f"population mean = {hypothesized_mean}",
            "confidence_level": confidence_level,
        }

    if method == "two_sample_t_test":
        if len(value_columns) != 1:
            raise ValueError("two_sample_t_test requires exactly one value column.")
        if not group_column:
            raise ValueError("two_sample_t_test requires group_column.")
        result = _two_sample_t_test(
            dataframe,
            value_column=value_columns[0],
            group_column=group_column,
            group_values=group_values,
            confidence_level=confidence_level,
        )
        return result, {
            "method": "two-sided Welch t-test",
            "confidence_level": confidence_level,
        }

    if method == "mann_whitney_u":
        if len(value_columns) != 1:
            raise ValueError("mann_whitney_u requires exactly one value column.")
        if not group_column:
            raise ValueError("mann_whitney_u requires group_column.")
        result = _mann_whitney_u(
            dataframe,
            value_column=value_columns[0],
            group_column=group_column,
            group_values=group_values,
        )
        return result, {"method": "Mann-Whitney U (independent samples)"}

    if method == "wilcoxon_signed_rank":
        if len(value_columns) != 2:
            raise ValueError(
                "wilcoxon_signed_rank requires exactly two value columns (before/after)."
            )
        result = _wilcoxon_signed_rank(
            dataframe,
            before_column=value_columns[0],
            after_column=value_columns[1],
        )
        return result, {"method": "Wilcoxon signed-rank (paired)", "paired": True}

    if method == "kruskal_wallis":
        if len(value_columns) != 1:
            raise ValueError("kruskal_wallis requires exactly one value column.")
        if not group_column:
            raise ValueError("kruskal_wallis requires group_column.")
        result = _kruskal_wallis(
            dataframe,
            value_column=value_columns[0],
            group_column=group_column,
        )
        return result, {"method": "Kruskal-Wallis H test"}

    series, metric_name = _row_metric(values)
    labels = (
        dataframe[label_columns].copy()
        if label_columns
        else pd.DataFrame({"row_index": dataframe.index}, index=dataframe.index)
    )
    labels["metric"] = series

    if method == "iqr_outliers":
        if threshold <= 0:
            raise ValueError("IQR threshold must be greater than zero.")
        quartiles = series.quantile([0.25, 0.75])
        first_quartile = float(quartiles.iloc[0])
        third_quartile = float(quartiles.iloc[1])
        spread = third_quartile - first_quartile
        lower = first_quartile - threshold * spread
        upper = third_quartile + threshold * spread
        mask = (series < lower) | (series > upper)
        result = labels[mask].copy()
        result["direction"] = result["metric"].map(
            lambda value: "low" if value < lower else "high"
        )
        result["lower_bound"] = lower
        result["upper_bound"] = upper
        result = result.sort_values("metric", ascending=False)
        if result.empty:
            # 原因：空 DataFrame 会被 Markdown 渲染成纯文本，最终回答阶段无法作为表格展示。
            # 作用：即使没有异常值，也保留 IQR 边界和 outlier_count 供用户核对。
            result = pd.DataFrame(
                [
                    {
                        "metric": metric_name,
                        "rule": f"{threshold} x IQR",
                        "outlier_count": 0,
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                ]
            )
        return result.round(4), {
            "metric": metric_name,
            "rule": f"{threshold} x IQR",
            "q1": round(first_quartile, 4),
            "q3": round(third_quartile, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": int(mask.sum()),
        }

    if threshold <= 0:
        raise ValueError("Z-score threshold must be greater than zero.")
    mean = float(series.mean())
    standard_deviation = float(series.std(ddof=0))
    if standard_deviation == 0:
        result = labels.iloc[0:0].copy()
        result["z_score"] = pd.Series(dtype=float)
    else:
        z_scores = (series - mean) / standard_deviation
        result = labels[z_scores.abs() >= threshold].copy()
        result["z_score"] = z_scores[z_scores.abs() >= threshold]
        result = result.sort_values(
            "z_score",
            key=lambda values: values.abs(),
            ascending=False,
        )
    if result.empty:
        # 原因：空异常值结果仍是有意义的统计结论，不能在 Chat 表格展示中消失。
        # 作用：把“没有发现 Z-score 异常值”和计算口径作为可见表格返回。
        result = pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "rule": f"absolute Z-score >= {threshold}",
                    "outlier_count": 0,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                }
            ]
        )
    return result.round(4), {
        "metric": metric_name,
        "rule": f"absolute Z-score >= {threshold}",
        "mean": round(mean, 4),
        "standard_deviation": round(standard_deviation, 4),
        "outlier_count": int(len(result)),
    }


def _lookup_rows(
    dataframe: pd.DataFrame,
    *,
    lookup_value: str,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find rows that contain one requested item label or value."""
    if not lookup_value:
        raise ValueError("lookup requires lookup_value.")
    normalized_query = lookup_value.casefold()
    if len(dataframe.columns) == 2:
        columns = [str(column) for column in dataframe.columns]
        for index, column in enumerate(columns):
            if column.casefold() != normalized_query:
                continue
            # 原因：部分表单型 Excel 会把第一项数据误识别为表头，例如 STR | 40。
            # 作用：单项查询仍能返回字段和值，而不被迫把 CON | 55 当作 STR 的答案。
            value_column = columns[1 - index]
            return pd.DataFrame(
                [
                    {
                        "key": column,
                        "value": value_column,
                        "match_source": "column_header_pair",
                    }
                ]
            ), {
                "lookup_value": lookup_value,
                "match_rule": "two-column header pair fallback before row search",
                "match_count": 1,
            }
    searchable = dataframe.astype("string").fillna("")
    exact_mask = searchable.map(lambda value: value.casefold() == normalized_query).any(axis=1)
    contains_mask = searchable.map(lambda value: normalized_query in value.casefold()).any(axis=1)
    mask = exact_mask if bool(exact_mask.any()) else contains_mask
    matches = dataframe.loc[mask].head(top_n).copy()
    if matches.empty:
        matches = pd.DataFrame(
            [{"lookup_value": lookup_value, "match": "no matching rows found"}]
        )
    else:
        # 原因：单项查询需要能回到 Excel 的原始行，尤其是 key_values 这种派生帧。
        # 作用：结果表保留 dataframe 行号，便于 Agent 解释“命中了哪一项”。
        matches.insert(0, "row_index", matches.index)
    return matches.reset_index(drop=True), {
        "lookup_value": lookup_value,
        "match_rule": "case-insensitive exact row match, then contains fallback",
        "match_count": int(mask.sum()),
    }


def _pivot_table(
    dataframe: pd.DataFrame,
    *,
    row_column: str,
    column_column: str,
    value_column: str,
    agg: str,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reshape one value column into a pivot table with a numeric aggregation."""
    if not row_column or not column_column or not value_column:
        raise ValueError("pivot requires row_column, column_column, and value_column.")
    missing = [
        column
        for column in [row_column, column_column, value_column]
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(", ".join(missing))
    allowed_aggs = {"sum", "mean", "count", "min", "max"}
    if agg not in allowed_aggs:
        raise ValueError(
            f"pivot agg must be one of: {', '.join(sorted(allowed_aggs))}."
        )
    work = dataframe[[row_column, column_column]].copy()
    if agg == "count":
        # 原因：count 语义是“该组合有多少个非空单元”，非数值文本（如 "N/A"）也是有效计数。
        # 作用：count 用原始值列，不做 to_numeric 强转，避免把文本单元静默当成缺失。
        work[value_column] = dataframe[value_column]
    else:
        work[value_column] = pd.to_numeric(
            dataframe[value_column], errors="coerce"
        )
    # 原因：aggfunc 只接受 pandas 白名单字符串，但 stubs 把类型标成 Literal 联合。
    # 作用：值已在 allowed_aggs 校验过，这里仅绕过 mypy 的 arg-type 误报。
    table = pd.pivot_table(
        work,
        index=row_column,
        columns=column_column,
        values=value_column,
        aggfunc=agg,  # type: ignore[arg-type]
    )
    # 原因：pivot 可能产生高基数列导致表格宽度爆炸。
    # 作用：按最常见的列类别截断，完整类别数留在 details 供 Agent 说明口径。
    column_order = (
        table.notna().sum().sort_values(ascending=False).index[:top_n]
    )
    result = table.loc[:, column_order].reset_index()
    return result.round(6), {
        "row_column": row_column,
        "column_column": column_column,
        "value_column": value_column,
        "agg": agg,
        "pivot_rows": int(table.shape[0]),
        "pivot_columns": int(table.shape[1]),
        "displayed_columns": int(result.shape[1]) - 1,
    }


def _date_extract(
    dataframe: pd.DataFrame,
    *,
    date_column: str,
    parts: list[str],
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract date components (year, month, quarter, weekday) into new columns."""
    if not date_column:
        raise ValueError("date_extract requires date_column.")
    if date_column not in dataframe.columns:
        raise KeyError(date_column)
    parsed = pd.to_datetime(dataframe[date_column], errors="coerce")
    parsed_count = int(parsed.notna().sum())
    if parsed_count == 0:
        raise ValueError(
            f"date_extract could not parse any value in column {date_column}."
        )
    valid_parts = {"year", "month", "quarter", "weekday"}
    selected = parts or ["year", "month"]
    unknown = [part for part in selected if part not in valid_parts]
    if unknown:
        raise ValueError(
            f"date_extract parts must be in: {', '.join(sorted(valid_parts))}."
        )
    result = dataframe[[date_column]].copy()
    year = parsed.dt.year
    for part in selected:
        if part == "year":
            result[f"{date_column}_year"] = year
        elif part == "month":
            result[f"{date_column}_month"] = parsed.dt.month
        elif part == "quarter":
            result[f"{date_column}_quarter"] = parsed.dt.quarter
        else:
            result[f"{date_column}_weekday"] = parsed.dt.dayofweek
    result = result[parsed.notna()].head(top_n).reset_index(drop=True)
    min_year = int(year.min())
    max_year = int(year.max())
    return result, {
        "date_column": date_column,
        "parts": ", ".join(selected),
        "parsed_count": parsed_count,
        "unparsed_count": int(parsed.isna().sum()),
        "min_year": min_year,
        "max_year": max_year,
    }


def _deduplicate_rows(
    dataframe: pd.DataFrame,
    *,
    columns: list[str],
    keep: str,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop duplicate rows on a column subset, keeping the first or last row."""
    if keep not in {"first", "last"}:
        raise ValueError("deduplicate keep must be 'first' or 'last'.")
    subset = columns or list(dataframe.columns)
    missing = [column for column in subset if column not in dataframe.columns]
    if missing:
        raise KeyError(", ".join(missing))
    # 原因：keep 已在上面校验为 "first"/"last"，pandas stubs 只认 Literal。
    # 作用：运行时语义正确，仅绕过 mypy 的 call-overload 误报。
    dropped = dataframe.drop_duplicates(  # type: ignore[call-overload]
        subset=subset, keep=keep
    )
    result = dropped.head(top_n).reset_index(drop=True)
    return result, {
        "subset": ", ".join(subset),
        "keep": keep,
        "total_rows": int(len(dataframe)),
        "kept_rows": int(len(dropped)),
        "dropped_count": int(len(dataframe) - len(dropped)),
    }


def _rank_rows(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    rank_method: str,
    into_bins: int,
    label_columns: list[str],
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank one numeric column, optionally binning it into ranked groups."""
    if not value_column:
        raise ValueError("rank requires value_column.")
    if value_column not in dataframe.columns:
        raise KeyError(value_column)
    values = pd.to_numeric(dataframe[value_column], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"rank found no numeric observations in {value_column}.")
    base_columns = list(label_columns or []) + [value_column]
    missing = [
        column for column in base_columns if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(", ".join(missing))
    result = dataframe[base_columns].copy()
    if rank_method == "ntile":
        if into_bins <= 0:
            raise ValueError("rank into_bins must be greater than zero.")
        # 原因：qcut 需要至少两行有效数据且组数不超过行数。
        # 作用：数据不足时明确失败，由调用方解释，而不是产生 NaN 桶。
        valid_count = int(values.notna().sum())
        if valid_count < 2:
            raise ValueError("rank ntile requires at least two numeric observations.")
        bin_count = min(into_bins, valid_count)
        bins = pd.qcut(values, q=bin_count, labels=False, duplicates="drop")
        actual_bins = int(bins.nunique())
        if actual_bins < 2:
            # 原因：全相等数据会让 qcut 折叠到单一边界并返回全 NaN。
            # 作用：退化为单一桶（全部 rank 1），而不是输出不可读的 NaN 排名。
            result["rank_ntile"] = 1
            actual_bins = 1
        else:
            result["rank_ntile"] = bins + 1
        details: dict[str, Any] = {
            "value_column": value_column,
            "rank_method": "ntile",
            "into_bins": actual_bins,
            "total_rows": int(len(dataframe)),
        }
    else:
        if rank_method not in {"rank", "average", "dense"}:
            raise ValueError(
                "rank rank_method must be one of: rank, average, dense, ntile."
            )
        # 原因：pandas Series.rank 没有 "rank" 方法值；"rank" 是 R 语义的
        # 平均秩（平局取均值），对应 pandas 的 "average"。
        # 作用：把 R 风格默认值映射到 pandas 合法方法，保持工具契约稳定。
        pandas_method = "average" if rank_method == "rank" else rank_method
        result["rank"] = values.rank(
            method=pandas_method,  # type: ignore[arg-type]
            na_option="keep",
        )
        details = {
            "value_column": value_column,
            "rank_method": rank_method,
            "total_rows": int(len(dataframe)),
        }
    return result.head(top_n).reset_index(drop=True), details


def _quantile_summary(values: pd.DataFrame) -> pd.DataFrame:
    """Return common percentiles for each numeric column."""
    percentiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows: list[dict[str, Any]] = []
    for column in map(str, values.columns):
        sample = values[column].dropna()
        if sample.empty:
            continue
        quantiles = sample.quantile(percentiles)
        row: dict[str, Any] = {"column": column, "count": int(len(sample))}
        row.update(
            {
                f"p{int(percentile * 100):02d}": float(quantiles.loc[percentile])
                for percentile in percentiles
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("quantiles requires at least one numeric observation.")
    return pd.DataFrame(rows).round(6)


def _normality_tests(values: pd.DataFrame) -> pd.DataFrame:
    """Run a conservative normality test per numeric column."""
    rows: list[dict[str, Any]] = []
    for column in map(str, values.columns):
        sample = values[column].dropna()
        count = int(len(sample))
        if count < 3:
            raise ValueError(
                f"normality_test requires at least three observations in {column}."
            )
        if count >= 8:
            test = stats.normaltest(sample.to_numpy(), nan_policy="omit")
            test_name = "dagostino_pearson"
        else:
            test = stats.shapiro(sample.to_numpy())
            test_name = "shapiro_wilk"
        p_value = float(test.pvalue)
        # 原因：正态性检验的 p 值经常被误读为“正态概率”。
        # 作用：把统计判读放进结果表，降低弱模型解释错误的概率。
        rows.append(
            {
                "column": column,
                "count": count,
                "test": test_name,
                "statistic": float(test.statistic),
                "p_value": p_value,
                "decision_at_0.05": (
                    "reject normality" if p_value < 0.05 else "do not reject normality"
                ),
            }
        )
    return pd.DataFrame(rows).round(6)


def _categorical_association(
    dataframe: pd.DataFrame,
    *,
    method: str,
    category_columns: list[str],
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a crosstab or chi-square test for two categorical columns."""
    if len(category_columns) != 2:
        raise ValueError(f"{method} requires exactly two category_columns.")
    missing_columns = [
        column for column in category_columns
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(", ".join(missing_columns))
    first, second = category_columns
    table = pd.crosstab(
        dataframe[first].astype("string").fillna("<missing>"),
        dataframe[second].astype("string").fillna("<missing>"),
    )
    if method == "crosstab":
        display_table = _bounded_crosstab(table, top_n=top_n)
        return display_table.reset_index(), {
            "row_column": first,
            "column_column": second,
            "row_count": int(len(dataframe)),
            "displayed_row_categories": int(display_table.shape[0]),
            "displayed_column_categories": int(display_table.shape[1]),
            "full_row_categories": int(table.shape[0]),
            "full_column_categories": int(table.shape[1]),
        }
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "chi_square_independence requires at least two categories in each column."
        )
    chi2, p_value, degrees_of_freedom, expected = stats.chi2_contingency(table)
    min_expected = float(pd.DataFrame(expected).min().min())
    result = pd.DataFrame(
        [
            {
                "row_column": first,
                "column_column": second,
                "chi_square": float(chi2),
                "degrees_of_freedom": int(degrees_of_freedom),
                "p_value": float(p_value),
                "min_expected_count": min_expected,
                "decision_at_0.05": (
                    "reject independence"
                    if float(p_value) < 0.05
                    else "do not reject independence"
                ),
            }
        ]
    )
    return result.round(6), {
        "method": "Pearson chi-square test of independence",
        "row_count": int(len(dataframe)),
        "row_categories": int(table.shape[0]),
        "column_categories": int(table.shape[1]),
    }


def _bounded_crosstab(table: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Keep displayed crosstabs compact while the caller can still explain truncation."""
    row_limit = max(1, min(top_n, CROSSTAB_MAX_DISPLAY_CATEGORIES))
    column_limit = CROSSTAB_MAX_DISPLAY_CATEGORIES
    row_order = table.sum(axis=1).sort_values(ascending=False).index[:row_limit]
    column_order = table.sum(axis=0).sort_values(ascending=False).index[:column_limit]
    # 原因：高基数日期、城市等交叉表会把回答撑到不可读。
    # 作用：展示最常见类别，完整类别数量留在 details 里供 Agent 说明截断口径。
    return table.loc[row_order, column_order]


def _mean_confidence_intervals(
    values: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    """Calculate a two-sided Student-t interval for each selected column mean."""
    _validate_confidence_level(confidence_level)
    rows: list[dict[str, Any]] = []
    for column in map(str, values.columns):
        sample = values[column].dropna()
        if len(sample) < 2:
            raise ValueError(
                f"mean_confidence_interval requires at least two observations in {column}."
            )
        count = int(len(sample))
        mean = float(sample.mean())
        standard_deviation = float(sample.std(ddof=1))
        standard_error = standard_deviation / count**0.5
        t_critical = float(
            stats.t.ppf((1.0 + confidence_level) / 2.0, count - 1)
        )
        margin = t_critical * standard_error
        rows.append(
            {
                "column": column,
                "count": count,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "degrees_of_freedom": count - 1,
                "confidence_level": confidence_level,
                "t_critical": t_critical,
                "margin_of_error": margin,
                "ci_lower": mean - margin,
                "ci_upper": mean + margin,
            }
        )
    return pd.DataFrame(rows).round(6)


def _one_sample_t_tests(
    values: pd.DataFrame,
    *,
    hypothesized_mean: float,
    confidence_level: float,
) -> pd.DataFrame:
    """Test each selected column mean against one explicit population value."""
    interval_rows = _mean_confidence_intervals(
        values,
        confidence_level=confidence_level,
    ).to_dict(orient="records")
    intervals: dict[str, dict[str, Any]] = {
        str(row["column"]): {str(key): value for key, value in row.items()}
        for row in interval_rows
    }
    alpha = 1.0 - confidence_level
    rows: list[dict[str, Any]] = []
    for column in map(str, values.columns):
        sample = values[column].dropna()
        test = stats.ttest_1samp(
            sample.to_numpy(),
            popmean=hypothesized_mean,
            nan_policy="omit",
        )
        interval = intervals[column]
        p_value = float(test.pvalue)
        rows.append(
            {
                "column": column,
                "count": int(interval["count"]),
                "sample_mean": float(interval["mean"]),
                "hypothesized_mean": hypothesized_mean,
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": int(interval["degrees_of_freedom"]),
                "p_value": p_value,
                "confidence_level": confidence_level,
                "ci_lower": float(interval["ci_lower"]),
                "ci_upper": float(interval["ci_upper"]),
                "decision": (
                    "reject null hypothesis"
                    if p_value < alpha
                    else "do not reject null hypothesis"
                ),
            }
        )
    return pd.DataFrame(rows).round(6)


def _split_group_samples(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    group_column: str,
    group_values: list[str],
    exact_groups: int | None = None,
) -> tuple[list[pd.Series], list[str]]:
    """Split one value column into one numeric sample per requested group.

    `group_values` names exactly the groups to compare; when empty, every
    non-null group in `group_column` is used.  `exact_groups` requires that many
    groups when two-sample tests need exactly two; otherwise any two or more
    groups are accepted.  Returns the numeric samples and group labels in the
    same order, validating that there are enough groups and at least two
    observations per sample.
    """
    observed_groups = list(
        dict.fromkeys(dataframe[group_column].dropna().astype(str))
    )
    selected_groups = group_values or observed_groups
    required = exact_groups if exact_groups is not None else 2
    if len(selected_groups) < required:
        raise ValueError(
            f"requires at least {required} groups in group_column, got "
            f"{len(selected_groups)}."
        )
    if exact_groups is not None and len(selected_groups) > exact_groups:
        raise ValueError(
            f"requires exactly {exact_groups} group_values, got "
            f"{len(selected_groups)}."
        )
    group_series = dataframe[group_column].astype(str)
    non_empty: list[tuple[pd.Series, str]] = []
    for group in selected_groups:
        sample = pd.to_numeric(
            dataframe.loc[group_series == group, value_column],
            errors="coerce",
        ).dropna()
        if not sample.empty:
            non_empty.append((sample, group))
    samples = [sample for sample, _ in non_empty]
    selected_groups = [group for _, group in non_empty]
    if len(samples) < required:
        raise ValueError(
            f"requires at least {required} non-empty groups in group_column."
        )
    if any(len(sample) < 2 for sample in samples):
        raise ValueError("requires at least two observations per group.")
    return samples, selected_groups


def _two_sample_t_test(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    group_column: str,
    group_values: list[str],
    confidence_level: float,
) -> pd.DataFrame:
    """Compare two independent group means with Welch's unequal-variance test."""
    _validate_confidence_level(confidence_level)
    samples, selected_groups = _split_group_samples(
        dataframe,
        value_column=value_column,
        group_column=group_column,
        group_values=group_values,
        exact_groups=2,
    )
    first, second = samples
    first_variance = float(first.var(ddof=1))
    second_variance = float(second.var(ddof=1))
    standard_error_squared = first_variance / len(first) + second_variance / len(second)
    if standard_error_squared == 0:
        raise ValueError("two_sample_t_test is undefined when both groups have zero variance.")
    standard_error = standard_error_squared**0.5
    degrees_of_freedom = standard_error_squared**2 / (
        (first_variance / len(first)) ** 2 / (len(first) - 1)
        + (second_variance / len(second)) ** 2 / (len(second) - 1)
    )
    test = stats.ttest_ind(
        first.to_numpy(),
        second.to_numpy(),
        equal_var=False,
        nan_policy="omit",
    )
    mean_difference = float(first.mean() - second.mean())
    t_critical = float(
        stats.t.ppf((1.0 + confidence_level) / 2.0, degrees_of_freedom)
    )
    margin = t_critical * standard_error
    p_value = float(test.pvalue)
    return pd.DataFrame(
        [
            {
                "value_column": value_column,
                "group_1": selected_groups[0],
                "group_1_count": len(first),
                "group_1_mean": float(first.mean()),
                "group_2": selected_groups[1],
                "group_2_count": len(second),
                "group_2_mean": float(second.mean()),
                "mean_difference": mean_difference,
                "standard_error": standard_error,
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": p_value,
                "confidence_level": confidence_level,
                "difference_ci_lower": mean_difference - margin,
                "difference_ci_upper": mean_difference + margin,
                "decision": (
                    "reject equal-means null hypothesis"
                    if p_value < 1.0 - confidence_level
                    else "do not reject equal-means null hypothesis"
                ),
            }
        ]
    ).round(6)


def _mann_whitney_u(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    group_column: str,
    group_values: list[str],
) -> pd.DataFrame:
    """Compare two independent groups with the non-parametric Mann-Whitney U test."""
    samples, selected_groups = _split_group_samples(
        dataframe,
        value_column=value_column,
        group_column=group_column,
        group_values=group_values,
        exact_groups=2,
    )
    first, second = samples
    test = stats.mannwhitneyu(
        first.to_numpy(),
        second.to_numpy(),
        alternative="two-sided",
        method="auto",
    )
    p_value = float(test.pvalue)
    return pd.DataFrame(
        [
            {
                "value_column": value_column,
                "group_1": selected_groups[0],
                "group_1_count": len(first),
                "group_1_median": float(first.median()),
                "group_2": selected_groups[1],
                "group_2_count": len(second),
                "group_2_median": float(second.median()),
                "u_statistic": float(test.statistic),
                "p_value": p_value,
                "decision_at_0.05": (
                    "reject equal-distributions null hypothesis"
                    if p_value < 0.05
                    else "do not reject equal-distributions null hypothesis"
                ),
            }
        ]
    ).round(6)


def _wilcoxon_signed_rank(
    dataframe: pd.DataFrame,
    *,
    before_column: str,
    after_column: str,
) -> pd.DataFrame:
    """Compare paired before/after measurements with the Wilcoxon signed-rank test."""
    before = pd.to_numeric(dataframe[before_column], errors="coerce")
    after = pd.to_numeric(dataframe[after_column], errors="coerce")
    paired = pd.DataFrame({"before": before, "after": after}).dropna()
    if len(paired) < 5:
        raise ValueError("wilcoxon_signed_rank requires at least five paired observations.")
    test = stats.wilcoxon(
        paired["before"].to_numpy(),
        paired["after"].to_numpy(),
        zero_method="wilcox",
    )
    p_value = float(test.pvalue)
    median_difference = float((paired["after"] - paired["before"]).median())
    return pd.DataFrame(
        [
            {
                "before_column": before_column,
                "after_column": after_column,
                "paired_count": len(paired),
                "median_difference": median_difference,
                "w_statistic": float(test.statistic),
                "p_value": p_value,
                "decision_at_0.05": (
                    "reject equal-medians null hypothesis"
                    if p_value < 0.05
                    else "do not reject equal-medians null hypothesis"
                ),
            }
        ]
    ).round(6)


def _kruskal_wallis(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    group_column: str,
) -> pd.DataFrame:
    """Compare two or more independent groups with the Kruskal-Wallis H test."""
    samples, selected_groups = _split_group_samples(
        dataframe,
        value_column=value_column,
        group_column=group_column,
        group_values=[],
    )
    test = stats.kruskal(*[sample.to_numpy() for sample in samples])
    p_value = float(test.pvalue)
    return pd.DataFrame(
        [
            {
                "value_column": value_column,
                "group_column": group_column,
                "group_count": len(samples),
                "groups": ", ".join(selected_groups),
                "h_statistic": float(test.statistic),
                "p_value": p_value,
                "decision_at_0.05": (
                    "reject equal-distributions null hypothesis"
                    if p_value < 0.05
                    else "do not reject equal-distributions null hypothesis"
                ),
            }
        ]
    ).round(6)


def _validate_confidence_level(confidence_level: float) -> None:
    """Reject percentages that cannot define a statistical confidence interval."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be greater than 0 and less than 1.")


def _apply_comparison_scope(
    frames: dict[str, pd.DataFrame],
    dataframe: pd.DataFrame,
    *,
    scope_table_name: str,
    scope_data_key: str,
    scope_lookup_key: str,
    scope_required_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limit analysis rows to entities classified by a workbook metadata table."""
    scope_arguments_present = (
        bool(scope_table_name),
        bool(scope_data_key),
        bool(scope_lookup_key),
        bool(scope_required_columns),
    )
    if not any(scope_arguments_present):
        return dataframe, {}
    if not all(scope_arguments_present):
        raise ValueError(
            "comparison scope requires scope_table_name, scope_data_key, "
            "scope_lookup_key, and scope_required_columns together."
        )

    lookup = frames[scope_table_name]
    missing_data_columns = [
        column for column in [scope_data_key] if column not in dataframe.columns
    ]
    missing_lookup_columns = [
        column
        for column in [scope_lookup_key, *scope_required_columns]
        if column not in lookup.columns
    ]
    if missing_data_columns or missing_lookup_columns:
        missing = ", ".join([*missing_data_columns, *missing_lookup_columns])
        raise KeyError(missing)

    eligible_lookup_rows = lookup[scope_required_columns].notna().all(axis=1)
    eligible_keys = lookup.loc[eligible_lookup_rows, scope_lookup_key].dropna()
    scoped = dataframe[dataframe[scope_data_key].isin(eligible_keys)].copy()
    if scoped.empty:
        raise ValueError("comparison scope selected no analysis rows.")

    # 原因：国家、地区汇总和收入组即使单位相同，也不是同层级的可比较样本。
    # 作用：利用工作簿自己的分类元数据确定总体，并在结果中留下可审计的筛选口径。
    return scoped, {
        "scope_rule": (
            f"{scope_data_key} matched {scope_table_name}.{scope_lookup_key}; "
            f"required non-null: {', '.join(scope_required_columns)}"
        ),
        "rows_before_scope": int(len(dataframe)),
        "rows_after_scope": int(len(scoped)),
    }


def _comparison_scope_candidate(
    frames: dict[str, pd.DataFrame],
    *,
    table_name: str,
    dataframe: pd.DataFrame,
    label_columns: list[str],
) -> dict[str, Any] | None:
    """Find one unambiguous metadata scope for classified and aggregate keys."""
    data_keys = label_columns or [
        str(column)
        for column in dataframe.select_dtypes(exclude="number").columns[:4]
    ]
    classification_name = re.compile(
        r"(region|category|type|level|group|class|地区|类别|类型|层级|分组)",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, dict[str, Any]]] = []
    for data_key in data_keys:
        if data_key not in dataframe.columns:
            continue
        data_values = set(dataframe[data_key].dropna().astype(str))
        if not data_values:
            continue
        for lookup_table_name, lookup in frames.items():
            if lookup_table_name == table_name:
                continue
            scope_columns = [
                str(column)
                for column in lookup.columns
                if classification_name.search(str(column))
                and lookup[column].notna().any()
                and lookup[column].isna().any()
            ]
            if not scope_columns:
                continue
            for lookup_key in lookup.columns:
                lookup_values = set(lookup[lookup_key].dropna().astype(str))
                overlap = data_values.intersection(lookup_values)
                if len(overlap) / len(data_values) < 0.8:
                    continue
                for scope_column in scope_columns:
                    classified = set(
                        lookup.loc[lookup[scope_column].notna(), lookup_key]
                        .dropna()
                        .astype(str)
                    )
                    unclassified = set(
                        lookup.loc[lookup[scope_column].isna(), lookup_key]
                        .dropna()
                        .astype(str)
                    )
                    if overlap.intersection(classified) and overlap.intersection(
                        unclassified
                    ):
                        # 原因：分类元数据中的空值常代表总计、地区或收入组，和实体行同算会扭曲分布。
                        # 作用：在严格键覆盖条件下自动建立同层级总体，
                        # 不依赖弱模型重复修正 Tool 参数。
                        candidates.append(
                            (
                                len(overlap),
                                {
                                    "scope_table_name": lookup_table_name,
                                    "scope_data_key": data_key,
                                    "scope_lookup_key": str(lookup_key),
                                    "scope_required_columns": [scope_column],
                                },
                            )
                        )
    if not candidates:
        return None
    # 原因：显示名称可能存在拼写差异，而稳定代码通常覆盖更多记录。
    # 作用：选择键覆盖数最高的元数据连接，减少自动作用域误删合法实体。
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _row_metric(values: pd.DataFrame) -> tuple[pd.Series, str]:
    """Use one selected metric or a same-unit row mean for outlier detection."""
    if len(values.columns) == 1:
        return values.iloc[:, 0], str(values.columns[0])
    return values.mean(axis=1), f"row mean across {len(values.columns)} selected columns"


def _validate_comparable_units(value_columns: list[str], method: str) -> None:
    """Reject explicit unit mixing before a row-level metric is calculated."""
    if method not in {"iqr_outliers", "zscore_outliers"} or len(value_columns) < 2:
        return
    signatures = {
        signature
        for column in value_columns
        if (signature := _unit_signature(column)) is not None
    }
    if len(signatures) > 1:
        # 原因：数量、百分比、货币等列的直接平均没有可解释的统计单位。
        # 作用：强制 Agent 返回 schema 重新选择同单位指标，而不是输出数值正确但语义错误的表。
        raise ValueError(
            "outlier value_columns contain incompatible explicit units: "
            f"{', '.join(sorted(signatures))}. Select one interpretable metric or "
            "repeated measurements with the same unit."
        )


def _unit_signature(column: str) -> str | None:
    """Extract a conservative unit marker from a human-readable column name."""
    parenthetical: list[str] = re.findall(r"\(([^()]*)\)", column.lower())
    if parenthetical:
        candidate = parenthetical[-1].strip()
        if candidate in {
            "%",
            "gbp",
            "minutes",
            "per cent",
            "percent",
            "percentage",
            "thousand",
            "thousands",
            "usd",
        }:
            return candidate
    lowered = column.lower()
    if "%" in lowered:
        return "percent"
    return None


def _column_list(value: Any) -> list[str]:
    """Normalize JSON-array Skill arguments into unique non-empty column names."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("column arguments must be JSON arrays of strings.")
    return list(
        dict.fromkeys(
            str(column).strip()
            for column in value
            if str(column).strip()
        )
    )


def _optional_text_argument(value: Any) -> str:
    """Normalize optional text arguments produced by weaker tool-calling models.

    原因：部分 OpenAI-compatible 本地模型会把可选空字段序列化成字符串 "null"。
    作用：让可选参数保持空值语义，避免把 "null" 误当成真实列名或表名。
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "null", "none"} else text


def create_skill() -> BaseSkill:
    """Factory used by SkillRegistry for zero-manual registration."""
    return ExcelStatisticsSkill()
