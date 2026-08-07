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
    use_chinese: bool = True,
) -> tuple[str, bool]:
    """Recompute each missing required method and verify the model's prose value.

    Every missing verifiable method is recomputed and either verified (the
    model's claimed value matches) or degraded (a local table is shown without
    citing the model's value).  Returns ``(prose, degraded)``:
    - ``prose`` is a replacement narrative when any method degraded, else
      ``""`` so the runner keeps the model's original answer.
    - ``degraded`` is True when at least one method had to fall back to a
      local table (the model's answer may contain a wrong self-computed value).
    """
    if not spreadsheet_paths:
        return "", False
    if not narrative:
        return "", False
    degraded = False
    degraded_prose = ""
    for tool_name, method in required_spreadsheet_methods:
        if tool_name not in missing_tools:
            continue
        verified = verify_one_method(
            (tool_name, method),
            spreadsheet_names=spreadsheet_names,
            spreadsheet_paths=spreadsheet_paths,
            user_question=user_question,
            narrative=narrative,
            debug_steps=debug_steps,
            step_number=len(steps),
            steps=steps,
            use_chinese=use_chinese,
        )
        if verified is None:
            continue
        synthetic_step, prose, is_verified = verified
        steps.append(synthetic_step)
        missing_tools.discard(tool_name)
        if not is_verified:
            degraded = True
            degraded_prose = prose
    # 原因：模型自算值全部匹配时保留原答案（prose 为空），只追加本地核验表；
    # 有任何降级时用中性 prose 替换，避免叙述与本地表冲突。
    if degraded:
        return degraded_prose, True
    return "", False


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
    use_chinese: bool = True,
) -> tuple[dict[str, Any], str, bool] | None:
    """Recompute one required method and accept when locally recomputable.

    Returns ``(synthetic_step, prose, is_verified)`` when the local recompute
    succeeds: ``is_verified`` marks a match, otherwise the method degraded and
    ``prose`` is a neutral line that does not cite the model's value.  Returns
    ``None`` only when the local recompute itself fails (no trusted table).
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
                            "method": method_name,
                        },
                    }
                }
            ],
        }
        claimed = _extract_claimed_value(method_name, narrative)
        strict = _VERIFIABLE_METHODS[method_name] in _STRICT_KEYS
        matched = False
        if claimed is not None and _values_match(
            local_values, claimed, strict=strict
        ):
            matched = True
        # 原因：describe 多列时 any-any 匹配可能把列 A 声称值当列 B 的值接受。
        # 作用：有列名上下文时要求声称列与本地列一致，避免误接受。
        if (
            method_name == "describe"
            and matched
            and (local_pairs := _local_describe_pairs(response.data))
            and (claimed_pairs := _claimed_describe_pairs(narrative))
        ):
            matched = any(
                local_column.casefold() == claimed_column.casefold()
                and _close_enough(local_mean, claimed_mean)
                for local_column, local_mean in local_pairs
                for claimed_column, claimed_mean in claimed_pairs
            )
        if matched:
            assert claimed is not None
            prose = (
                f"已在 {spreadsheet_name} 上复核 {method_name}："
                f"{_prose_claim(method_name, claimed)}。"
                if use_chinese
                else (
                    f"Verified {method_name} on {spreadsheet_name}: "
                    f"{_prose_claim_english(method_name, claimed)}."
                )
            )
            debug_steps.append(
                f"本地校验通过：{spreadsheet_name} / {method_name} / "
                "自算值与本地复算一致"
            )
            return synthetic_step, prose, True
        # 原因：模型自算值缺失或与本地不符，但本地复算已成功。
        # 作用：降级为展示本地核验表，不再引用模型自算值，避免用户拿到报错。
        prose = (
            f"已在 {spreadsheet_name} 上按 {method_name} 重新计算，核验表见下方。"
            if use_chinese
            else f"Recomputed {method_name} on {spreadsheet_name}; verified table below."
        )
        debug_steps.append(
            f"本地校验降级：{spreadsheet_name} / {method_name} / "
            "自算值缺失或不符，改用本地核验表"
        )
        return synthetic_step, prose, False
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
            # 原因：analysis_frames 可能暴露 Sheet1 和 Sheet1::table_1 两个帧，
            # 子串方向必须让“候选是 schema 表名的精确/父级匹配”，否则会错选未解析的大帧。
            # 作用：先精确匹配，再允许 schema 表名是候选的前缀（如 Sheet1 匹配 Sheet1::table_1）。
            exact = next(
                (
                    candidate
                    for candidate in frames
                    if candidate.casefold() == table_name.casefold()
                ),
                None,
            )
            if exact is not None:
                return exact
            for candidate in frames:
                if table_name.casefold() in candidate.casefold():
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
    if method_name in {"linear_regression", "logistic_regression"}:
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
    # 原因：pivot/date_extract/deduplicate/rank 用各自的参数而非 value_columns，
    # 且必须 dispatch 早于 value_columns 校验。
    # 作用：从问题文字和 schema 列名推导这些数据整理参数。
    if method_name == "pivot":
        row_column = next(
            (
                column
                for column in non_numeric_columns
                if column.casefold() in lowered
            ),
            non_numeric_columns[0] if non_numeric_columns else "",
        )
        column_column = next(
            (
                column
                for column in non_numeric_columns
                if column != row_column and column.casefold() in lowered
            ),
            non_numeric_columns[1] if len(non_numeric_columns) > 1 else "",
        )
        value_column = next(
            (
                column
                for column in numeric_columns
                if column.casefold() in lowered
            ),
            numeric_columns[-1] if numeric_columns else "",
        )
        agg = (
            "mean"
            if any(marker in lowered for marker in ("平均", "mean", "average"))
            else "sum"
        )
        return {
            "row_column": row_column,
            "column_column": column_column,
            "value_column": value_column,
            "agg": agg,
        }
    if method_name == "date_extract":
        all_columns = [str(column) for column in frame.columns]
        date_column = next(
            (
                column
                for column in all_columns
                if column.casefold() in lowered
            ),
            next(
                (
                    column
                    for column in all_columns
                    if any(
                        marker in column.casefold()
                        for marker in ("date", "时间", "日期")
                    )
                ),
                all_columns[0] if all_columns else "",
            ),
        )
        part_markers = {
            "year": ("年份", "年"),
            "month": ("月份", "月"),
            "quarter": ("季度", "季"),
            "weekday": ("周几", "星期", "weekday"),
        }
        parts = [
            part
            for part, markers in part_markers.items()
            if any(marker in lowered for marker in markers)
        ]
        return {"date_column": date_column, "parts": parts}
    if method_name == "deduplicate":
        all_columns = [str(column) for column in frame.columns]
        mentioned = [
            column for column in all_columns if column.casefold() in lowered
        ]
        keep = (
            "last"
            if any(marker in lowered for marker in ("保留最后", "keep last"))
            else "first"
        )
        return {"columns": mentioned, "keep": keep}
    if method_name == "rank":
        value_column = next(
            (
                column
                for column in numeric_columns
                if column.casefold() in lowered
            ),
            numeric_columns[0] if numeric_columns else "",
        )
        rank_method = (
            "ntile"
            if any(
                marker in lowered
                for marker in ("分位", "分桶", "ntile", "四分位", "top n", "分成")
            )
            else "rank"
        )
        return {"value_column": value_column, "rank_method": rank_method}
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
        # 原因：skill 要求恰好两个分类列，少于两个时传 [] 会让 skill 明确失败，
        # 由调用方按 fail-closed 处理，而不是传一个残缺参数产生误导性结果。
        if len(non_numeric_columns) < 2:
            return {"category_columns": []}
        return {"category_columns": non_numeric_columns[:2]}
    if method_name in {"two_sample_t_test", "mann_whitney_u", "kruskal_wallis"}:
        return {
            "value_columns": value_columns[:1],
            "group_column": (
                non_numeric_columns[0] if non_numeric_columns else ""
            ),
        }
    if method_name == "wilcoxon_signed_rank":
        return {"value_columns": value_columns[:2]}
    if method_name in {"one_sample_t_test"}:
        # 原因：t 检验的 H0 均值应尽量取自问题（如“是否为 5”），否则用 0 作为默认。
        # 作用：提取的假设值交给双尾 t 检验，方向词（大于/小于）无法体现，仍给出信息性 p 值。
        return {
            "value_columns": value_columns[:1],
            "hypothesized_mean": _extract_hypothesized_mean(user_question),
        }
    if method_name == "correlation":
        return {"value_columns": value_columns}
    return {"value_columns": value_columns}


def _extract_hypothesized_mean(question: str) -> float:
    """Extract a hypothesized mean from a t-test question, defaulting to 0.0."""
    lowered = question.strip()
    patterns = (
        re.compile(
            r"(?:为|等于|是|大于|小于|from|vs|against)\s*([-+]?\d+(?:\.\d+)?)",
            re.I,
        ),
        re.compile(r"[=:]\s*([-+]?\d+(?:\.\d+)?)"),
        re.compile(r"≠\s*([-+]?\d+(?:\.\d+)?)"),
    )
    for pattern in patterns:
        match = pattern.search(lowered)
        if match is not None:
            return float(match.group(1))
    return 0.0


# Methods whose headline value a model can plausibly self-report.  The value is
# the comparison key used to locate the number in rows/tables/details.
_VERIFIABLE_METHODS = {
    "describe": "mean",
    "iqr_outliers": "outlier_count",
    "zscore_outliers": "outlier_count",
    "missing": "missing",
    "quantiles": "p50",
    "normality_test": "p_value",
    "chi_square_independence": "p_value",
    "linear_regression": "r_squared",
    "one_way_anova": "p_value",
    "logistic_regression": "lr_p_value",
    "mean_confidence_interval": "ci_lower",
    "one_sample_t_test": "p_value",
    "two_sample_t_test": "p_value",
    "mann_whitney_u": "p_value",
    "wilcoxon_signed_rank": "p_value",
    "kruskal_wallis": "p_value",
    "pivot": "pivot_rows",
    "date_extract": "min_year",
    "deduplicate": "dropped_count",
}

_COMPARISON_EXTRACTORS = {
    "outlier_count": re.compile(
        r"outlier[_ ]?count\s*(?::?|=|is)\s*(\d+)", re.I
    ),
    "mean": re.compile(
        r"(?:平均|均值|mean)\D{0,20}?(-?\d[\d,]*(?:\.\d+)?)", re.I
    ),
    "missing": re.compile(r"(?:缺失|missing)\D{0,20}?(\d+)", re.I),
    "p50": re.compile(r"(?:p50|中位|median)\D{0,20}?(-?\d[\d,]*(?:\.\d+)?)", re.I),
    "p_value": re.compile(
        r"(?:p[- ]?value|p值|p 值)\D{0,12}?(\d*\.?\d+(?:e-?\d+)?)", re.I
    ),
    "ci_lower": re.compile(
        r"(?:ci[-_ ]?lower|置信下限|下限)\D{0,12}?(-?\d[\d,]*(?:\.\d+)?)", re.I
    ),
    "lr_p_value": re.compile(
        r"(?:lr[_-]?p[_-]?value|整体\s*p\s*值|likelihood ratio p|logistic p)\D{0,12}?"
        r"(\d*\.?\d+(?:e-?\d+)?)",
        re.I,
    ),
    "r_squared": re.compile(
        r"(?:r.?squared|R²|r²)\D{0,12}?(\d[\d.]*)", re.I
    ),
    "pivot_rows": re.compile(
        r"(?:pivot rows|透视行数|行数|rows)\D{0,12}?(\d+)", re.I
    ),
    "min_year": re.compile(
        r"(?:min year|最早年份|起始年份|年份范围|min_year)\D{0,12}?(\d{4})", re.I
    ),
    "dropped_count": re.compile(
        r"(?:dropped count|删除行数|移除行数|删除行)\D{0,12}?(\d+)", re.I
    ),
}


def _local_comparison_values(
    method_name: str,
    data: dict[str, Any],
) -> tuple[float, ...] | None:
    """Extract the local numeric value(s) to compare against for a method.

    统计方法的 headline 值(mean/p_value/r_squared/outlier_count)通常在结果表
    rows 或 named tables 里。describe 额外保留列名,供列身份校验使用。
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


# 原因：describe 的结果按列各一行，模型若声称“列 A 均值”却给出列 B 的值，
# any-any 匹配会误接受。作用：对该方法保留列身份，声称值与本地同列才接受。
def _local_describe_pairs(data: dict[str, Any]) -> list[tuple[str, float]] | None:
    """Return ``(column, mean)`` pairs for the describe result."""
    rows = data.get("rows")
    if not isinstance(rows, list):
        return None
    pairs: list[tuple[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        column = row.get("column")
        mean = row.get("mean")
        if isinstance(column, str) and isinstance(mean, (int, float)):
            pairs.append((column, float(mean)))
    return pairs or None


def _claimed_describe_pairs(
    narrative: str,
) -> list[tuple[str, float]] | None:
    """Return ``(column, mean)`` pairs from model prose mentioning a column + mean.

    原因：模型叙述形如“Sepal.Length 的平均值是 5.63”或“The mean of Sepal.Width
    is 3.06”。作用：在 平均/mean 之前就近捕获一个 ASCII 列名片段作为身份提示。
    """
    pattern = re.compile(
        r"(?P<column>[A-Za-z][A-Za-z0-9_.]+)"
        r".{0,40}?(?:平均|均值|mean)\D{0,12}?"
        r"(?P<value>-?\d[\d,]*(?:\.\d+)?)",
        re.I,
    )
    pairs: list[tuple[str, float]] = []
    for match in pattern.finditer(narrative):
        column = match.group("column").strip()
        value = float(re.sub(r"[,\s]", "", match.group("value")))
        if column:
            pairs.append((column, value))
    return pairs or None


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


# 原因：p 值/R² 值域在 [0,1]，绝对容差 1e-2 会让 0.049 与 0.059（跨显著性阈值）
# 被误判一致。作用：对这些键只用相对容差，显著差异不被接受。
_STRICT_KEYS = {"p_value", "r_squared", "lr_p_value"}
_STRICT_REL_TOL = 5e-3


def _values_match(
    local: tuple[float, ...],
    claimed: tuple[float, ...],
    *,
    strict: bool = False,
) -> bool:
    """Accept when any local value matches any claimed value within tolerance."""
    return any(
        _close_enough(local_value, claimed_value, strict=strict)
        for local_value in local
        for claimed_value in claimed
    )


def _close_enough(local: float, claimed: float, *, strict: bool = False) -> bool:
    if int(local) == int(claimed) and float(local).is_integer() and float(
        claimed
    ).is_integer():
        return int(local) == int(claimed)
    difference = abs(claimed - local)
    if strict:
        return difference <= _STRICT_REL_TOL * max(abs(local), 1e-6)
    return difference <= VERIFY_ABS_TOL or difference <= VERIFY_REL_TOL * max(
        abs(local), 1.0
    )


def _prose_claim(method_name: str, claimed: tuple[float, ...]) -> str:
    """Render a stable human sentence describing the verified claimed value."""
    key = _VERIFIABLE_METHODS[method_name]
    if key == "outlier_count":
        return f"离群点数量 {int(claimed[0])}"
    return f"{key} = {claimed[0]:g}"


def _prose_claim_english(method_name: str, claimed: tuple[float, ...]) -> str:
    """Render an English prose claim matching :func:`_prose_claim`."""
    key = _VERIFIABLE_METHODS[method_name]
    if key == "outlier_count":
        return f"outlier count {int(claimed[0])}"
    return f"{key} = {claimed[0]:g}"
