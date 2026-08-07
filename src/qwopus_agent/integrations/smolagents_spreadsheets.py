"""Spreadsheet-specific policies used by the smolagents runtime."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from qwopus_agent.analysis.excel_processing import read_spreadsheet
from qwopus_agent.integrations import smolagents_debug
from qwopus_agent.skills import SkillRequest
from qwopus_agent.skills.excel_statistics import ExcelStatisticsSkill

# 原因：方法词元在每次 intent 路由都会用到，作为常量只构建一次。
# 作用：避免 required_spreadsheet_methods 每次调用都重建这一组嵌套元组。
_METHOD_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lookup", ("lookup", "某项", "某一项", "某行", "某一行", "sku", "是多少")),
    (
        "chi_square_independence",
        ("chi-square", "chi square", "卡方", "独立性", "independence"),
    ),
    (
        "normality_test",
        ("normality", "normal distribution", "正态", "正态性", "normal test"),
    ),
    ("quantiles", ("quantile", "percentile", "分位", "百分位", "p90", "p50")),
    ("covariance", ("covariance", "协方差")),
    ("correlation", ("correlation", "相关", "相关性")),
    ("group_summary", ("group summary", "分组统计", "按组统计", "by group")),
    ("missing", ("missing", "缺失", "空值", "null", "na")),
    # 原因：z-score 与 iqr 都回答“离群点”，但弱模型会混淆两者。
    # 作用：z-score 家族词显式路由到 zscore_outliers，裸“离群/异常”仍走 iqr。
    (
        "zscore_outliers",
        ("z-score", "z score", "zscore", "z值", "z 值", "z分数", "z 分数"),
    ),
    ("iqr_outliers", ("outlier", "异常", "离群", "极端值")),
    ("frequency", ("frequency", "count", "counts", "频数", "频率", "计数")),
    (
        "mean_confidence_interval",
        # 原因：裸 "ci" 会子串命中 "specific"/"significant" 等常见英文词，误路由到置信区间。
        # 作用：只保留明确表达置信区间的词，避免抢走 describe/显著性检验。
        ("confidence interval", "置信区间", "置信"),
    ),
    (
        "two_sample_t_test",
        ("two-sample", "two sample", "双样本", "两组", "两样本", "组间比较"),
    ),
    (
        "mann_whitney_u",
        (
            "mann-whitney",
            "mann whitney",
            "曼-惠特尼",
            "曼惠特尼",
            "u test",
            "u检验",
            "u 检验",
            "秩和检验",
            "rank-sum",
            "rank sum",
        ),
    ),
    # 原因：one_sample 用 "t 检验"/"t-test" 词，必须在 wilcoxon 的 "配对" 之前，
    # 否则 "配对 t 检验" 会被误路由到非参 wilcoxon。
    # 作用：显式 t 检验词优先；wilcoxon 专用词不受影响。
    (
        "one_sample_t_test",
        ("t-test", "t test", "t 检验", "t检验", "单样本t"),
    ),
    (
        "wilcoxon_signed_rank",
        ("wilcoxon", "威尔科克森", "符号秩", "signed rank", "配对"),
    ),
    (
        "kruskal_wallis",
        (
            "kruskal-wallis",
            "kruskal",
            "克鲁斯卡尔",
            "多组比较",
            "h检验",
            "h 检验",
        ),
    ),
    # 原因：pivot/date_extract/deduplicate/rank 是数据整理意图，与统计方法并列。
    # 作用：裸 "date"/"year"/"rank" 太易误撞（date 常见于普通列名、rank 撞 rank-sum），
    # 只用带动作的组合词做 marker，避免抢走 describe/回归等更明确的意图。
    ("pivot", ("pivot", "透视")),
    (
        "deduplicate",
        ("去重", "重复项", "重复行", "duplicate rows", "dedup", "删重复"),
    ),
    (
        "date_extract",
        (
            "提取日期",
            "提取年份",
            "提取月份",
            "提取季度",
            "提取周几",
            "年月日拆分",
            "拆分日期",
            "extract year",
            "extract month",
            "year and month",
            "月份和年份",
        ),
    ),
    # 原因：rank 的 marker 只用明确的“排名”动作词；"前 10"/"top 10" 常表示 top-N 排行
    # 或“前几行”，不是逐行排名，会让 describe/lookup 意图被误路由。
    ("rank", ("排名", "名次", "排位", "rank of", "按分数排序")),
    (
        "describe",
        (
            "summary",
            "describe",
            "mean",
            "average",
            "平均",
            "均值",
            "概况",
            "统计摘要",
            "描述统计",
        ),
    ),
)


def required_spreadsheet_method(user_question: str) -> tuple[str, str] | None:
    """Map explicit modeling requests to the deterministic local Skill method."""
    methods = required_spreadsheet_methods(user_question)
    return methods[0] if methods else None


def spreadsheet_intent_guidance(user_question: str) -> str:
    """Describe deterministic spreadsheet method requirements in the Agent prompt."""
    methods = required_spreadsheet_methods(user_question)
    if not methods:
        return "Spreadsheet intent: infer the smallest reviewed computation needed."
    names = ", ".join(".".join(method) for method in methods)
    lines = [f"Spreadsheet intent decomposition: required computations are {names}."]
    if ("excel_statistics", "lookup") in methods:
        # 原因：弱模型知道要 lookup 后仍可能忘记传 lookup_value。
        # 作用：把用户问题中的行名、字段名、SKU、角色属性或标签作为查询键交给本地 Skill。
        lines.append(
            "For lookup, set lookup_value to the exact item label, row name, SKU, "
            "character stat, or field named in the user question; return the matched row "
            "as a Markdown table before explaining it."
        )
    if len(methods) > 1:
        # 原因：抽象问题通常不是一个统计量能回答，弱模型容易停在第一张表。
        # 作用：要求 Agent 先完成每个本地计算，再综合解释哪些发现重要。
        lines.append(
            "Run every listed computation before writing the answer, then synthesize which "
            "findings are important instead of treating the first table as sufficient."
        )
    return " ".join(lines)


def spreadsheet_self_computed_warning(
    required_methods: tuple[tuple[str, str], ...],
) -> str:
    """Return a strong warning for weak models that skip the Excel compute tools.

    原因：弱模型会直接在 final_answer 里给出自算数值，但这些数值不会进入
    最终表格（运行时只接受本地 Skill 工具返回的表）。
    作用：明确告诉模型自算无效，必须调用本地计算工具。
    """
    if not required_methods:
        return ""
    names = ", ".join(".".join(method) for method in required_methods)
    return (
        "Self-computed statistics are NOT accepted as final tables: reported "
        "values are verified against a local recomputation when the tool was "
        "skipped. Prefer calling "
        f"{names} so every reported value comes from its returned table before "
        "writing the answer; do not compute mean, p-value, coefficient, or "
        "R-squared yourself."
    )


def has_required_spreadsheet_method(
    steps: list[dict[str, Any]],
    *,
    user_question: str,
    required_method: tuple[str, str],
) -> bool:
    """Return whether a required spreadsheet method was called with valid arguments."""
    tool_name, method = required_method
    if method != "lookup":
        return smolagents_debug.has_successful_tool_method(
            steps,
            tool_name=tool_name,
            method=method,
        )
    normalized_question = normalize_lookup_text(user_question)
    for arguments in successful_tool_arguments(steps, tool_name=tool_name):
        if arguments.get("method") != "lookup":
            continue
        lookup_value = str(arguments.get("lookup_value") or "").strip()
        if not lookup_value:
            continue
        # 原因：弱模型会把表格样例中的 CON/SIZ 等值误当成用户查询目标。
        # 作用：lookup 只有在查询键来自用户问题时才算满足必需计算。
        if normalize_lookup_text(lookup_value) in normalized_question:
            return True
    return False


def successful_tool_arguments(
    steps: list[dict[str, Any]],
    *,
    tool_name: str,
) -> list[dict[str, Any]]:
    """Extract successful Tool arguments for local runtime validation."""
    arguments_list: list[dict[str, Any]] = []
    for step in steps:
        if (
            not isinstance(step, dict)
            or step.get("error")
            or not isinstance(step.get("observations"), str)
            or not step["observations"].strip()
        ):
            continue
        for tool_call in step.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict) or function.get("name") != tool_name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if isinstance(arguments, dict):
                arguments_list.append(arguments)
    return arguments_list


def normalize_lookup_text(value: str) -> str:
    """Normalize lookup text for exact user-question containment checks."""
    return "".join(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", value.casefold()))


def required_spreadsheet_methods(user_question: str) -> tuple[tuple[str, str], ...]:
    """Map spreadsheet intents to deterministic local Skill methods."""
    normalized = user_question.casefold()
    if any(marker in normalized for marker in ("logistic", "logit", "逻辑回归", "二分类")):
        # 原因：“逻辑回归”含“回归”，必须先于 generic regression 检查命中。
        # 作用：把二分类目标问题路由到 logistic_regression，而不是线性回归。
        return (("excel_modeling", "logistic_regression"),)
    if any(marker in normalized for marker in ("anova", "方差分析")):
        return (("excel_modeling", "one_way_anova"),)
    if any(
        marker in normalized
        for marker in ("regression", "回归", "summary(lm", "linear model")
    ):
        return (("excel_modeling", "linear_regression"),)
    # 原因：用户常用“这个数据有什么问题/帮我看看”这类抽象需求，但弱模型容易只复述 schema。
    # 作用：把泛化诊断拆成几项稳定的本地统计检查，再交给模型解释。
    diagnostic_markers = (
        "有什么问题",
        "数据问题",
        "质量",
        "health",
        "diagnose",
        "diagnostic",
        "profile",
        "inspect",
        "帮我看看",
        "分析一下",
    )
    if any(marker in normalized for marker in diagnostic_markers):
        return (
            ("excel_statistics", "missing"),
            ("excel_statistics", "describe"),
            ("excel_statistics", "quantiles"),
            ("excel_statistics", "iqr_outliers"),
        )
    distribution_markers = (
        "分布怎么样",
        "分布情况",
        "distribution",
        "distributed",
    )
    if any(marker in normalized for marker in distribution_markers):
        return (
            ("excel_statistics", "describe"),
            ("excel_statistics", "quantiles"),
            ("excel_statistics", "normality_test"),
        )
    relationship_markers = (
        "两个分类变量",
        "分类变量有关",
        "categorical association",
        "categorical relationship",
    )
    if any(marker in normalized for marker in relationship_markers):
        return (
            ("excel_statistics", "crosstab"),
            ("excel_statistics", "chi_square_independence"),
        )
    # 原因：弱模型常能理解“异常/分布/某项”但不会稳定选择正确 Excel Skill 方法。
    # 作用：把高频抽象统计意图映射到本地确定性方法，由运行时强制补调对应工具。
    for method, markers in _METHOD_MARKERS:
        if any(marker in normalized for marker in markers):
            return (("excel_statistics", method),)
    return ()


def apply_missing_spreadsheet_fallbacks(
    steps: list[dict[str, Any]],
    *,
    spreadsheet_paths: dict[str, Path],
    spreadsheet_names: list[str],
    user_question: str,
    missing_tools: set[str],
    required_spreadsheet_methods: tuple[tuple[str, str], ...],
    debug_steps: list[str],
) -> str:
    """Run deterministic local spreadsheet fallback when the Agent misses a required lookup."""
    if (
        ("excel_statistics", "lookup") not in required_spreadsheet_methods
        or "excel_statistics" not in missing_tools
        or not spreadsheet_paths
    ):
        return ""
    lookup_value = lookup_value_from_question(user_question)
    if not lookup_value:
        return ""
    skill = ExcelStatisticsSkill()
    for spreadsheet_name in spreadsheet_names:
        path = spreadsheet_paths.get(spreadsheet_name)
        if path is None:
            continue
        try:
            table_names = tuple(read_spreadsheet(path).analysis_frames())
        except (OSError, ValueError) as exc:
            debug_steps.append(f"本地 lookup 兜底读取失败：{spreadsheet_name}: {exc}")
            continue
        for table_name in table_names:
            response = asyncio.run(
                skill.run(
                    SkillRequest(
                        query=user_question,
                        arguments={
                            "file_path": str(path),
                            "table_name": table_name,
                            "method": "lookup",
                            "lookup_value": lookup_value,
                        },
                    )
                )
            )
            if not response.success or not response.data.get("rows"):
                continue
            # 原因：弱模型可能连续两轮把样例值当 lookup_value，导致正确工具永远不达标。
            # 作用：在已授权文件范围内补一次确定性查找，并写回同一条工具轨迹供表格渲染。
            steps.append(
                {
                    "step_number": len(steps) + 1,
                    "observations": response.content,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "excel_statistics",
                                "arguments": {
                                    "file_name": spreadsheet_name,
                                    "table_name": table_name,
                                    "method": "lookup",
                                    "lookup_value": lookup_value,
                                },
                            }
                        }
                    ],
                }
            )
            missing_tools.discard("excel_statistics")
            debug_steps.append(
                f"本地 lookup 兜底完成：{spreadsheet_name} / {table_name} / {lookup_value}"
            )
            if any("\u4e00" <= character <= "\u9fff" for character in user_question):
                return f"{lookup_value} 的数值见下方本地核验表。"
            return f"The value for {lookup_value} is shown in the verified local table below."
    return ""


def lookup_value_from_question(user_question: str) -> str:
    """Extract a concise lookup key from common item-value questions."""
    question = user_question.strip()
    for pattern in (
        r"\b([A-Za-z][A-Za-z0-9_.-]{1,30})\b\s*(?:是|=|为|多少)",
        r"(?:what is|lookup|find)\s+([A-Za-z][A-Za-z0-9_.-]{1,30})\b",
    ):
        if match := re.search(pattern, question, flags=re.IGNORECASE):
            return match.group(1).strip()
    if "是多少" in question:
        return question.split("是多少", 1)[0].strip(" ，,?？")
    return ""


def spreadsheet_result_tables(steps: list[dict[str, Any]]) -> list[str]:
    """Extract only bounded Markdown tables returned by spreadsheet compute tools."""
    tables: list[str] = []
    for tool_name in ("excel_statistics", "excel_modeling", "excel_analysis"):
        for observation in smolagents_debug.extract_tool_observations(steps, tool_name):
            tables.extend(extract_markdown_tables(observation))
    return list(dict.fromkeys(tables))


def spreadsheet_computation_summary(
    steps: list[dict[str, Any]],
    *,
    user_question: str,
    use_chinese: bool,
) -> str:
    """Summarize successful local spreadsheet computations in prose."""
    method_observations: dict[str, str] = {}
    lookup_values: list[str] = []
    for step in steps:
        if (
            not isinstance(step, dict)
            or step.get("error")
            or not isinstance(step.get("observations"), str)
            or not step["observations"].strip()
        ):
            continue
        observation = step["observations"]
        for tool_call in step.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            if function.get("name") not in {"excel_statistics", "excel_modeling"}:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, dict):
                continue
            method = str(arguments.get("method") or "").strip()
            if not method or method in method_observations:
                continue
            method_observations[method] = observation
            if method == "lookup":
                lookup_value = str(arguments.get("lookup_value") or "").strip()
                if lookup_value:
                    lookup_values.append(lookup_value)
    if not method_observations:
        return ""

    # 原因：部分模型会调用了统计工具但正文漏解释关键结果，尤其“0 个异常值”。
    # 作用：用本地 Observation 的元数据补一段稳定解读，避免最终答案只剩附表。
    if use_chinese:
        lines = ["## 本地统计解读"]
        if "lookup" in method_observations:
            key = lookup_values[0] if lookup_values else user_question.strip()
            lines.append(f"- 已按用户问题中的 `{key}` 执行精确查找，具体命中值见下方核验表。")
        if "missing" in method_observations:
            lines.append(
                "- 已检查缺失值；各列缺失数量和比例见核验表，可用于判断数据完整性。"
            )
        if "describe" in method_observations:
            lines.append(
                "- 已生成描述统计，包括均值、中位数、标准差、范围、四分位数和分布形态指标。"
            )
        if "quantiles" in method_observations:
            lines.append("- 已计算关键分位数，用来观察数据集中区间、尾部范围和潜在偏态。")
        if "iqr_outliers" in method_observations:
            count = extract_outlier_count(method_observations["iqr_outliers"])
            if count == 0:
                lines.append("- IQR 异常值检查显示 outlier_count 为 0，按本次阈值未发现离群点。")
            elif count is None:
                lines.append("- 已完成 IQR 异常值检查，具体异常行或边界见核验表。")
            else:
                lines.append(
                    f"- IQR 异常值检查发现 {count} 个候选离群点，"
                    "需结合业务背景判断是否为错误。"
                )
        if "zscore_outliers" in method_observations:
            count = extract_outlier_count(method_observations["zscore_outliers"])
            if count == 0:
                lines.append(
                    "- Z-score 异常值检查显示 outlier_count 为 0，按本次阈值未发现极端值。"
                )
            elif count is None:
                lines.append("- 已完成 Z-score 异常值检查，具体结果见核验表。")
            else:
                lines.append(f"- Z-score 异常值检查发现 {count} 个候选极端值。")
        if "one_sample_t_test" in method_observations:
            lines.append(
                "- 已执行单样本 t 检验：样本均值与假设总体均值的差异及 p 值见核验表。"
            )
        if "two_sample_t_test" in method_observations:
            lines.append(
                "- 已对两组执行 Welch 双样本 t 检验，统计量、p 值与差值区间见核验表。"
            )
        if "mann_whitney_u" in method_observations:
            lines.append(
                "- 已对两组执行 Mann-Whitney U 检验，统计量与 p 值见核验表。"
            )
        if "wilcoxon_signed_rank" in method_observations:
            lines.append(
                "- 已对配对前后执行 Wilcoxon 符号秩检验，统计量与 p 值见核验表。"
            )
        if "kruskal_wallis" in method_observations:
            lines.append(
                "- 已对多个组执行 Kruskal-Wallis H 检验，统计量与 p 值见核验表。"
            )
        if "pivot" in method_observations:
            lines.append(
                "- 已按两个分类字段对数值列透视汇总，聚合单元格值见核验表。"
            )
        if "date_extract" in method_observations:
            lines.append(
                "- 已从日期列提取年月等时间分量，解析数量与年份范围见核验表。"
            )
        if "deduplicate" in method_observations:
            lines.append(
                "- 已按指定列去除重复行，原始行数、保留行数与删除行数见核验表。"
            )
        if "rank" in method_observations:
            lines.append("- 已对数值列计算排名或分位分组，每行结果见核验表。")
        return "\n".join(lines)

    lines = ["## Local Statistical Reading"]
    if "lookup" in method_observations:
        key = lookup_values[0] if lookup_values else user_question.strip()
        lines.append(
            f"- The workbook was searched for `{key}`; the verified match is shown below."
        )
    if "missing" in method_observations:
        lines.append(
            "- Missing values were checked; counts and rates are shown in the verified table."
        )
    if "describe" in method_observations:
        lines.append(
            "- Descriptive statistics were computed, including mean, median, standard "
            "deviation, range, quartiles, and shape metrics."
        )
    if "quantiles" in method_observations:
        lines.append(
            "- Quantiles were computed to inspect central range, tails, and possible skew."
        )
    if "iqr_outliers" in method_observations:
        count = extract_outlier_count(method_observations["iqr_outliers"])
        if count == 0:
            lines.append(
                "- The IQR outlier check returned outlier_count 0, so no outliers "
                "were found under this threshold."
            )
        elif count is None:
            lines.append(
                "- The IQR outlier check was completed; candidate rows or bounds "
                "are shown below."
            )
        else:
            lines.append(
                f"- The IQR outlier check found {count} candidate outliers; "
                "interpret them with domain context."
            )
    if "zscore_outliers" in method_observations:
        count = extract_outlier_count(method_observations["zscore_outliers"])
        if count == 0:
            lines.append(
                "- The Z-score outlier check returned outlier_count 0, so no extreme "
                "values were found under this threshold."
            )
        elif count is None:
            lines.append("- The Z-score outlier check was completed; details are shown below.")
        else:
            lines.append(f"- The Z-score outlier check found {count} candidate extreme values.")
    if "one_sample_t_test" in method_observations:
        lines.append(
            "- A one-sample t-test was run; the difference from the hypothesized mean "
            "and the p-value are in the verified table."
        )
    if "two_sample_t_test" in method_observations:
        lines.append(
            "- A Welch two-sample t-test was run; the statistic, p-value, and "
            "difference interval are in the verified table."
        )
    if "mann_whitney_u" in method_observations:
        lines.append(
            "- A Mann-Whitney U test was run for the two groups; the statistic and "
            "p-value are in the verified table."
        )
    if "wilcoxon_signed_rank" in method_observations:
        lines.append(
            "- A Wilcoxon signed-rank test was run on the paired measurements; the "
            "statistic and p-value are in the verified table."
        )
    if "kruskal_wallis" in method_observations:
        lines.append(
            "- A Kruskal-Wallis H test was run across groups; the statistic and "
            "p-value are in the verified table."
        )
    if "pivot" in method_observations:
        lines.append(
            "- The value column was pivoted by two categorical fields; the aggregated "
            "cells are in the verified table."
        )
    if "date_extract" in method_observations:
        lines.append(
            "- Date components were extracted from the date column; the parsed count "
            "and year range are in the verified table."
        )
    if "deduplicate" in method_observations:
        lines.append(
            "- Duplicate rows were removed by the selected columns; total, kept, and "
            "dropped counts are in the verified table."
        )
    if "rank" in method_observations:
        lines.append(
            "- The numeric column was ranked or binned; per-row results are shown below."
        )
    return "\n".join(lines)


def extract_outlier_count(observation: str) -> int | None:
    """Read outlier_count from a successful spreadsheet Tool observation."""
    match = re.search(r"(?:^|[-|]\s*)outlier_count\s*[:|]\s*(\d+)", observation, re.MULTILINE)
    if match:
        return int(match.group(1))
    lines = [line.strip() for line in observation.splitlines()]
    for index, line in enumerate(lines):
        if "outlier_count" not in line or not line.startswith("|"):
            continue
        if index + 2 >= len(lines) or not is_markdown_table_delimiter(lines[index + 1]):
            continue
        columns = [cell.strip() for cell in line.strip("|").split("|")]
        try:
            count_index = columns.index("outlier_count")
        except ValueError:
            continue
        data_cells = [cell.strip() for cell in lines[index + 2].strip("|").split("|")]
        if count_index < len(data_cells) and data_cells[count_index].isdigit():
            return int(data_cells[count_index])
    return None


def extract_markdown_tables(content: str) -> list[str]:
    """Extract contiguous GFM table blocks without returning surrounding Tool text."""
    lines = content.splitlines()
    tables: list[str] = []
    index = 0
    while index + 1 < len(lines):
        if (
            lines[index].strip().startswith("|")
            and lines[index].strip().endswith("|")
            and is_markdown_table_delimiter(lines[index + 1])
        ):
            end = index + 2
            while (
                end < len(lines)
                and lines[end].strip().startswith("|")
                and lines[end].strip().endswith("|")
            ):
                end += 1
            tables.append("\n".join(line.strip() for line in lines[index:end]))
            index = end
            continue
        index += 1
    return tables


def remove_markdown_tables(content: str) -> str:
    """Remove model-authored GFM tables before trusted local tables are attached."""
    lines = content.splitlines()
    retained: list[str] = []
    index = 0
    while index < len(lines):
        if _looks_like_table_row(lines[index]):
            # 原因：弱模型常输出缺少 GFM delimiter 的伪表格，合法表格检测无法移除。
            # 作用：清掉所有模型自写 pipe-row 表格，只保留本地 Tool 重新附加的核验表。
            index += 1
            while index < len(lines) and _looks_like_table_row(lines[index]):
                index += 1
            continue
        retained.append(lines[index])
        index += 1
    return "\n".join(retained)


def sanitize_spreadsheet_narrative(
    content: str,
    *,
    required_method: tuple[str, str] | None,
    use_chinese: bool,
) -> str:
    """Remove known method contradictions and add one deterministic limitation."""
    blocked = re.compile(
        r"(不等方差.{0,24}Tukey|Tukey.{0,24}不等方差|"
        r"unequal[- ]variance.{0,24}Tukey|Tukey.{0,24}unequal[- ]variance|"
        r"Welch[-\N{NON-BREAKING HYPHEN}\N{EN DASH} ]Tukey|"
        r"来自模型输出|from (?:the )?model output|1\.96\s*[×*]\s*SE)",
        flags=re.IGNORECASE,
    )
    retained = [line for line in content.splitlines() if not blocked.search(line)]
    if required_method == ("excel_modeling", "one_way_anova"):
        note = (
            "方法限制：Tukey HSD 假设组内方差近似相等；若 Levene 检验显著，"
            "其比较只能作为探索性结果，本次未计算 Games-Howell。"
            if use_chinese
            else (
                "Method limit: Tukey HSD assumes similar within-group variances. "
                "When Levene's test is significant, treat it as exploratory; "
                "Games-Howell was not computed."
            )
        )
        retained.extend(["", f"> {note}"])
    elif required_method == ("excel_modeling", "linear_regression"):
        note = (
            "方法限制：原始回归系数受变量量纲影响，不能单独作为变量重要性排名。"
            if use_chinese
            else (
                "Method limit: raw regression coefficients depend on variable scale "
                "and are not a standalone variable-importance ranking."
            )
        )
        retained.extend(["", f"> {note}"])
    elif required_method == ("excel_modeling", "logistic_regression"):
        note = (
            "方法限制：逻辑回归系数以 log-odds 为单位，不能直接解释为概率变化。"
            if use_chinese
            else (
                "Method limit: logistic regression coefficients are in log-odds and "
                "are not direct probability changes."
            )
        )
        retained.extend(["", f"> {note}"])
    return "\n".join(retained)


def is_markdown_table_delimiter(line: str) -> bool:
    """Validate the GFM delimiter row used to distinguish tables from prose."""
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _looks_like_table_row(line: str) -> bool:
    """Return whether a line is a table row, even if the table is malformed."""
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")
