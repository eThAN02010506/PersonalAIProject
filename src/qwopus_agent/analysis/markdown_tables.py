"""Small, dependency-free Markdown table rendering for bounded pandas results."""

from __future__ import annotations

import pandas as pd


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub-Flavored Markdown table."""
    if dataframe.empty:
        return "_Empty result._"
    columns = [str(column) for column in dataframe.columns]
    rows = dataframe.astype(str).values.tolist()
    # 原因：DataFrame.to_markdown 依赖未安装的 tabulate，直接调用会让干净环境失败。
    # 作用：分析页、聊天页和 Agent Observation 共用稳定的 GFM 表格格式。
    header = "| " + " | ".join(_escape_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_escape_cell(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _escape_cell(value: object) -> str:
    """Keep separators and line breaks inside one Markdown table cell."""
    return str(value).replace("|", r"\|").replace("\n", " ")
