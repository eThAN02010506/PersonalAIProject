"""Uploaded document and spreadsheet analysis.

This layer performs local analysis only. It intentionally does not send full files or full Excel
tables to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qwopus_agent.documents.parser import ParsedDocument, parse_document


SPREADSHEET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass(frozen=True)
class AnalysisResult:
    """Structured analysis result for UI and future report generation."""

    # 原因：UI 和报告模块需要统一入口展示分析摘要。
    # 作用：保存人类可读 Markdown 总结。
    markdown_summary: str

    # 原因：Streamlit 可以直接展示表格；报告模块也能复用。
    # 作用：保存多个可展示 dataframe。
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    # 原因：后续接 MiniRAG 或报告生成时需要轻量元数据。
    # 作用：保存 schema、样本大小、文件类型等结构化信息。
    metadata: dict[str, Any] = field(default_factory=dict)

    # 原因：非结构化文档后续要入 MiniRAG。
    # 作用：保存 Markdown 正文；Excel 不保存整表。
    markdown_document: str | None = None

    # 原因：用户在页面输入的是分析问题，不能只回显 preview。
    # 作用：保存 LLM 基于解析内容生成的真正分析答案。
    llm_analysis: str | None = None


def analyze_uploaded_file(file_path: str | Path, user_question: str = "") -> AnalysisResult:
    """Analyze one uploaded file using local Python."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in SPREADSHEET_EXTENSIONS:
        return _analyze_spreadsheet(path, user_question=user_question)

    parsed = parse_document(path)
    return _analyze_markdown_document(parsed, user_question=user_question)


def _analyze_spreadsheet(path: Path, user_question: str = "") -> AnalysisResult:
    """Inspect spreadsheet schema, sample rows, and safe local summaries."""
    sheets = _read_spreadsheet(path)
    summary_lines = [
        f"# Spreadsheet Analysis: {path.name}",
        "",
        "This analysis uses local pandas inspection only.",
        "Full table data is not sent to an LLM.",
    ]
    tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {"source_type": "spreadsheet", "sheets": {}}

    for sheet_name, df in sheets.items():
        schema = pd.DataFrame(
            {
                "column": [str(column) for column in df.columns],
                "dtype": [str(dtype) for dtype in df.dtypes],
                "non_null": [int(df[column].notna().sum()) for column in df.columns],
                "missing": [int(df[column].isna().sum()) for column in df.columns],
            }
        )
        sample = df.head(5)
        numeric_summary = df.describe(include="number").transpose().reset_index()
        if not numeric_summary.empty:
            numeric_summary = numeric_summary.rename(columns={"index": "column"})

        tables[f"{sheet_name}_schema"] = schema
        tables[f"{sheet_name}_sample"] = sample
        if not numeric_summary.empty:
            tables[f"{sheet_name}_numeric_summary"] = numeric_summary

        metadata["sheets"][sheet_name] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(column) for column in df.columns],
        }

        # 原因：展示层需要快速看出文件规模和字段情况。
        # 作用：生成不包含全量数据的摘要文本。
        summary_lines.extend(
            [
                "",
                f"## Sheet: {sheet_name}",
                f"- Rows: {len(df)}",
                f"- Columns: {len(df.columns)}",
                f"- Column names: {', '.join(str(column) for column in df.columns)}",
            ]
        )

    if user_question.strip():
        summary_lines.extend(["", "## User Question", user_question.strip()])

    return AnalysisResult(
        markdown_summary="\n".join(summary_lines),
        tables=tables,
        metadata=metadata,
        markdown_document=_spreadsheet_analysis_context(summary_lines, tables),
    )


def _spreadsheet_analysis_context(
    summary_lines: list[str],
    tables: dict[str, pd.DataFrame],
) -> str:
    """Build a bounded LLM context from schema/sample/summary tables only."""
    sections = ["\n".join(summary_lines)]
    for table_name, dataframe in tables.items():
        # 原因：Excel 分析不能把整表塞给 LLM。
        # 作用：只提供 schema、前几行 sample 和数值统计这些安全摘要。
        sections.append(f"## {table_name}\n\n{dataframe.head(8).to_string(index=False)}")
    return "\n\n".join(sections)


def _read_spreadsheet(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {"csv": pd.read_csv(path)}
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=None)
    raise ValueError(f"Unsupported spreadsheet type: {suffix}")


def _analyze_markdown_document(
    parsed: ParsedDocument,
    user_question: str = "",
) -> AnalysisResult:
    """Create a lightweight local summary for a Markdown-normalized document."""
    preview = parsed.markdown[:1200]
    summary_lines = [
        f"# Document Analysis: {parsed.source_path.name}",
        "",
        f"- Type: {parsed.metadata.get('source_type')}",
        f"- Characters: {parsed.metadata.get('characters')}",
        f"- Non-empty lines: {parsed.metadata.get('non_empty_lines')}",
        f"- Words: {parsed.metadata.get('words')}",
        "",
        "## Preview",
        preview if preview else "_No extractable text found._",
    ]

    if user_question.strip():
        summary_lines.extend(["", "## User Question", user_question.strip()])

    metadata_table = pd.DataFrame(
        [{"key": key, "value": value} for key, value in parsed.metadata.items()]
    )
    return AnalysisResult(
        markdown_summary="\n".join(summary_lines),
        tables={"metadata": metadata_table},
        metadata=dict(parsed.metadata),
        markdown_document=parsed.markdown,
    )
