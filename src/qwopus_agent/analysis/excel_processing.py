"""Excel-specific loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook


@dataclass(frozen=True)
class SpreadsheetReadResult:
    """Loaded spreadsheet sheets plus read metadata."""

    sheets: dict[str, pd.DataFrame]

    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    form_summaries: dict[str, pd.DataFrame] = field(default_factory=dict)


def read_spreadsheet(path: Path) -> SpreadsheetReadResult:
    """Read CSV or Excel with Excel-specific header detection."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return SpreadsheetReadResult(sheets={"csv": pd.read_csv(path)})
    if suffix == ".xlsx":
        return _read_xlsx_with_detected_headers(path)
    if suffix == ".xls":
        return SpreadsheetReadResult(sheets=pd.read_excel(path, sheet_name=None))
    raise ValueError(f"Unsupported spreadsheet type: {suffix}")


def _read_xlsx_with_detected_headers(path: Path) -> SpreadsheetReadResult:
    """Read XLSX sheets after detecting the likely header row."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheets: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    form_summaries: dict[str, pd.DataFrame] = {}

    try:
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            header_row = _detect_header_row(worksheet)
            form_summary = _extract_form_summary(worksheet)
            # 原因：很多 Excel 文件前几行是标题、说明或空行，pandas 默认 header=0 会误判字段。
            # 作用：先用 openpyxl 找真实表头，再交给 pandas 做类型推断和后续分析。
            dataframe = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
            dataframe = _clean_loaded_dataframe(dataframe)
            sheets[sheet_name] = dataframe
            form_summaries[sheet_name] = form_summary
            metadata[sheet_name] = {
                "header_row": header_row + 1,
                "header_detection": "openpyxl",
                "form_pairs": int(len(form_summary)),
            }
    finally:
        workbook.close()

    return SpreadsheetReadResult(
        sheets=sheets,
        metadata=metadata,
        form_summaries=form_summaries,
    )


def _detect_header_row(worksheet, max_scan_rows: int = 25) -> int:
    """Detect the most likely 0-based header row index."""
    rows = list(
        worksheet.iter_rows(
            min_row=1,
            max_row=min(max_scan_rows, worksheet.max_row),
            values_only=True,
        )
    )
    best_index = 0
    best_score = -1
    for index, row in enumerate(rows):
        next_row = rows[index + 1] if index + 1 < len(rows) else ()
        score = _header_score(row, next_row)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _header_score(row: tuple[Any, ...], next_row: tuple[Any, ...]) -> int:
    values = [_normalize_cell(value) for value in row]
    non_empty_values = [value for value in values if value]
    if len(non_empty_values) < 2:
        return -1

    unique_count = len(set(non_empty_values))
    text_count = sum(1 for value in non_empty_values if not _looks_numeric(value))
    next_non_empty_count = sum(1 for value in next_row if _normalize_cell(value))
    if text_count == 0:
        return -1

    # 原因：真实表头通常有多个非空、文本型、相对唯一的字段，并且下一行有数据。
    # 作用：用轻量启发式避开“报表标题”“日期说明”等单行描述。
    return len(non_empty_values) * 2 + unique_count + text_count * 2 + next_non_empty_count


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _extract_form_summary(
        worksheet,
        max_rows: int = 200,
        max_columns: int = 80,
        max_pairs: int = 120,
) -> pd.DataFrame:
    """Extract key-value pairs from form-like sheets."""
    rows = list(
        worksheet.iter_rows(
            min_row=1,
            max_row=min(max_rows, worksheet.max_row),
            max_col=min(max_columns, worksheet.max_column),
            values_only=True,
        )
    )
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            label = _normalize_cell(value)
            if not _looks_like_label(label):
                continue

            for direction, neighbor in _neighbor_values(rows, row_index, column_index):
                normalized_value = _normalize_cell(neighbor)
                if not normalized_value or normalized_value == label:
                    continue
                identity = (label, normalized_value, direction)
                if identity in seen:
                    continue
                seen.add(identity)
                # 原因：角色卡、申请表这类 Excel 的信息散落在单元格中，不适合强行按二维表头解析。
                # 作用：抽取“标签-值”对，让 LLM 能看到姓名、职业、年龄等表单信息。
                pairs.append(
                    {
                        "key": label,
                        "value": normalized_value,
                        "row": row_index + 1,
                        "column": column_index + 1,
                        "direction": direction,
                    }
                )
                break

            if len(pairs) >= max_pairs:
                return pd.DataFrame(pairs)

    return pd.DataFrame(pairs)


def _neighbor_values(rows: list[tuple[Any, ...]], row_index: int, column_index: int):
    row = rows[row_index]
    if column_index + 1 < len(row):
        yield "right", row[column_index + 1]
    if row_index + 1 < len(rows):
        next_row = rows[row_index + 1]
        if column_index < len(next_row):
            yield "below", next_row[column_index]


def _looks_like_label(value: str) -> bool:
    if not value or _looks_numeric(value):
        return False
    if len(value) > 40:
        return False
    return any(char.isalpha() or "\u4e00" <= char <= "\u9fff" for char in value)


def _clean_loaded_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.dropna(axis=0, how="all").dropna(axis=1, how="all")
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]
    return dataframe
