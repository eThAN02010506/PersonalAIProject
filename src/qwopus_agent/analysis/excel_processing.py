"""Excel-specific loading helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qwopus_agent.analysis.workbook_profile import (
    TableRegionProfile,
    WorkbookProfile,
    inspect_workbook,
)

MAX_ANALYSIS_REGIONS_PER_SHEET = 12
MAX_FRAGMENTED_REGIONS_PER_SHEET = 24
MAX_FRAGMENTED_ANALYSIS_REGIONS_PER_SHEET = 4


@dataclass(frozen=True)
class SpreadsheetReadResult:
    """Loaded spreadsheet sheets plus read metadata."""

    sheets: dict[str, pd.DataFrame]

    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    form_summaries: dict[str, pd.DataFrame] = field(default_factory=dict)

    region_sheets: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)

    profile: WorkbookProfile | None = None

    def analysis_frames(self) -> dict[str, pd.DataFrame]:
        """Return every locally analyzable table under a stable unique name."""
        frames = dict(self.sheets)
        if self.profile is None:
            return frames
        for sheet_profile in self.profile.sheets:
            sheet_frames = self.region_sheets.get(sheet_profile.name, {})
            region_limit = (
                MAX_FRAGMENTED_ANALYSIS_REGIONS_PER_SHEET
                if len(sheet_profile.table_regions) > MAX_FRAGMENTED_REGIONS_PER_SHEET
                else MAX_ANALYSIS_REGIONS_PER_SHEET
            )
            secondary_regions = sorted(
                (
                    region
                    for region in sheet_profile.table_regions
                    if region.region_id != sheet_profile.primary_region_id
                    and region.non_empty_cells >= 4
                    and len(sheet_frames.get(region.region_id, ())) > 0
                    and len(
                        getattr(sheet_frames.get(region.region_id), "columns", ())
                    )
                    >= 2
                ),
                key=lambda region: (
                    region.non_empty_cells,
                    region.confidence,
                ),
                reverse=True,
            )[:region_limit]
            for region in secondary_regions:
                # 原因：一个工作表可能包含多张独立表，只暴露主区域会让 Agent 永远看不到其他数据。
                # 作用：保留主表名，并以 Sheet::table_N 暴露有数据的高信息次级区域，过滤说明碎片。
                frames[f"{sheet_profile.name}::{region.region_id}"] = sheet_frames[
                    region.region_id
                ]
        return frames


def read_spreadsheet(path: Path) -> SpreadsheetReadResult:
    """Read CSV or Excel with Excel-specific header detection."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return SpreadsheetReadResult(sheets={"csv": pd.read_csv(path)})
    if suffix == ".xlsx":
        return _read_xlsx_with_profiles(path)
    if suffix == ".xls":
        return SpreadsheetReadResult(sheets=pd.read_excel(path, sheet_name=None))
    raise ValueError(f"Unsupported spreadsheet type: {suffix}")


def _read_xlsx_with_profiles(path: Path) -> SpreadsheetReadResult:
    """Read XLSX sheets from an explicit workbook structure profile."""
    profile = inspect_workbook(path)
    raw_sheets = pd.read_excel(path, sheet_name=None, header=None)
    sheets: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    form_summaries: dict[str, pd.DataFrame] = {}
    region_sheets: dict[str, dict[str, pd.DataFrame]] = {}

    for sheet_profile in profile.sheets:
        raw = raw_sheets.get(sheet_profile.name, pd.DataFrame())
        form_summary = (
            _extract_form_summary(raw)
            if sheet_profile.kind == "form"
            else pd.DataFrame()
        )
        frames = {
            region.region_id: _dataframe_from_region(
                raw,
                region,
                use_headers=sheet_profile.kind != "form",
            )
            for region in sheet_profile.table_regions
        }
        primary = sheet_profile.primary_region()
        dataframe = (
            _generic_dataframe(raw)
            if len(sheet_profile.table_regions) > MAX_FRAGMENTED_REGIONS_PER_SHEET
            else (
                frames[primary.region_id]
                if primary is not None
                else _generic_dataframe(raw)
            )
        )
        sheets[sheet_profile.name] = dataframe
        region_sheets[sheet_profile.name] = frames
        form_summaries[sheet_profile.name] = form_summary
        profile_data = sheet_profile.model_dump(mode="json")
        metadata[sheet_profile.name] = {
            "header_row": (
                primary.header_rows[-1]
                if primary is not None and primary.header_rows
                else None
            ),
            "header_rows": list(primary.header_rows) if primary is not None else [],
            "header_detection": "workbook_profile",
            "primary_strategy": (
                "full_sheet_fallback"
                if len(sheet_profile.table_regions) > MAX_FRAGMENTED_REGIONS_PER_SHEET
                else "detected_region"
            ),
            "sheet_kind": sheet_profile.kind,
            "form_pairs": int(len(form_summary)),
            "table_regions": profile_data["table_regions"],
            "primary_region_id": sheet_profile.primary_region_id,
            "formula_count": sheet_profile.formula_count,
            "broken_formula_reference_count": (
                sheet_profile.broken_formula_reference_count
            ),
            "merged_range_count": sheet_profile.merged_range_count,
            "chart_count": sheet_profile.chart_count,
            "image_count": sheet_profile.image_count,
            "data_validation_count": sheet_profile.data_validation_count,
            "profile_truncated": sheet_profile.profile_truncated,
        }

    return SpreadsheetReadResult(
        sheets=sheets,
        metadata=metadata,
        form_summaries=form_summaries,
        region_sheets=region_sheets,
        profile=profile,
    )


def _normalize_cell(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    return str(value).strip()


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _extract_form_summary(
    dataframe: pd.DataFrame,
    max_rows: int = 200,
    max_columns: int = 80,
    max_pairs: int = 120,
) -> pd.DataFrame:
    """Extract key-value pairs from form-like sheets."""
    rows = [
        tuple(row)
        for row in dataframe.iloc[:max_rows, :max_columns].itertuples(
            index=False,
            name=None,
        )
    ]
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


def _neighbor_values(
    rows: list[tuple[Any, ...]],
    row_index: int,
    column_index: int,
) -> Iterator[tuple[str, Any]]:
    row = rows[row_index]
    for offset in range(1, 4):
        right_index = column_index + offset
        if right_index >= len(row):
            break
        if _normalize_cell(row[right_index]):
            # 原因：合并标签在 pandas 中会留下一个或多个 NaN 占位列。
            # 作用：跨过这些空列找到同一行的实际表单值，同时限制搜索半径避免串到远处字段。
            direction = "right" if offset == 1 else f"right+{offset}"
            yield direction, row[right_index]
            break
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
    # 原因：Excel 多行表头会把换行带入真实列名，而 Markdown schema 会把它显示为空格。
    # 作用：让 Agent 从 schema 复制的列名可直接用于沙箱计算，避免同列产生两种表示。
    dataframe.columns = [
        " ".join(str(column).split())
        for column in dataframe.columns
    ]
    # 原因：header=None 会让包含表头文字的原始列先变成 object，切掉表头后不会自动恢复数字类型。
    # 作用：在区域裁剪完成后重新推断 int/float/datetime，保证统计和 pandas 沙箱按真实类型运行。
    return dataframe.infer_objects()


def _dataframe_from_region(
    raw: pd.DataFrame,
    region: TableRegionProfile,
    *,
    use_headers: bool,
) -> pd.DataFrame:
    window = raw.iloc[
        region.min_row - 1 : region.max_row,
        region.min_column - 1 : region.max_column,
    ].copy()
    if use_headers and region.header_rows and region.data_start_row is not None:
        data_offset = max(0, region.data_start_row - region.min_row)
        window = window.iloc[data_offset:].copy()
    window.columns = list(region.column_names)
    return _clean_loaded_dataframe(window)


def _generic_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    dataframe = raw.copy()
    dataframe.columns = [
        f"column_{index}" for index in range(1, len(dataframe.columns) + 1)
    ]
    return _clean_loaded_dataframe(dataframe)
