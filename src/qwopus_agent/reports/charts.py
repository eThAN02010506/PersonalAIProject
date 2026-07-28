"""Render report tables into real PNG and SVG chart files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

CHART_COLORS = ("#2563EB", "#E85D3F", "#169B62", "#D89B16", "#7C3AED")
MAX_CATEGORY_ROWS = 30
MAX_SERIES = 5


@dataclass(frozen=True)
class RenderedChart:
    """One rendered chart format."""

    kind: str
    path: Path
    table_name: str
    chart_type: str


@dataclass
class ChartRenderer:
    """Select a simple chart type and render it with a headless backend."""

    output_dir: Path

    def render(
        self,
        tables: dict[str, pd.DataFrame],
        basename: str,
    ) -> list[RenderedChart]:
        """Render PNG and SVG files for every chartable table."""
        charts: list[RenderedChart] = []
        for index, (table_name, dataframe) in enumerate(tables.items(), start=1):
            chart_base = f"{basename}_{index}_{_safe_name(table_name)}"
            rendered = self._render_table(dataframe, table_name, chart_base)
            charts.extend(rendered)
        return charts

    def _render_table(
        self,
        dataframe: pd.DataFrame,
        table_name: str,
        chart_base: str,
    ) -> list[RenderedChart]:
        if dataframe.empty:
            return []
        numeric_columns: list[Any] = list(
            dataframe.select_dtypes(include="number").columns[:MAX_SERIES]
        )
        if not numeric_columns:
            return []

        # 原因：Matplotlib 默认后端可能尝试打开 macOS 窗口，阻塞本地 API 服务。
        # 作用：按需加载 Agg 后端，只写入图像文件且不增加报告模块启动成本。
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "PingFang SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
        chart_type = self._draw(axis, dataframe, numeric_columns)
        axis.set_title(str(table_name), fontsize=14, pad=14)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "chart_png": self.output_dir / f"{chart_base}.png",
            "chart_svg": self.output_dir / f"{chart_base}.svg",
        }
        try:
            figure.savefig(paths["chart_png"], dpi=160, facecolor="white")
            figure.savefig(paths["chart_svg"], format="svg", facecolor="white")
        finally:
            plt.close(figure)
        return [
            RenderedChart(kind=kind, path=path, table_name=table_name, chart_type=chart_type)
            for kind, path in paths.items()
        ]

    def _draw(
        self,
        axis: Axes,
        dataframe: pd.DataFrame,
        numeric_columns: list[Any],
    ) -> str:
        category_column = next(
            (
                column
                for column in dataframe.columns
                if column not in numeric_columns
                and 0 < dataframe[column].nunique(dropna=True) <= MAX_CATEGORY_ROWS
            ),
            None,
        )
        datetime_column = next(
            (
                column
                for column in dataframe.columns
                if pd.api.types.is_datetime64_any_dtype(dataframe[column])
            ),
            None,
        )
        if datetime_column is not None:
            self._draw_lines(axis, dataframe, datetime_column, numeric_columns)
            return "line"
        if category_column is not None:
            self._draw_bars(axis, dataframe, category_column, numeric_columns)
            return "bar"
        if len(numeric_columns) == 1:
            values = pd.to_numeric(dataframe[numeric_columns[0]], errors="coerce").dropna()
            axis.hist(values, bins=min(12, max(3, len(values))), color=CHART_COLORS[0], alpha=0.85)
            axis.set_xlabel(str(numeric_columns[0]))
            axis.set_ylabel("Frequency")
            return "histogram"

        self._draw_lines(axis, dataframe, None, numeric_columns)
        return "line"

    def _draw_bars(
        self,
        axis: Axes,
        dataframe: pd.DataFrame,
        category_column: Any,
        numeric_columns: list[Any],
    ) -> None:
        frame = dataframe.head(MAX_CATEGORY_ROWS)
        positions = list(range(len(frame)))
        width = 0.8 / len(numeric_columns)
        for index, column in enumerate(numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            offsets = [position - 0.4 + width / 2 + index * width for position in positions]
            axis.bar(
                offsets,
                values,
                width=width,
                label=str(column),
                color=CHART_COLORS[index % len(CHART_COLORS)],
            )
        axis.set_xticks(positions, frame[category_column].astype(str), rotation=30, ha="right")
        axis.set_xlabel(str(category_column))
        axis.set_ylabel("Value")
        axis.legend(frameon=False)

    def _draw_lines(
        self,
        axis: Axes,
        dataframe: pd.DataFrame,
        x_column: Any | None,
        numeric_columns: list[Any],
    ) -> None:
        x_values = dataframe[x_column] if x_column is not None else list(range(len(dataframe)))
        for index, column in enumerate(numeric_columns):
            values = pd.to_numeric(dataframe[column], errors="coerce")
            axis.plot(
                x_values,
                values,
                marker="o",
                markersize=3.5,
                linewidth=2,
                label=str(column),
                color=CHART_COLORS[index % len(CHART_COLORS)],
            )
        axis.set_xlabel(str(x_column) if x_column is not None else "Row")
        axis.set_ylabel("Value")
        axis.legend(frameon=False)


def _safe_name(name: str) -> str:
    """Convert a table name into a stable chart filename component."""
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return cleaned.strip("_") or "table"
