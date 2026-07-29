"""Generate reports from one unified module."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer

from qwopus_agent.reports.charts import ChartRenderer

_REPORT_FONT_NAME = "QwopusReportUnicode"
_REPORT_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)


@dataclass(frozen=True)
class ReportArtifact:
    """One generated report artifact."""

    kind: str

    path: Path


@dataclass(frozen=True)
class GeneratedReport:
    """All artifacts created for one report request."""

    markdown: Path

    artifacts: list[ReportArtifact] = field(default_factory=list)


@dataclass
class ReportGenerator:
    """Generate Markdown, Excel, real chart, and PDF artifacts."""

    output_dir: Path = Path("storage/reports")

    def generate(
        self,
        title: str,
        markdown_body: str,
        tables: dict[str, pd.DataFrame] | None = None,
        basename: str = "report",
    ) -> GeneratedReport:
        """Generate all currently supported report artifacts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_basename = _safe_name(basename)
        tables = tables or {}

        markdown_path = self.output_dir / f"{safe_basename}.md"
        markdown_path.write_text(_markdown_document(title, markdown_body), encoding="utf-8")

        artifacts = [ReportArtifact(kind="markdown", path=markdown_path)]
        if tables:
            # 原因：报告模块要统一生成 Excel，而不是让 UI 或 Skill 各自写文件。
            # 作用：把所有 dataframe 收口写入同一个 workbook。
            artifacts.append(
                ReportArtifact(
                    kind="excel",
                    path=_write_excel(self.output_dir / f"{safe_basename}.xlsx", tables),
                )
            )

        # 原因：manifest 只能描述图表，用户无法查看或用于报告交付。
        # 作用：统一生成真实 PNG/SVG，并继续通过 ReportArtifact 交给 UI 下载。
        artifacts.extend(
            ReportArtifact(kind=chart.kind, path=chart.path)
            for chart in ChartRenderer(self.output_dir).render(tables, safe_basename)
        )
        artifacts.append(
            ReportArtifact(
                kind="pdf",
                path=_write_simple_pdf(
                    self.output_dir / f"{safe_basename}.pdf",
                    title,
                    markdown_body,
                ),
            )
        )

        return GeneratedReport(markdown=markdown_path, artifacts=artifacts)


def _markdown_document(title: str, body: str) -> str:
    """Build the canonical Markdown report body."""
    clean_title = title.strip() or "Qwopus Report"
    clean_body = body.strip() or "_No report content._"
    return f"# {clean_title}\n\n{clean_body}\n"


def _safe_name(name: str) -> str:
    """Convert user-controlled report names into local filenames."""
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return cleaned.strip("_") or "report"


def _write_excel(path: Path, tables: dict[str, pd.DataFrame]) -> Path:
    """Write report tables to one workbook."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for index, (name, dataframe) in enumerate(tables.items(), start=1):
            sheet_name = _excel_sheet_name(name, fallback=f"Sheet{index}")
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def _excel_sheet_name(name: str, fallback: str) -> str:
    """Return an Excel-safe sheet name."""
    cleaned = "".join("_" if char in "[]:*?/\\\"" else char for char in name).strip()
    return (cleaned or fallback)[:31]


def _write_simple_pdf(path: Path, title: str, body: str) -> Path:
    """Write the complete report as a paginated Unicode PDF."""
    font_name = _register_report_font()
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title.strip() or "Qwopus Report",
    )
    title_style = ParagraphStyle(
        "QwopusReportTitle",
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=10 * mm,
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "QwopusReportBody",
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        spaceAfter=2.5 * mm,
        wordWrap="CJK",
        splitLongWords=True,
    )
    story: list[Flowable] = [
        Paragraph(escape(title.strip() or "Qwopus Report"), title_style)
    ]
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 2.5 * mm))
            continue
        # 原因：报告正文来自 Markdown，直接交给 Paragraph 会把尖括号误解为 XML 标签。
        # 作用：转义用户内容并保留换行，让 Platypus 自动分页而不再截断正文。
        story.append(Paragraph(escape(stripped), body_style))
    if not body.strip():
        story.append(Paragraph("<i>No report content.</i>", body_style))

    document.build(story)
    return path


@lru_cache(maxsize=1)
def _register_report_font() -> str:
    """Register one broad Unicode font and return its ReportLab name."""
    for font_path in _REPORT_FONT_CANDIDATES:
        if font_path.is_file():
            # 原因：Helvetica 只支持单字节字符，中文、日文等内容会被替换成问号。
            # 作用：嵌入本机 Unicode TrueType 字体，使 PDF 可显示且可搜索原始文本。
            pdfmetrics.registerFont(TTFont(_REPORT_FONT_NAME, str(font_path)))
            return _REPORT_FONT_NAME

    # 原因：并非所有 Linux 部署都预装 Noto/Arial Unicode 字体。
    # 作用：使用 ReportLab 内置的简体中文 CID 字体保底，避免重新退回 latin-1。
    fallback_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(fallback_name))
    return fallback_name
