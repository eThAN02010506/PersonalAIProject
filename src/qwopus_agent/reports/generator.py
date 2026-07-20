"""Generate reports from one unified module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from qwopus_agent.reports.charts import ChartRenderer


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
    """Write a minimal PDF without introducing a new runtime dependency."""
    content = _escape_pdf_text(f"{title}\n\n{body}"[:3500])
    stream = f"BT /F1 12 Tf 72 760 Td 14 TL ({content}) Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        (
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
        ),
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        (
            "5 0 obj << /Length "
            f"{len(stream.encode('latin-1', errors='replace'))} >> stream\n"
            f"{stream}\nendstream endobj"
        ),
    ]
    offsets: list[int] = []
    pdf = "%PDF-1.4\n"
    for obj in objects:
        offsets.append(len(pdf.encode("latin-1", errors="replace")))
        pdf += obj + "\n"
    xref_start = len(pdf.encode("latin-1", errors="replace"))
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    path.write_bytes(pdf.encode("latin-1", errors="replace"))
    return path


def _escape_pdf_text(text: str) -> str:
    """Escape text for a simple PDF text operator."""
    ascii_text = text.encode("latin-1", errors="replace").decode("latin-1")
    return (
        ascii_text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", "\\n")
    )
