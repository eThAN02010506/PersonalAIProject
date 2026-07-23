"""Deterministic heading and page extraction from normalized Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path

from qwopus_agent.documents.models import DocumentSection, DocumentStructure

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE_HEADING = re.compile(r"^(?:Page\s+(\d+)|第\s*(\d+)\s*页)$", re.IGNORECASE)
_NUMBERED_HEADING = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[章节篇部]|"
    r"[一二三四五六七八九十百千万]+[.、\s]+|"
    r"\d+(?:\.\d+){0,5}(?:[.\s、]+))\s*\S.{0,100}$"
)


@dataclass
class _SectionBuilder:
    """Mutable parser state converted to an immutable public model at the boundary."""

    id: str
    title: str
    level: int
    parent_id: str | None
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    lines: list[str] = field(default_factory=list)


def build_document_structure(
    markdown: str,
    *,
    source: str,
    document_id: str | None = None,
    infer_plaintext_headings: bool | None = None,
) -> DocumentStructure:
    """Build a section tree while keeping pages independent from heading levels."""
    resolved_id = document_id or _stable_id("document", f"{source}\n{markdown}")
    infer_plaintext = (
        Path(source).suffix.lower() in {".docx", ".txt"}
        if infer_plaintext_headings is None
        else infer_plaintext_headings
    )
    require_bold_numbered = Path(source).suffix.lower() == ".docx"
    sections: list[_SectionBuilder] = []
    heading_stack: list[_SectionBuilder] = []
    current: _SectionBuilder | None = None
    current_page: int | None = None

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        heading = _heading(
            line,
            infer_plaintext=infer_plaintext,
            require_bold_numbered=require_bold_numbered,
        )
        if heading is not None:
            level, title = heading
            page = _page_number(title)
            if page is not None:
                current_page = page
                if current is not None:
                    current.page_end = page
                continue

            while heading_stack and heading_stack[-1].level >= level:
                heading_stack.pop()
            parent = heading_stack[-1] if heading_stack else None
            section_path = (*parent.section_path, title) if parent else (title,)
            current = _SectionBuilder(
                id=_stable_id(
                    "section",
                    f"{resolved_id}\n{'/'.join(section_path)}\n{len(sections)}",
                ),
                title=title,
                level=level,
                parent_id=parent.id if parent else None,
                section_path=section_path,
                page_start=current_page,
                page_end=current_page,
            )
            sections.append(current)
            heading_stack.append(current)
            continue

        if not line.strip() and current is None:
            continue
        if current is None:
            # 原因：部分文档在第一个标题前有摘要、封面说明或根本没有标题。
            # 作用：为前置正文建立稳定根节点，避免内容在结构化阶段丢失。
            current = _SectionBuilder(
                id=_stable_id("section", f"{resolved_id}\npreamble"),
                title="Document content",
                level=0,
                parent_id=None,
                section_path=("Document content",),
                page_start=current_page,
                page_end=current_page,
            )
            sections.append(current)
        current.lines.append(line)
        if current_page is not None:
            current.page_end = current_page

    return DocumentStructure(
        document_id=resolved_id,
        source=source,
        sections=tuple(
            DocumentSection(
                id=section.id,
                title=section.title,
                level=section.level,
                parent_id=section.parent_id,
                section_path=section.section_path,
                page_start=section.page_start,
                page_end=section.page_end,
                content="\n".join(section.lines).strip(),
            )
            for section in sections
        ),
    )


def _heading(
    line: str,
    *,
    infer_plaintext: bool,
    require_bold_numbered: bool,
) -> tuple[int, str] | None:
    markdown_match = _MARKDOWN_HEADING.match(line.strip())
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()
    compact = line.strip()
    is_bold_line = compact.startswith("**") and compact.endswith("**")
    if is_bold_line:
        compact = compact[2:-2].strip()
    # 原因：MinerU 会保留标题前的图标字符，编号本身仍是最稳定的章节信号。
    # 作用：只在启用编号推断时忽略编号前符号，不影响普通 Markdown 标题正文。
    compact = re.sub(r"^[^\w一二三四五六七八九十百千万]+", "", compact)
    if (
        infer_plaintext
        and (is_bold_line or not require_bold_numbered)
        and len(compact) <= 120
        and _NUMBERED_HEADING.match(compact)
    ):
        return _numbered_heading_level(compact), compact
    return None


def _numbered_heading_level(title: str) -> int:
    number = re.match(r"^(\d+(?:\.\d+)*)", title)
    if number:
        return min(number.group(1).count(".") + 1, 6)
    return 1


def _page_number(title: str) -> int | None:
    match = _PAGE_HEADING.match(title.strip())
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _stable_id(kind: str, value: str) -> str:
    digest = blake2b(value.encode("utf-8"), digest_size=12).hexdigest()
    return f"{kind}-{digest}"
