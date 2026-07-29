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
    r"[一二三四五六七八九十百千万]+[.．、\s]+|"
    r"\d+(?:\.\d+){0,5}(?:[.．\s、]+))\s*\S.{0,100}$"
)
_STRONG_PLAIN_HEADING = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百千万0-9]+[章节篇部][：:、.．\s]+\S|"
    r"[一二三四五六七八九十百千万]+[.．、]\s*\S"
    r").{0,100}$"
)
_SINGLE_ARABIC_HEADING = re.compile(
    r"^(?P<number>\d{1,3})[.．、]\s*(?P<title>\S.{0,100})$"
)
_DECIMAL_ARABIC_HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+)+)(?:[.．、]\s*|\s+)(?P<title>\S.{0,100})$"
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
    raw_lines = markdown.splitlines()
    plain_heading_indexes = (
        _infer_docx_plain_heading_indexes(raw_lines)
        if infer_plaintext and require_bold_numbered
        else set()
    )
    sections: list[_SectionBuilder] = []
    heading_stack: list[_SectionBuilder] = []
    current: _SectionBuilder | None = None
    current_page: int | None = None

    for line_index, raw_line in enumerate(raw_lines):
        line = raw_line.rstrip()
        heading = _heading(
            line,
            infer_plaintext=infer_plaintext,
            require_bold_numbered=require_bold_numbered,
            allow_plain_numbered=line_index in plain_heading_indexes,
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
    allow_plain_numbered: bool = False,
) -> tuple[int, str] | None:
    markdown_match = _MARKDOWN_HEADING.match(line.strip())
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()
    compact, is_bold_line = _plain_heading_text(line)
    if (
        infer_plaintext
        and (is_bold_line or not require_bold_numbered or allow_plain_numbered)
        and len(compact) <= 120
        and _NUMBERED_HEADING.match(compact)
    ):
        return _numbered_heading_level(compact), compact
    return None


def _numbered_heading_level(title: str) -> int:
    number = re.match(r"^(\d+(?:\.\d+)*)", title)
    if number:
        # 原因：中文大纲常用“一、主题”作为一级，再用“1. 子题”作为二级；
        # 旧逻辑把单个阿拉伯数字也定为一级，导致子题与父级互相覆盖。
        # 作用：单个阿拉伯编号至少为二级，小数点层数继续表达更深层级。
        return min(max(2, number.group(1).count(".") + 1), 6)
    return 1


def _plain_heading_text(line: str) -> tuple[str, bool]:
    """Normalize one possible visual heading while retaining its bold signal."""
    compact = line.strip()
    is_bold_line = compact.startswith("**") and compact.endswith("**")
    if is_bold_line:
        compact = compact[2:-2].strip()
    # 原因：MinerU 会保留标题前的图标字符，编号本身仍是最稳定的章节信号。
    # 作用：只移除编号前装饰符，不触碰标题正文。
    compact = re.sub(r"^[^\w一二三四五六七八九十百千万]+", "", compact)
    return compact, is_bold_line


def _infer_docx_plain_heading_indexes(lines: list[str]) -> set[int]:
    """Conservatively recover unstyled numbered headings from DOCX text.

    Word documents frequently store visual headings as ordinary ``Normal`` paragraphs, and
    MinerU therefore emits neither ``#`` nor ``**``. Chinese chapter markers are strong enough
    to accept directly. Ambiguous Arabic markers are accepted only as a sequential outline with
    substantive body text beneath at least two siblings, which keeps ordinary numbered lists in
    their parent section.
    """
    allowed: set[int] = set()
    boundaries: list[int] = []
    for index, line in enumerate(lines):
        compact, is_bold = _plain_heading_text(line)
        if (
            len(compact) <= 120
            and _NUMBERED_HEADING.match(compact)
            and (is_bold or _STRONG_PLAIN_HEADING.match(compact))
        ):
            boundaries.append(index)
            if _STRONG_PLAIN_HEADING.match(compact):
                allowed.add(index)

    for boundary_index, start in enumerate(boundaries):
        end = (
            boundaries[boundary_index + 1]
            if boundary_index + 1 < len(boundaries)
            else len(lines)
        )
        candidates: list[tuple[int, tuple[int, ...]]] = []
        for index in range(start + 1, end):
            compact, is_bold = _plain_heading_text(lines[index])
            if is_bold or len(compact) > 120:
                continue
            decimal = _DECIMAL_ARABIC_HEADING.match(compact)
            if decimal is not None:
                number_path = tuple(
                    int(part) for part in decimal.group("number").split(".")
                )
                candidates.append((index, number_path))
                continue
            single = _SINGLE_ARABIC_HEADING.match(compact)
            if single is not None:
                candidates.append((index, (int(single.group("number")),)))

        for number_depth in sorted({len(path) for _, path in candidates}):
            by_parent: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
            for candidate in candidates:
                index, number_path = candidate
                if len(number_path) != number_depth:
                    continue
                by_parent.setdefault(number_path[:-1], []).append((index, number_path))
            for siblings in by_parent.values():
                siblings.sort(key=lambda item: item[0])
                ordinals = [path[-1] for _, path in siblings]
                if len(siblings) < 2 or ordinals[0] != 1 or ordinals != list(
                    range(ordinals[0], ordinals[0] + len(ordinals))
                ):
                    continue
                substantive = 0
                for sibling_index, (line_index, _) in enumerate(siblings):
                    body_end = (
                        siblings[sibling_index + 1][0]
                        if sibling_index + 1 < len(siblings)
                        else end
                    )
                    if _has_substantive_body(lines, line_index + 1, body_end):
                        substantive += 1
                if substantive >= 2:
                    allowed.update(index for index, _ in siblings)
    return allowed


def _has_substantive_body(lines: list[str], start: int, end: int) -> bool:
    body = "".join(line.strip() for line in lines[start:end] if line.strip())
    return len(body) >= 16


def _page_number(title: str) -> int | None:
    match = _PAGE_HEADING.match(title.strip())
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _stable_id(kind: str, value: str) -> str:
    digest = blake2b(value.encode("utf-8"), digest_size=12).hexdigest()
    return f"{kind}-{digest}"
