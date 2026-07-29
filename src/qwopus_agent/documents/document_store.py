"""Persistent source-of-truth and derived structure storage for parsed documents."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwopus_agent.documents.models import DocumentStructure
from qwopus_agent.documents.summarizer import HierarchicalDocumentSummary

DEFAULT_DOCUMENT_STORE = Path("storage/documents")
_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class StoredDocumentError(Exception):
    """Base class for unavailable or malformed persisted document records."""


class InvalidDocumentIdError(ValueError, StoredDocumentError):
    """Raised before an unsafe document id can participate in path construction."""


class StoredDocumentNotFoundError(FileNotFoundError, StoredDocumentError):
    """Raised when no complete saved record exists for a valid document id."""


class CorruptStoredDocumentError(StoredDocumentError):
    """Raised when a saved record violates the persistence contract."""


@dataclass(frozen=True)
class StoredDocument:
    """Inventory metadata for one complete locally persisted document."""

    document_id: str
    source: str
    file_type: str
    size_bytes: int
    section_count: int
    saved_at: str
    summary_available: bool


@dataclass(frozen=True)
class StoredDocumentContent:
    """Validated saved-document files used by attach and direct-analysis boundaries."""

    document: StoredDocument
    metadata: dict[str, Any]
    normalized_markdown: str
    normalized_path: Path
    original_path: Path
    structure: DocumentStructure


@dataclass(frozen=True)
class DocumentStore:
    """Persist originals and rebuildable document derivatives by document id."""

    root: Path = DEFAULT_DOCUMENT_STORE

    def validate_document_id(self, document_id: str) -> None:
        """Validate identifier syntax without reading or confirming a stored record."""
        self._document_directory(document_id)

    def persist(
        self,
        *,
        original_path: Path,
        markdown: str,
        structure: DocumentStructure,
        metadata: dict[str, Any],
    ) -> Path:
        directory = self._document_directory(structure.document_id)
        directory.mkdir(parents=True, exist_ok=True)
        original_target = directory / f"original{original_path.suffix.lower()}"
        if original_path.resolve() != original_target.resolve():
            shutil.copy2(original_path, original_target)

        # 原因：原文是事实来源，结构、Chunk 和摘要都必须能够独立重建。
        # 作用：按文档版本保存规范化 Markdown 与显式派生文件，避免只剩向量索引。
        _atomic_write(directory / "normalized.md", markdown)
        _atomic_write(
            directory / "metadata.json",
            json.dumps(
                {
                    **metadata,
                    "document_id": structure.document_id,
                    "source": structure.source,
                    "original_path": str(original_target),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _atomic_write(
            directory / "structure.json",
            structure.model_dump_json(indent=2),
        )
        _atomic_write(
            directory / "chunks.jsonl",
            "\n".join(chunk.model_dump_json() for chunk in structure.chunks),
        )
        return directory

    def load_structure(self, document_id: str) -> DocumentStructure:
        return self.load_document(document_id).structure

    def load_metadata(self, document_id: str) -> dict[str, Any]:
        """Read a copy of metadata from a complete, confined record."""
        return dict(self.load_document(document_id).metadata)

    def load_normalized(self, document_id: str) -> str:
        """Read normalized Markdown only from a complete, confined record."""
        return self.load_document(document_id).normalized_markdown

    def load_original_path(self, document_id: str) -> Path:
        """Return the confined original file path for direct document analysis."""
        return self.load_document(document_id).original_path

    def load_document(self, document_id: str) -> StoredDocumentContent:
        """Load one complete record while rejecting traversal, symlinks, and id drift."""
        directory = self._document_directory(document_id, require_existing=True)
        metadata_path = _confined_file(directory, "metadata.json")
        normalized_path = _confined_file(directory, "normalized.md")
        structure_path = _confined_file(directory, "structure.json")
        metadata = _read_object(metadata_path)
        try:
            structure = DocumentStructure.model_validate_json(
                structure_path.read_text(encoding="utf-8")
            )
            normalized_markdown = normalized_path.read_text(encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            raise CorruptStoredDocumentError(
                f"Saved document is malformed: {document_id}"
            ) from exc

        persisted_id = metadata.get("document_id")
        if (
            not isinstance(persisted_id, str)
            or persisted_id != document_id
            or structure.document_id != document_id
        ):
            raise CorruptStoredDocumentError(
                f"Saved document id does not match its directory: {document_id}"
            )

        source = metadata.get("source")
        if not isinstance(source, str) or not source.strip():
            source = structure.source
        if source != structure.source:
            raise CorruptStoredDocumentError(
                f"Saved document source metadata is inconsistent: {document_id}"
            )
        original = _resolve_original(directory, metadata)
        saved_timestamp = max(
            metadata_path.stat().st_mtime,
            structure_path.stat().st_mtime,
            normalized_path.stat().st_mtime,
            original.stat().st_mtime,
        )
        inventory = StoredDocument(
            document_id=document_id,
            source=source,
            file_type=original.suffix.lower().lstrip("."),
            size_bytes=original.stat().st_size,
            section_count=len(structure.sections),
            saved_at=datetime.fromtimestamp(saved_timestamp, tz=UTC).isoformat(),
            summary_available=(directory / "document_summary.md").is_file(),
        )
        return StoredDocumentContent(
            document=inventory,
            metadata=metadata,
            normalized_markdown=normalized_markdown,
            normalized_path=normalized_path,
            original_path=original,
            structure=structure,
        )

    def list_documents(self) -> list[StoredDocument]:
        """List complete records while ignoring interrupted or malformed writes."""
        documents: list[StoredDocument] = []
        if not self.root.is_dir():
            return documents

        for directory in self.root.glob("document-*"):
            try:
                documents.append(self.load_document(directory.name).document)
            except (OSError, StoredDocumentError):
                continue

        return sorted(documents, key=lambda item: item.saved_at, reverse=True)

    def persist_summary(self, summary: HierarchicalDocumentSummary) -> None:
        directory = self._document_directory(summary.document_id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            directory / "section_summaries.json",
            json.dumps(
                [
                    section.model_dump(mode="json")
                    for section in summary.section_summaries
                ],
                ensure_ascii=False,
                indent=2,
            ),
        )
        _atomic_write(directory / "document_summary.md", summary.document_summary)

    def _document_directory(
        self,
        document_id: str,
        *,
        require_existing: bool = False,
    ) -> Path:
        """Resolve exactly one direct child of the configured document root."""
        normalized_id = document_id.strip()
        if (
            normalized_id != document_id
            or not _DOCUMENT_ID_PATTERN.fullmatch(normalized_id)
        ):
            raise InvalidDocumentIdError(
                "document_id contains unsupported characters"
            )
        root = self.root.resolve()
        directory = root / normalized_id
        resolved = directory.resolve()
        if resolved.parent != root or directory.is_symlink():
            raise InvalidDocumentIdError("document_id escapes the document store")
        if require_existing and not directory.is_dir():
            raise StoredDocumentNotFoundError(
                f"Saved document not found: {document_id}"
            )
        return directory


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path.name} must contain an object")
        return payload
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise CorruptStoredDocumentError(
            f"Saved document metadata is malformed: {path.parent.name}"
        ) from exc


def _confined_file(directory: Path, name: str) -> Path:
    """Require one regular, non-symlinked file directly inside a saved record."""
    path = directory / name
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CorruptStoredDocumentError(
            f"Saved document is incomplete: {directory.name}"
        ) from exc
    if path.is_symlink() or resolved.parent != directory.resolve() or not resolved.is_file():
        raise CorruptStoredDocumentError(
            f"Saved document contains an unsafe file: {directory.name}"
        )
    return resolved


def _resolve_original(directory: Path, metadata: dict[str, Any]) -> Path:
    """Resolve the persisted original by basename and never trust its absolute path."""
    raw_original = metadata.get("original_path")
    if isinstance(raw_original, str) and raw_original.strip():
        basename = Path(raw_original).name
        if basename.startswith("original."):
            candidate = directory / basename
            if candidate.exists():
                return _confined_file(directory, basename)

    candidates = sorted(
        path.name
        for path in directory.glob("original.*")
        if path.is_file() and not path.is_symlink()
    )
    if not candidates:
        raise CorruptStoredDocumentError(
            f"Saved document has no original file: {directory.name}"
        )
    return _confined_file(directory, candidates[0])


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
