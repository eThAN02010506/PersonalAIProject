"""Persistent source-of-truth and derived structure storage for parsed documents."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwopus_agent.documents.models import DocumentStructure
from qwopus_agent.documents.summarizer import HierarchicalDocumentSummary

DEFAULT_DOCUMENT_STORE = Path("storage/documents")


@dataclass(frozen=True)
class DocumentStore:
    """Persist originals and rebuildable document derivatives by document id."""

    root: Path = DEFAULT_DOCUMENT_STORE

    def persist(
        self,
        *,
        original_path: Path,
        markdown: str,
        structure: DocumentStructure,
        metadata: dict[str, Any],
    ) -> Path:
        directory = self.root / structure.document_id
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
        path = self.root / Path(document_id).name / "structure.json"
        return DocumentStructure.model_validate_json(path.read_text(encoding="utf-8"))

    def persist_summary(self, summary: HierarchicalDocumentSummary) -> None:
        directory = self.root / summary.document_id
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


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
