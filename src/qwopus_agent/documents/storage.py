"""Upload storage helpers.

The UI should only save uploaded bytes and pass a file path to services. Keeping this logic here
prevents Streamlit code from becoming business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class StoredUpload:
    """Metadata for one saved upload."""

    # 原因：后续分析、报告和 MiniRAG 都需要稳定的本地文件路径。
    # 作用：指向 storage/uploads 下的实际文件。
    path: Path

    # 原因：UI 展示应该保留用户上传时的原始名称。
    # 作用：用于页面显示和报告标题。
    original_name: str


def save_uploaded_bytes(
    filename: str,
    content: bytes,
    upload_dir: Path = Path("storage/uploads"),
) -> StoredUpload:
    """Persist uploaded bytes into the project upload directory."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name

    # 原因：给文件名前加 uuid，避免同名上传互相覆盖。
    # 作用：生成可重复分析的本地沙箱路径。
    stored_path = upload_dir / f"{uuid4().hex}_{safe_name}"
    stored_path.write_bytes(content)
    return StoredUpload(path=stored_path, original_name=safe_name)
