"""Safe local-folder discovery for direct document analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qwopus_agent.documents.parser import SUPPORTED_DOCUMENT_EXTENSIONS

SUPPORTED_LOCAL_FILE_EXTENSIONS = (
    SUPPORTED_DOCUMENT_EXTENSIONS | {".csv", ".xlsx", ".xls"}
)
MAX_LOCAL_FOLDER_FILES = 2_000
MAX_LOCAL_FOLDER_SELECTION = 100
MAX_LOCAL_FOLDER_ENTRIES = 20_000
MAX_LOCAL_FOLDER_DEPTH = 64


class LocalFolderError(ValueError):
    """Raised when a local folder or selection cannot be read safely."""


@dataclass(frozen=True)
class LocalFolderNode:
    """One directory or supported document in a local folder tree."""

    name: str
    relative_path: str
    kind: Literal["directory", "file"]
    children: tuple[LocalFolderNode, ...] = ()


@dataclass(frozen=True)
class LocalFolderTree:
    """Resolved folder root and its filtered document tree."""

    root: Path
    file_count: int
    tree: LocalFolderNode


def scan_local_folder(path: str | Path) -> LocalFolderTree:
    """Return a deterministic tree containing only supported local documents."""
    candidate = Path(path).expanduser()
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise LocalFolderError(f"Folder does not exist: {candidate}") from exc
    if not root.is_dir():
        raise LocalFolderError(f"Path is not a folder: {root}")

    # 原因：本地目录可能包含依赖缓存或数万文件，直接递归会阻塞 API 和浏览器。
    # 作用：只展示可分析文件，并用明确上限保护本地服务。
    counts = [0, 0]
    children = _scan_children(root, root, counts, depth=0)
    tree = LocalFolderNode(
        name=root.name,
        relative_path=".",
        kind="directory",
        children=children,
    )
    return LocalFolderTree(root=root, file_count=counts[0], tree=tree)


def resolve_selected_files(
    root: str | Path,
    relative_paths: list[str],
) -> tuple[Path, ...]:
    """Resolve user-selected relative paths without escaping the scanned root."""
    resolved_root = Path(root).expanduser().resolve(strict=True)
    if not resolved_root.is_dir():
        raise LocalFolderError(f"Path is not a folder: {resolved_root}")

    selected: list[Path] = []
    seen: set[Path] = set()
    for raw_path in relative_paths:
        relative = Path(raw_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise LocalFolderError(f"Invalid relative file path: {raw_path}")

        # 原因：resolve() 会隐藏符号链接跳转，单看最终路径无法确认用户实际选择了什么。
        # 作用：拒绝路径中的任意符号链接，避免目录树与实际读取目标不一致。
        current = resolved_root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise LocalFolderError(f"Symbolic links are not allowed: {raw_path}")

        try:
            resolved = current.resolve(strict=True)
        except OSError as exc:
            raise LocalFolderError(f"Selected file does not exist: {raw_path}") from exc
        if not resolved.is_relative_to(resolved_root):
            raise LocalFolderError(f"Selected file escapes folder root: {raw_path}")
        if not resolved.is_file():
            raise LocalFolderError(f"Selection is not a file: {raw_path}")
        if resolved.suffix.lower() not in SUPPORTED_LOCAL_FILE_EXTENSIONS:
            raise LocalFolderError(f"Unsupported file type: {raw_path}")
        if resolved not in seen:
            seen.add(resolved)
            selected.append(resolved)

    if not selected:
        raise LocalFolderError("Select at least one supported document.")
    return tuple(selected)


def _scan_children(
    directory: Path,
    root: Path,
    counts: list[int],
    *,
    depth: int,
) -> tuple[LocalFolderNode, ...]:
    if depth > MAX_LOCAL_FOLDER_DEPTH:
        raise LocalFolderError(
            f"Folder nesting exceeds {MAX_LOCAL_FOLDER_DEPTH} levels."
        )
    try:
        entries = sorted(
            directory.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
    except OSError as exc:
        raise LocalFolderError(f"Cannot read folder: {directory}") from exc

    nodes: list[LocalFolderNode] = []
    for entry in entries:
        counts[1] += 1
        if counts[1] > MAX_LOCAL_FOLDER_ENTRIES:
            # 原因：只限制支持文件数量无法约束包含大量空目录或无关文件的树。
            # 作用：扫描工作量由总访问节点数封顶，避免长时间遍历和递归资源耗尽。
            raise LocalFolderError(
                f"Folder contains more than {MAX_LOCAL_FOLDER_ENTRIES} entries."
            )
        # 原因：隐藏目录和符号链接常包含缓存、循环或根目录之外的内容。
        # 作用：目录分析保持可预测，并且不会越过用户明确选择的根目录。
        if entry.name.startswith(".") or entry.is_symlink():
            continue
        if entry.is_dir():
            children = _scan_children(entry, root, counts, depth=depth + 1)
            if children:
                nodes.append(
                    LocalFolderNode(
                        name=entry.name,
                        relative_path=entry.relative_to(root).as_posix(),
                        kind="directory",
                        children=children,
                    )
                )
            continue
        if not entry.is_file() or entry.suffix.lower() not in SUPPORTED_LOCAL_FILE_EXTENSIONS:
            continue

        counts[0] += 1
        if counts[0] > MAX_LOCAL_FOLDER_FILES:
            raise LocalFolderError(
                f"Folder contains more than {MAX_LOCAL_FOLDER_FILES} supported files."
            )
        nodes.append(
            LocalFolderNode(
                name=entry.name,
                relative_path=entry.relative_to(root).as_posix(),
                kind="file",
            )
        )
    return tuple(nodes)
