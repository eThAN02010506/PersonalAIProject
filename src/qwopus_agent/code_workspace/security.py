"""Fail-closed path, file, and repository validation for source workspaces."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath

from qwopus_agent.code_workspace.models import (
    CodeFileView,
    CodeSearchMatch,
    CodeTreeNode,
    CodeWorkspaceTree,
)

MAX_CODE_FILES = 3000
MAX_CODE_DEPTH = 16
MAX_CODE_FILE_BYTES = 512 * 1024
MAX_READ_LINES = 600
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_QUERY = 200
CODE_SUFFIXES = {
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".pyi",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
CODE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Procfile",
    "README",
}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "logs",
    "models",
    "node_modules",
    "storage",
    "vendor",
}
SENSITIVE_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
    "secrets.toml",
    "secrets.yaml",
    "secrets.yml",
}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class CodeWorkspaceError(ValueError):
    """Raised when a code-workspace operation violates a safety boundary."""


def resolve_git_workspace(path: str | Path) -> Path:
    """Resolve a user-selected path and require it to be the Git repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise CodeWorkspaceError("Workspace root cannot be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CodeWorkspaceError("Workspace path does not exist.") from exc
    if not resolved.is_dir():
        raise CodeWorkspaceError("Workspace path must be a directory.")
    result = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        raise CodeWorkspaceError("Workspace must be a Git repository.")
    git_root = Path(result.stdout.strip()).resolve(strict=True)
    if git_root != resolved:
        # 原因：接受仓库子目录会让 UI 展示的范围与后端实际 Git 根目录不一致。
        # 作用：管理员必须明确选择完整仓库，避免无意读取父目录中的源码。
        raise CodeWorkspaceError(f"Select the Git repository root: {git_root}")
    return resolved


def scan_code_workspace(path: str | Path) -> CodeWorkspaceTree:
    """Return a filtered source tree without reading file contents."""
    root = resolve_git_workspace(path)
    file_counter = [0]
    tree = _scan_directory(root, root, 0, file_counter)
    return CodeWorkspaceTree(root=str(root), file_count=file_counter[0], tree=tree)


def read_code_file(
    root_path: str | Path,
    relative_path: str,
    *,
    start_line: int = 1,
    end_line: int = MAX_READ_LINES,
) -> CodeFileView:
    """Read a bounded line range from one validated UTF-8 source file."""
    root = resolve_git_workspace(root_path)
    path = resolve_code_file(root, relative_path)
    content = _read_utf8(path)
    lines = content.splitlines(keepends=True)
    total_lines = max(1, len(lines))
    bounded_start = max(1, start_line)
    bounded_end = min(max(bounded_start, end_line), bounded_start + MAX_READ_LINES - 1)
    selected = "".join(lines[bounded_start - 1 : bounded_end])
    return CodeFileView(
        root=str(root),
        path=relative_path,
        sha256=sha256_text(content),
        content=selected,
        total_lines=total_lines,
        start_line=bounded_start,
        end_line=min(bounded_end, total_lines),
    )


def read_complete_code_file(root: Path, relative_path: str) -> str:
    """Read one complete validated file for a bounded model proposal."""
    return _read_utf8(resolve_code_file(root, relative_path))


def search_code_workspace(
    root_path: str | Path,
    query: str,
    *,
    limit: int = 100,
) -> list[CodeSearchMatch]:
    """Perform a bounded literal search over eligible UTF-8 source files."""
    root = resolve_git_workspace(root_path)
    normalized_query = query.strip()
    if not normalized_query or len(normalized_query) > MAX_SEARCH_QUERY:
        raise CodeWorkspaceError("Search query must contain 1-200 characters.")
    bounded_limit = min(max(1, limit), MAX_SEARCH_RESULTS)
    matches: list[CodeSearchMatch] = []
    for path in _iter_code_files(root):
        try:
            lines = _read_utf8(path).splitlines()
        except CodeWorkspaceError:
            continue
        for line_number, line in enumerate(lines, start=1):
            column = line.find(normalized_query)
            if column < 0:
                continue
            matches.append(
                CodeSearchMatch(
                    path=path.relative_to(root).as_posix(),
                    line=line_number,
                    column=column + 1,
                    preview=line[:500],
                )
            )
            if len(matches) >= bounded_limit:
                return matches
    return matches


def resolve_code_file(root: Path, relative_path: str) -> Path:
    """Resolve one relative source path while rejecting traversal and symlinks."""
    pure_path = PurePosixPath(relative_path)
    if (
        not relative_path
        or pure_path.is_absolute()
        or ".." in pure_path.parts
        or "." in pure_path.parts
    ):
        raise CodeWorkspaceError("Code file path must be a safe relative path.")
    if any(_is_sensitive_name(part) for part in pure_path.parts):
        raise CodeWorkspaceError("Sensitive files cannot be opened in Code Workspace.")
    candidate = root.joinpath(*pure_path.parts)
    current = root
    for part in pure_path.parts:
        current = current / part
        if current.is_symlink():
            raise CodeWorkspaceError("Symbolic links are not allowed in Code Workspace.")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CodeWorkspaceError("Code file is outside the selected workspace.") from exc
    if not resolved.is_file() or not _is_code_file(resolved):
        raise CodeWorkspaceError("Selected path is not a supported source file.")
    return resolved


def sha256_text(content: str) -> str:
    """Return the stable conflict-detection hash for UTF-8 text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Replace one source file without exposing a partially written state."""
    temporary = path.with_name(f".{path.name}.qwopus-tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _scan_directory(
    root: Path,
    directory: Path,
    depth: int,
    file_counter: list[int],
) -> CodeTreeNode:
    if depth > MAX_CODE_DEPTH:
        raise CodeWorkspaceError("Workspace directory nesting exceeds the safe limit.")
    children: list[CodeTreeNode] = []
    try:
        entries = sorted(
            directory.iterdir(),
            key=lambda item: (item.is_file(), item.name.casefold()),
        )
    except OSError as exc:
        raise CodeWorkspaceError(f"Could not read workspace directory: {directory.name}") from exc
    for entry in entries:
        if entry.is_symlink() or _is_sensitive_name(entry.name):
            continue
        if entry.is_dir():
            if entry.name in EXCLUDED_DIRECTORIES:
                continue
            child = _scan_directory(root, entry, depth + 1, file_counter)
            if child.children:
                children.append(child)
            continue
        if not _is_code_file(entry):
            continue
        file_counter[0] += 1
        if file_counter[0] > MAX_CODE_FILES:
            raise CodeWorkspaceError("Workspace contains more than 3000 supported source files.")
        children.append(
            CodeTreeNode(
                name=entry.name,
                relative_path=entry.relative_to(root).as_posix(),
                kind="file",
            )
        )
    return CodeTreeNode(
        name=directory.name,
        relative_path=directory.relative_to(root).as_posix() or ".",
        kind="directory",
        children=children,
    )


def _iter_code_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_names[:] = [
            name
            for name in sorted(directory_names)
            if name not in EXCLUDED_DIRECTORIES
            and not _is_sensitive_name(name)
            and not (Path(directory) / name).is_symlink()
        ]
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if not path.is_symlink() and _is_code_file(path):
                result.append(path)
                if len(result) >= MAX_CODE_FILES:
                    return result
    return result


def _read_utf8(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CodeWorkspaceError("Could not inspect source file.") from exc
    if size > MAX_CODE_FILE_BYTES:
        raise CodeWorkspaceError("Source file exceeds the 512 KiB safety limit.")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CodeWorkspaceError("Source file must be readable UTF-8 text.") from exc
    if "\x00" in content:
        raise CodeWorkspaceError("Binary files cannot be opened in Code Workspace.")
    return content


def _is_code_file(path: Path) -> bool:
    return (
        path.name in CODE_FILENAMES or path.suffix.casefold() in CODE_SUFFIXES
    ) and not _is_sensitive_name(path.name)


def _is_sensitive_name(name: str) -> bool:
    folded = name.casefold()
    return (
        folded in SENSITIVE_NAMES
        or folded.startswith(".env.")
        or Path(folded).suffix in SENSITIVE_SUFFIXES
    )
