"""Optional MinerU document parser integration."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MINERU_OUTPUT_DIR = Path("storage/cache/mineru")
MINERU_COMMANDS = ("mineru", "magic-pdf")
VENDOR_MINERU_DIR = Path("vendor/MinerU")


class MinerUUnavailableError(RuntimeError):
    """Raised when MinerU is not installed or cannot produce Markdown."""


@dataclass(frozen=True)
class MinerUResult:
    """Markdown extracted by MinerU."""

    markdown: str

    output_path: Path

    command: str


def parse_document_with_mineru(
        document_path: Path,
        output_root: Path = MINERU_OUTPUT_DIR,
) -> MinerUResult:
    """Convert one document to Markdown through the MinerU command line."""
    command = _build_mineru_command()
    output_root.mkdir(parents=True, exist_ok=True)
    before = set(output_root.rglob("*.md"))

    # 原因：MinerU 已 vendored 到项目内，但用户也可能装了系统命令。
    # 作用：优先使用 vendor 源码入口，缺失时再用系统命令。
    process = subprocess.run(
        [
            *command.args,
            "-p",
            str(document_path),
            "-o",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=command.env,
    )
    if process.returncode != 0:
        raise MinerUUnavailableError(
            f"MinerU failed with exit code {process.returncode}: {process.stderr.strip()}"
        )

    markdown_path = _find_generated_markdown(output_root, before)
    if markdown_path is None:
        raise MinerUUnavailableError("MinerU did not produce a Markdown file.")

    markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
    if not markdown.strip():
        raise MinerUUnavailableError("MinerU produced an empty Markdown file.")

    return MinerUResult(
        markdown=markdown,
        output_path=markdown_path,
        command=command.label,
    )


@dataclass(frozen=True)
class MinerUCommand:
    """Command invocation details for MinerU."""

    args: list[str]

    label: str

    env: dict[str, str] | None = None


def _build_mineru_command() -> MinerUCommand:
    if VENDOR_MINERU_DIR.exists():
        env = os.environ.copy()
        pythonpath_parts = [
            str(VENDOR_MINERU_DIR.resolve()),
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
        return MinerUCommand(
            args=[sys.executable, "-m", "mineru.cli.client"],
            label=f"{sys.executable} -m mineru.cli.client",
            env=env,
        )

    for command in MINERU_COMMANDS:
        resolved = shutil.which(command)
        if resolved:
            return MinerUCommand(args=[resolved], label=resolved)
    raise MinerUUnavailableError("MinerU command not found. Install MinerU first.")


def _find_generated_markdown(output_root: Path, before: set[Path]) -> Path | None:
    candidates = [path for path in output_root.rglob("*.md") if path not in before]
    if not candidates:
        candidates = list(output_root.rglob("*.md"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
