"""Optional MinerU document parser integration."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

MINERU_OUTPUT_DIR = Path("storage/cache/mineru")
MINERU_COMMANDS = ("mineru", "magic-pdf")
VENDOR_MINERU_DIR = Path("vendor/MinerU")
MINERU_BACKEND = "pipeline"
OCR_IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg"}


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
    # 原因：图片没有 PDF 文本层，auto 可能只保留图片引用而不识别其中的文字。
    # 作用：PNG/JPEG 强制使用 OCR；其他文档继续让 MinerU 自动选择解析方法。
    parse_method = "ocr" if document_path.suffix.lower() in OCR_IMAGE_EXTENSIONS else "auto"
    run_output_dir = output_root / uuid4().hex
    # 原因：并行解析若共享一个输出目录，“最新 Markdown”可能属于另一份文件。
    # 作用：每次 MinerU 调用只扫描自己的目录，保证上传文件与解析结果一一对应。
    run_output_dir.mkdir(parents=True, exist_ok=False)

    # 原因：MinerU 已 vendored 到项目内，但用户也可能装了系统命令。
    # 作用：优先使用 vendor 源码入口，缺失时再用系统命令。
    try:
        process = subprocess.run(
            [
                *command.args,
                "-p",
                str(document_path),
                "-o",
                str(run_output_dir),
                # 原因：MinerU 默认 hybrid-engine 对本机内存和模型依赖要求较高。
                # 作用：固定使用支持 CPU/MPS 的 pipeline，稳定处理 PDF、图片和 OCR。
                "-b",
                MINERU_BACKEND,
                "-m",
                parse_method,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=command.env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # 原因：PDF/DOCX 有确定性回退解析器，但只有统一异常类型才会触发它。
        # 作用：命令无法启动或超时时安全降级；图片仍向用户报告 MinerU 不可用。
        raise MinerUUnavailableError(f"MinerU could not complete: {exc}") from exc
    if process.returncode != 0:
        raise MinerUUnavailableError(
            f"MinerU failed with exit code {process.returncode}: {process.stderr.strip()}"
        )

    markdown_path = _find_generated_markdown(run_output_dir, set())
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
