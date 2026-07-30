"""Allowlisted verification commands for approved source changes."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from qwopus_agent.code_workspace.models import CodeCommandView, CodeTestResult
from qwopus_agent.code_workspace.security import CodeWorkspaceError

MAX_TEST_OUTPUT = 40_000
TEST_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class CodeCommand:
    id: str
    label: str
    description: str
    arguments: tuple[str, ...]


COMMANDS = {
    "git_diff_check": CodeCommand(
        id="git_diff_check",
        label="Git diff check",
        description="Check whitespace errors in the current working tree.",
        arguments=("git", "diff", "--check"),
    ),
    "python_tests": CodeCommand(
        id="python_tests",
        label="Python tests",
        description="Run the repository unittest suite with its local virtual environment.",
        arguments=(".venv/bin/python", "-m", "unittest", "discover", "-s", "tests"),
    ),
    "ruff": CodeCommand(
        id="ruff",
        label="Ruff",
        description="Run the repository's configured Python lint checks.",
        arguments=(".venv/bin/ruff", "check", "."),
    ),
    "mypy": CodeCommand(
        id="mypy",
        label="Mypy",
        description="Run strict type checking for the Qwopus-Agent package.",
        arguments=(".venv/bin/mypy", "src/qwopus_agent"),
    ),
    "frontend_build": CodeCommand(
        id="frontend_build",
        label="Frontend build",
        description="Type-check and build the React frontend.",
        arguments=("pnpm", "--dir", "frontend", "build"),
    ),
}


def list_code_commands(root: Path) -> list[CodeCommandView]:
    """Return commands whose required local executable and target exist."""
    result: list[CodeCommandView] = []
    for command in COMMANDS.values():
        executable = command.arguments[0]
        if executable.startswith(".") and not (root / executable).is_file():
            continue
        if command.id == "python_tests" and not (root / "tests").is_dir():
            continue
        if command.id == "frontend_build" and not (root / "frontend").is_dir():
            continue
        result.append(
            CodeCommandView(
                id=command.id,
                label=command.label,
                description=command.description,
            )
        )
    return result


def run_code_command(root: Path, command_id: str) -> CodeTestResult:
    """Run one fixed argv list without a shell, interpolation, or user arguments."""
    command = COMMANDS.get(command_id)
    if command is None or command_id not in {item.id for item in list_code_commands(root)}:
        raise CodeWorkspaceError("Selected verification command is unavailable.")
    environment = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
    }
    try:
        result = subprocess.run(
            list(command.arguments),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        output = _bounded_output(
            _timeout_text(exc.stdout) + _timeout_text(exc.stderr)
        )
        return CodeTestResult(
            command_id=command_id,
            command=list(command.arguments),
            return_code=124,
            success=False,
            timed_out=True,
            output=output or "Verification timed out after 180 seconds.",
        )
    output = _bounded_output(result.stdout + result.stderr)
    return CodeTestResult(
        command_id=command_id,
        command=list(command.arguments),
        return_code=result.returncode,
        success=result.returncode == 0,
        output=output,
    )


def _bounded_output(output: str) -> str:
    if len(output) <= MAX_TEST_OUTPUT:
        return output
    return output[:MAX_TEST_OUTPUT] + "\n... output truncated ..."


def _timeout_text(value: bytes | str | None) -> str:
    """Normalize subprocess timeout output across Python platform implementations."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
