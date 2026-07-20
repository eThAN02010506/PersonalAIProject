"""Restricted pandas code execution for spreadsheet analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SandboxExecutionResult:
    """Normalized result returned by the pandas sandbox."""

    value: Any

    markdown: str


_ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_ALLOWED_ROOT_NAMES = set(_ALLOWED_BUILTINS) | {"dfs", "pd", "result"}
_ALLOWED_ASSIGN_NAMES = {
    "data",
    "df",
    "grouped",
    "mask",
    "result",
    "rows",
    "series",
    "summary",
    "table",
    "value",
    "values",
}
_BLOCKED_CALLS = {
    "__import__",
    "compile",
    "dir",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
}
_BLOCKED_NAMES = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "urllib"}
_MAX_CODE_CHARS = 4000
_MAX_RESULT_MARKDOWN_CHARS = 6000


def execute_pandas_code(code: str, dataframes: dict[str, pd.DataFrame]) -> SandboxExecutionResult:
    """Execute LLM-generated pandas code against in-memory dataframes."""
    stripped_code = _strip_code_fence(code)
    if len(stripped_code) > _MAX_CODE_CHARS:
        raise ValueError("Pandas sandbox code is too long.")
    tree = ast.parse(stripped_code, mode="exec")
    _validate_ast(tree)

    safe_dfs = {name: dataframe.copy(deep=True) for name, dataframe in dataframes.items()}
    globals_env = {"__builtins__": _ALLOWED_BUILTINS, "pd": pd}
    locals_env: dict[str, Any] = {"dfs": safe_dfs, "result": None}
    # 原因：LLM 只能基于本地 DataFrame 计算，不能访问文件、网络、系统或完整外部环境。
    # 作用：执行受限 AST 后只读取 result，避免任意副作用成为分析输出。
    exec(compile(tree, "<qwopus_pandas_sandbox>", "exec"), globals_env, locals_env)
    value = locals_env.get("result")
    if value is None:
        raise ValueError("Sandbox code must assign the final answer to result.")
    return SandboxExecutionResult(
        value=value,
        markdown=_truncate_markdown(_result_to_markdown(value)),
    )


def _strip_code_fence(code: str) -> str:
    """Remove Markdown code fences around generated Python."""
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _validate_ast(tree: ast.AST) -> None:
    """Reject nodes and names that could escape the dataframe sandbox."""
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.Delete,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise ValueError(f"Unsupported sandbox syntax: {type(node).__name__}")
        if isinstance(node, ast.Assign):
            _validate_assignment(node)
        if isinstance(node, ast.AugAssign):
            raise ValueError("Augmented assignment is not allowed in pandas sandbox.")
        if isinstance(node, ast.Name):
            _validate_name(node)
        if isinstance(node, ast.Attribute):
            _validate_attribute(node)
        if isinstance(node, ast.Call):
            _validate_call(node)


def _validate_assignment(node: ast.Assign) -> None:
    """Allow assignments only to simple local variables."""
    for target in node.targets:
        if not isinstance(target, ast.Name) or target.id not in _ALLOWED_ASSIGN_NAMES:
            raise ValueError("Sandbox assignments must use approved local variable names.")


def _validate_name(node: ast.Name) -> None:
    """Block dangerous names and unknown root reads."""
    if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
        raise ValueError(f"Blocked sandbox name: {node.id}")
    if (
        isinstance(node.ctx, ast.Load)
        and node.id not in _ALLOWED_ROOT_NAMES
        and node.id not in _ALLOWED_ASSIGN_NAMES
    ):
        raise ValueError(f"Unknown sandbox name: {node.id}")


def _validate_attribute(node: ast.Attribute) -> None:
    """Block private and escape-prone attribute access."""
    if node.attr.startswith("_"):
        raise ValueError(f"Blocked private attribute access: {node.attr}")
    if node.attr in {
        "eval",
        "query",
        "read_csv",
        "read_excel",
        "read_sql",
        "to_csv",
        "to_excel",
        "to_json",
        "to_pickle",
        "to_sql",
    }:
        raise ValueError(f"Blocked sandbox attribute: {node.attr}")


def _validate_call(node: ast.Call) -> None:
    """Reject direct calls to dangerous builtins or private attributes."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
        raise ValueError(f"Blocked sandbox call: {func.id}")
    if isinstance(func, ast.Attribute):
        _validate_attribute(func)


def _result_to_markdown(value: Any) -> str:
    """Convert sandbox result into bounded Markdown for LLM/UI context."""
    if isinstance(value, pd.DataFrame):
        return _dataframe_to_markdown(value.head(20))
    if isinstance(value, pd.Series):
        return _dataframe_to_markdown(value.head(20).to_frame(name=value.name or "value"))
    if isinstance(value, dict):
        rows = [{"key": key, "value": item} for key, item in value.items()]
        return _dataframe_to_markdown(pd.DataFrame(rows).head(50))
    if isinstance(value, (list, tuple, set)):
        return _dataframe_to_markdown(pd.DataFrame({"value": list(value)}).head(50))
    return str(value)


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without optional tabulate dependency."""
    if dataframe.empty:
        return "_Empty result._"
    columns = [str(column) for column in dataframe.columns]
    rows = dataframe.astype(str).values.tolist()
    # 原因：pandas.to_markdown 需要 tabulate，可选依赖不一定安装。
    # 作用：用项目内轻量渲染保证沙箱结果在干净环境中也能展示。
    header = "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _escape_markdown_cell(value: str) -> str:
    """Escape table cell separators for Markdown output."""
    return value.replace("|", "\|").replace("\n", " ")


def _truncate_markdown(markdown: str) -> str:
    """Bound sandbox output before adding it to LLM context."""
    if len(markdown) <= _MAX_RESULT_MARKDOWN_CHARS:
        return markdown
    return markdown[:_MAX_RESULT_MARKDOWN_CHARS] + "\n\n[Sandbox result truncated.]"
