"""Restricted pandas code execution for spreadsheet analysis."""

from __future__ import annotations

import ast
import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import pandas as pd

from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens


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
_ALLOWED_PANDAS_CALLS = {
    "DataFrame",
    "Series",
    "concat",
    "crosstab",
    "cut",
    "isna",
    "notna",
    "pivot_table",
    "qcut",
    "to_datetime",
    "to_numeric",
}
_ALLOWED_METHOD_CALLS = {
    "abs",
    "agg",
    "aggregate",
    "all",
    "any",
    "astype",
    "between",
    "clip",
    "contains",
    "copy",
    "corr",
    "count",
    "cov",
    "describe",
    "drop",
    "drop_duplicates",
    "dropna",
    "endswith",
    "extract",
    "fillna",
    "first",
    "get",
    "groupby",
    "head",
    "idxmax",
    "idxmin",
    "isin",
    "join",
    "last",
    "len",
    "lower",
    "map",
    "max",
    "mean",
    "median",
    "melt",
    "merge",
    "min",
    "nlargest",
    "nsmallest",
    "nunique",
    "pivot",
    "pivot_table",
    "quantile",
    "rename",
    "replace",
    "reset_index",
    "round",
    "set_index",
    "size",
    "sort_index",
    "sort_values",
    "split",
    "startswith",
    "std",
    "strip",
    "sum",
    "tail",
    "to_dict",
    "to_frame",
    "to_list",
    "to_numpy",
    "tolist",
    "transform",
    "transpose",
    "unique",
    "unstack",
    "upper",
    "value_counts",
    "var",
}
_MAX_CODE_CHARS = 4000
_MAX_AST_NODES = 200
_MAX_NUMERIC_LITERAL = 1_000_000
_MAX_RESULT_MARKDOWN_TOKENS = 3000
_SANDBOX_TIMEOUT_SECONDS = 8.0


def execute_pandas_code(code: str, dataframes: dict[str, pd.DataFrame]) -> SandboxExecutionResult:
    """Execute LLM-generated pandas code against in-memory dataframes."""
    stripped_code = _strip_code_fence(code)
    if len(stripped_code) > _MAX_CODE_CHARS:
        raise ValueError("Pandas sandbox code is too long.")
    tree = ast.parse(stripped_code, mode="exec")
    _validate_ast(tree)
    return _execute_in_subprocess(stripped_code, dataframes)


def _execute_validated_code(
    code: str,
    dataframes: dict[str, pd.DataFrame],
) -> SandboxExecutionResult:
    """Execute already-validated code inside the isolated worker process."""
    safe_dfs = {name: dataframe.copy(deep=True) for name, dataframe in dataframes.items()}
    globals_env = {"__builtins__": _ALLOWED_BUILTINS, "pd": pd}
    locals_env: dict[str, Any] = {"dfs": safe_dfs, "result": None}
    # 原因：LLM 只能基于本地 DataFrame 计算，不能访问文件、网络、系统或完整外部环境。
    # 作用：执行受限 AST 后只读取 result，避免任意副作用成为分析输出。
    exec(
        compile(code, "<qwopus_pandas_sandbox>", "exec"),
        globals_env,
        locals_env,
    )
    value = locals_env.get("result")
    if value is None:
        raise ValueError("Sandbox code must assign the final answer to result.")
    return SandboxExecutionResult(
        value=_bounded_result_value(value),
        markdown=_truncate_markdown(_result_to_markdown(value)),
    )


def _execute_in_subprocess(
    code: str,
    dataframes: dict[str, pd.DataFrame],
) -> SandboxExecutionResult:
    """Contain pandas execution crashes and enforce a wall-clock timeout."""
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_sandbox_worker,
        args=(writer, code, dataframes),
        daemon=True,
    )
    try:
        process.start()
    except Exception:
        reader.close()
        writer.close()
        process.close()
        raise
    writer.close()
    deadline = time.monotonic() + _SANDBOX_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if reader.poll(0.05):
                status, payload = reader.recv()
                process.join(timeout=1)
                if status == "ok" and isinstance(payload, SandboxExecutionResult):
                    return payload
                raise ValueError(str(payload))
            if not process.is_alive():
                process.join(timeout=0.1)
                raise RuntimeError(
                    f"Pandas sandbox process exited unexpectedly ({process.exitcode})."
                )
        process.terminate()
        process.join(timeout=1)
        raise TimeoutError("Pandas sandbox execution timed out.")
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        process.close()


def _sandbox_worker(
    connection: Connection,
    code: str,
    dataframes: dict[str, pd.DataFrame],
) -> None:
    """Return one serializable result without exposing worker internals."""
    try:
        connection.send(("ok", _execute_validated_code(code, dataframes)))
    except Exception as exc:  # noqa: BLE001 - normalize all worker failures at the boundary.
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


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
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise ValueError("Pandas sandbox code is too complex.")
    for node in nodes:
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
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
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
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and abs(node.value) > _MAX_NUMERIC_LITERAL
        ):
            raise ValueError("Pandas sandbox numeric literal is too large.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise ValueError("Exponentiation is not allowed in pandas sandbox.")


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
        "plot",
        "query",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_fwf",
        "read_hdf",
        "read_html",
        "read_json",
        "read_orc",
        "read_parquet",
        "read_pickle",
        "read_sql",
        "read_xml",
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_html",
        "to_json",
        "to_latex",
        "to_markdown",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_xml",
    }:
        raise ValueError(f"Blocked sandbox attribute: {node.attr}")


def _validate_call(node: ast.Call) -> None:
    """Reject direct calls to dangerous builtins or private attributes."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
        raise ValueError(f"Blocked sandbox call: {func.id}")
    if isinstance(func, ast.Attribute):
        _validate_attribute(func)
        root_name = _call_root_name(func)
        if root_name in _BLOCKED_CALLS:
            raise ValueError(f"Blocked sandbox call: {root_name}")
        allowed_calls = (
            _ALLOWED_PANDAS_CALLS
            if root_name == "pd"
            else _ALLOWED_METHOD_CALLS
        )
        if func.attr not in allowed_calls:
            raise ValueError(f"Unsupported sandbox method: {func.attr}")


def _call_root_name(node: ast.AST) -> str | None:
    """Resolve the first loaded name in a chained pandas expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_root_name(node.value)
    if isinstance(node, ast.Subscript):
        return _call_root_name(node.value)
    if isinstance(node, ast.Call):
        return _call_root_name(node.func)
    return None


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


def _bounded_result_value(value: Any) -> Any:
    """Keep the subprocess response small while preserving common scalar results."""
    if isinstance(value, pd.DataFrame):
        return value.head(20)
    if isinstance(value, pd.Series):
        return value.head(20)
    if isinstance(value, dict):
        return dict(list(value.items())[:50])
    if isinstance(value, (list, tuple)):
        return type(value)(value[:50])
    if isinstance(value, set):
        return set(list(value)[:50])
    return value


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
    if estimate_tokens(markdown) <= _MAX_RESULT_MARKDOWN_TOKENS:
        return markdown
    return (
        f"{truncate_to_tokens(markdown, _MAX_RESULT_MARKDOWN_TOKENS)}\n\n"
        "[Sandbox result truncated.]"
    )
