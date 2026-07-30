"""Restricted pandas code execution for spreadsheet analysis."""

from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qwopus_agent.analysis.markdown_tables import dataframe_to_markdown
from qwopus_agent.utils.token_budget import estimate_tokens, truncate_to_tokens


@dataclass(frozen=True)
class SandboxExecutionResult:
    """Normalized result returned by the pandas sandbox."""

    value: Any

    markdown: str


# 原因：弱模型常按普通 pandas 脚本习惯重复读文件、导入模块或修改原 DataFrame。
# 作用：向 Prompt、Tool 描述和失败反馈提供同一份可执行合同，避免安全规则在多处漂移。
PANDAS_SANDBOX_CODE_GUIDANCE = (
    "Sandbox code contract: the workbook is already loaded in dfs; start with "
    'df = dfs["exact sheet or table name"]. Do not import, read files, call '
    "pd.read_excel, use loops/lambdas/comprehensions, mutate df[...] columns, or "
    "call output methods such as to_markdown. Assign only simple approved variables: "
    "data, df, grouped, mask, result, rows, series, summary, table, value, values. "
    "Always assign the final scalar, Series, or DataFrame to result. Safe pattern: "
    'df = dfs["Table 1"]\\n'
    'values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")\\n'
    'result = pd.DataFrame({"label": df.iloc[:, 0], '
    '"mean": values.mean(axis=1).round(3)})'
)


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
    # 原因：真实报表列常混有标题和占位符，需要逐列调用白名单转换函数后才能计算。
    # 作用：允许 apply(pd.to_numeric)，lambda、函数定义和危险调用仍由 AST 规则拒绝。
    "apply",
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
_MAX_SERIALIZED_INPUT_BYTES = 256 * 1024 * 1024
_MACOS_SANDBOX_EXECUTABLE = Path("/usr/bin/sandbox-exec")


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
    """Execute validated code in a bounded worker and normalize its response."""
    serialized_request = pickle.dumps((code, dataframes), protocol=pickle.HIGHEST_PROTOCOL)
    if len(serialized_request) > _MAX_SERIALIZED_INPUT_BYTES:
        raise ValueError("Pandas sandbox input is too large.")

    command = _sandbox_command()
    try:
        # 原因：墙钟超时必须由父进程掌握，CPU 限制无法终止阻塞的本地库调用。
        # 作用：subprocess.run 在超时后杀死整个 worker，不遗留后台分析进程。
        completed = subprocess.run(
            command,
            input=serialized_request,
            capture_output=True,
            check=False,
            cwd=_sandbox_working_directory(),
            env=_sandbox_environment(),
            timeout=_SANDBOX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Pandas sandbox execution timed out.") from exc

    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        detail = error[-1000:] if error else f"exit code {completed.returncode}"
        raise RuntimeError(f"Pandas sandbox process exited unexpectedly ({detail}).")
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Pandas sandbox returned an invalid response.") from exc
    if not isinstance(response, dict):
        raise RuntimeError("Pandas sandbox returned an invalid response.")
    if response.get("status") == "ok" and isinstance(response.get("markdown"), str):
        # 原因：受限 worker 的输出属于不可信边界，父进程不能对它执行 pickle 反序列化。
        # 作用：只接收 JSON 数据并重建公开结果对象，避免 worker 构造可执行对象图。
        return SandboxExecutionResult(
            value=response.get("value"),
            markdown=response["markdown"],
        )
    raise ValueError(str(response.get("error", "Pandas sandbox execution failed.")))


def _sandbox_command() -> list[str]:
    """Build the platform-specific worker command."""
    worker = [sys.executable, "-m", "qwopus_agent.analysis._pandas_worker"]
    if sys.platform != "darwin":
        return worker
    if not _MACOS_SANDBOX_EXECUTABLE.is_file():
        raise RuntimeError("macOS pandas sandbox requires /usr/bin/sandbox-exec.")
    # 原因：AST 校验限制 Python 表达能力，但本身不是操作系统安全边界。
    # 作用：Apple Silicon 主路径用 Seatbelt 拒绝网络、写文件和 fork；即使验证器
    # 将来出现缺口，生成代码仍不能产生这些系统副作用。
    return [
        str(_MACOS_SANDBOX_EXECUTABLE),
        "-p",
        _macos_sandbox_profile(),
        *worker,
    ]


def _macos_sandbox_profile() -> str:
    """Return the restrictive Seatbelt policy applied to the pandas worker."""
    project_root = Path(__file__).resolve().parents[3]
    home = Path.home()
    protected_paths = (
        home / ".aws",
        home / ".codex",
        home / ".config",
        home / ".ssh",
        home / "Documents",
        home / "Downloads",
        home / "Library",
        project_root / ".env",
        project_root / ".git",
        project_root / "storage",
    )
    read_denials = "".join(
        f'(deny file-read* (subpath "{_seatbelt_path(path)}"))'
        for path in protected_paths
    )
    return (
        "(version 1)"
        "(allow default)"
        "(deny network*)"
        "(deny process-fork)"
        "(deny file-write*)"
        f"{read_denials}"
    )


def _seatbelt_path(path: Path) -> str:
    """Escape one absolute path for a generated Seatbelt string literal."""
    return str(path.expanduser().resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _sandbox_environment() -> dict[str, str]:
    """Pass only deterministic runtime settings needed by pandas and the worker."""
    source_path = str(Path(__file__).resolve().parents[2])
    inherited_python_path = [
        str((Path.cwd() / item).resolve()) if not Path(item).is_absolute() else item
        for item in os.getenv("PYTHONPATH", "").split(os.pathsep)
        if item
    ]
    python_path = os.pathsep.join(dict.fromkeys((source_path, *inherited_python_path)))
    # 原因：继承模型密钥、代理和用户 HOME 会扩大生成代码意外接触的环境范围。
    # 作用：worker 只收到模块路径、语言和单线程数值库配置，不包含应用凭据。
    return {
        "HOME": "/private/var/empty" if sys.platform == "darwin" else "/tmp",
        "LANG": os.getenv("LANG", "C.UTF-8"),
        "LC_ALL": os.getenv("LC_ALL", ""),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": os.getenv("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": python_path,
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _sandbox_working_directory() -> Path:
    """Keep generated code outside the project and uploaded-file directories."""
    empty_directory = Path("/private/var/empty")
    return empty_directory if empty_directory.is_dir() else Path("/")


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
        return dataframe_to_markdown(value.head(20))
    if isinstance(value, pd.Series):
        return dataframe_to_markdown(value.head(20).to_frame(name=value.name or "value"))
    if isinstance(value, dict):
        rows = [{"key": key, "value": item} for key, item in value.items()]
        return dataframe_to_markdown(pd.DataFrame(rows).head(50))
    if isinstance(value, (list, tuple, set)):
        return dataframe_to_markdown(pd.DataFrame({"value": list(value)}).head(50))
    # 原因：单个平均值若返回裸数字，模型容易把计算结果改写成无法核对的叙述。
    # 作用：即使结果是标量也保留统一表格契约，原始 typed value 仍供内部测试使用。
    return dataframe_to_markdown(pd.DataFrame([{"result": value}]))


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


def _truncate_markdown(markdown: str) -> str:
    """Bound sandbox output before adding it to LLM context."""
    if estimate_tokens(markdown) <= _MAX_RESULT_MARKDOWN_TOKENS:
        return markdown
    return (
        f"{truncate_to_tokens(markdown, _MAX_RESULT_MARKDOWN_TOKENS)}\n\n"
        "[Sandbox result truncated.]"
    )
