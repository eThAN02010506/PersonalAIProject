"""Normalize smolagents execution records for safe orchestration and local debugging."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentDebugRun:
    """One complete local smolagents run retained only for the debug console."""

    label: str
    prompt: str
    max_steps: int
    state: str | None
    output: str
    steps: tuple[dict[str, Any], ...] = ()


def unpack_agent_run_result(
    run_result: Any,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Normalize smolagents RunResult and older direct return values."""
    if hasattr(run_result, "output"):
        output = getattr(run_result, "output", None)
        state = getattr(run_result, "state", None)
        steps = getattr(run_result, "steps", None)
        return str(output or ""), str(state) if state is not None else None, steps or []
    return str(run_result), None, []


def build_agent_debug_run(
    *,
    label: str,
    prompt: str,
    max_steps: int,
    state: str | None,
    output: str,
    steps: list[dict[str, Any]],
) -> AgentDebugRun:
    """Copy a JSON-safe raw run so the local debug console can inspect every step."""
    # 原因：smolagents step 中可能混入 Pydantic 对象或其他不可序列化值。
    # 作用：保留完整可读内容，同时确保 spawn Queue 和 JSON 下载都能稳定传输。
    normalized_steps = tuple(
        {
            str(key): _normalize_debug_value(value)
            for key, value in step.items()
        }
        if isinstance(step, dict)
        else {"value": _normalize_debug_value(step)}
        for step in steps
    )
    return AgentDebugRun(
        label=label,
        prompt=prompt,
        max_steps=max_steps,
        state=state,
        output=output,
        steps=normalized_steps,
    )


def extract_agent_tool_calls(steps: list[dict[str, Any]]) -> list[str]:
    """Extract Tool names from smolagents' succinct step records."""
    names: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for tool_call in step.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
    return names


def extract_successful_agent_tool_calls(steps: list[dict[str, Any]]) -> list[str]:
    """Extract Tool names only when execution produced a usable observation."""
    names: list[str] = []
    for step in steps:
        if (
            not isinstance(step, dict)
            or step.get("error")
            or not isinstance(step.get("observations"), str)
            or not step["observations"].strip()
        ):
            continue
        names.extend(extract_agent_tool_calls([step]))
    return names


def has_successful_tool_method(
    steps: list[dict[str, Any]],
    *,
    tool_name: str,
    method: str,
) -> bool:
    """Return whether one successful Tool call used the requested method."""
    for step in steps:
        if tool_name not in extract_successful_agent_tool_calls([step]):
            continue
        for tool_call in step.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict) or function.get("name") != tool_name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if isinstance(arguments, dict) and arguments.get("method") == method:
                return True
    return False


def extract_agent_observations(steps: list[dict[str, Any]]) -> list[str]:
    """Return observations produced by non-final tools without model thoughts."""
    observations: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_names = extract_agent_tool_calls([step])
        observation = step.get("observations")
        if (
            isinstance(observation, str)
            and observation.strip()
            and any(name != "final_answer" for name in tool_names)
        ):
            observations.append(observation.strip())
    return observations


def extract_tool_observations(
    steps: list[dict[str, Any]],
    tool_name: str,
) -> list[str]:
    """Return observations produced by one named Tool."""
    observations: list[str] = []
    for step in steps:
        if (
            isinstance(step, dict)
            and not step.get("error")
            and tool_name in extract_agent_tool_calls([step])
            and isinstance(step.get("observations"), str)
            and step["observations"].strip()
        ):
            observations.append(step["observations"].strip())
    return observations


def extract_inspected_file_names(steps: list[dict[str, Any]]) -> set[str]:
    """Return files passed to a content-bearing current-document Tool."""
    inspected: set[str] = set()
    content_tools = {
        "document_search",
        "document_read_section",
        "document_summary",
        "excel_schema",
        "excel_analysis",
        "excel_modeling",
        "excel_statistics",
    }
    for step in steps:
        if not isinstance(step, dict):
            continue
        for tool_call in step.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict) or function.get("name") not in content_tools:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if isinstance(arguments, dict) and isinstance(arguments.get("file_name"), str):
                inspected.add(arguments["file_name"])
    return inspected


def extract_collection_covered_file_names(steps: list[dict[str, Any]]) -> set[str]:
    """Read the trusted source manifest emitted by document_collection_summary."""
    covered: set[str] = set()
    marker = re.compile(r"QWOPUS_SOURCE_COVERAGE=(\[[^\r\n]*\])")
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "document_collection_summary" not in extract_agent_tool_calls([step]):
            continue
        observation = step.get("observations")
        if not isinstance(observation, str):
            continue
        match = marker.search(observation)
        if match is None:
            continue
        try:
            sources = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(sources, list):
            covered.update(
                source
                for source in sources
                if isinstance(source, str) and source.strip()
            )
    return covered


def required_file_tools(spreadsheet_names: list[str]) -> set[str]:
    """Return the minimum Tool chain required before a file answer is accepted."""
    required: set[str] = set()
    if spreadsheet_names:
        # 原因：模型可能直接根据 sample 心算，并猜测不存在的字段或单位。
        # 作用：所有 Excel 回答必须先看 schema；计算工具的“二选一”由下方独立检查。
        required.add("excel_schema")
    return required


def missing_required_file_tools(
    *,
    spreadsheet_names: list[str],
    required_tools: set[str],
    successful_tool_calls: list[str],
) -> set[str]:
    """Resolve fixed requirements plus the spreadsheet computation alternative."""
    missing = required_tools.difference(successful_tool_calls)
    if (
        spreadsheet_names
        and {"excel_statistics", "excel_modeling", "excel_analysis"}.isdisjoint(
            successful_tool_calls
        )
    ):
        # 原因：Tool 名出现在失败步骤中不代表本地计算产生了可信结果。
        # 作用：要求至少一次带 Observation 的真实计算，阻止模型用猜测表格冒充 Skill 输出。
        missing.add("excel_statistics")
    return missing


def agent_debug_steps(
    state: str | None,
    steps: list[dict[str, Any]],
    tool_calls: list[str],
    prefix: str = "smolagents",
) -> list[str]:
    """Build a safe trace without exposing Tool observations or model reasoning."""
    trace = [f"{prefix} 运行状态：{state or 'completed'}；步骤数：{len(steps)}。"]
    for tool_name in tool_calls:
        # 原因：用户可以选择查看 Agent 过程，但原始 Observation 可能包含整段文件内容。
        # 作用：调试区只展示调用了哪个 Tool，不展示参数、推理文本或 Tool 返回正文。
        trace.append(f"{prefix} 调用 Tool：{tool_name}")
    for step in steps:
        if isinstance(step, dict) and step.get("error"):
            step_number = step.get("step_number", "?")
            trace.append(f"{prefix} 第 {step_number} 步发生错误，Agent 已按运行策略处理。")
    return trace


def looks_like_tool_observation(text: str) -> bool:
    """Detect model output that is still exposing tool observations."""
    lowered = text.lower()
    return "observation:" in lowered or "document analysis:" in lowered or "## preview" in lowered


def extract_final_answer(text: str) -> str:
    """Extract final_answer(...) when a CodeAgent-style answer leaks through."""
    match = re.search(r"final_answer\((?P<quote>['\"])(?P<answer>.*?)(?P=quote)\)", text, re.S)
    if match:
        return match.group("answer").strip()
    return text.strip()


def _normalize_debug_value(value: Any) -> Any:
    """Convert nested debug values to JSON-safe primitives without dropping content."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_debug_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize_debug_value(model_dump(mode="json"))
    return str(value)
