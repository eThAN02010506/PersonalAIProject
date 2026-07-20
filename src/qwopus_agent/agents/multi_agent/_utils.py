"""Private normalization helpers used by multi-agent components."""

from __future__ import annotations

import json
import re
from typing import Any


def result_content(result: Any) -> str:
    """Normalize common Agent result envelopes into text."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("final_answer", "content", "answer"):
            if key in result:
                return str(result[key])
    for attribute in ("final_answer", "content"):
        value = getattr(result, attribute, None)
        if value is not None:
            return str(value)
    execution = getattr(result, "execution", None)
    if execution is not None and getattr(execution, "content", None) is not None:
        return str(execution.content)
    return json.dumps(json_safe(result), ensure_ascii=False)


def result_success(result: Any) -> bool:
    """Read success from common Agent result envelopes."""
    if result is None:
        return False
    if isinstance(result, dict) and "success" in result:
        return bool(result["success"])
    success = getattr(result, "success", None)
    if success is not None:
        return bool(success)
    execution = getattr(result, "execution", None)
    if execution is not None and getattr(execution, "success", None) is not None:
        return bool(execution.success)
    return True


def result_confidence(result: Any) -> float:
    """Normalize confidence into the closed interval from zero to one."""
    value: Any = None
    if isinstance(result, dict):
        value = result.get("confidence")
    if value is None:
        value = getattr(result, "confidence", None)
    try:
        return max(0.0, min(1.0, float(value))) if value is not None else 0.5
    except (TypeError, ValueError):
        return 0.5


def normalize_content(content: str) -> str:
    """Normalize text for exact consensus comparison."""
    return " ".join(content.casefold().split())


def safe_identifier(value: str) -> str:
    """Convert an Agent name into a stable task-id component."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")
    return normalized or "agent"


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first JSON object from plain or fenced model output."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    start = stripped.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", stripped, 0)
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise TypeError("Model output must be a JSON object.")
    return value


def json_safe(value: Any) -> Any:
    """Convert arbitrary context into a JSON-safe prompt payload."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)
