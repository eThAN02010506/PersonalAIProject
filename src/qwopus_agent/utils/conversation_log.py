"""Local JSONL conversation logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_LOG_PATH = Path("logs/conversations.jsonl")


def append_conversation_event(
        event_type: str,
        payload: dict[str, Any],
        log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one conversation event to a local JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    # 原因：用户需要对话留存，但当前阶段不需要数据库。
    # 作用：用 append-only JSONL 保存聊天和分析事件，便于后续升级为持久记忆。
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chat_messages(
        log_path: Path = DEFAULT_LOG_PATH,
        limit: int = 50,
) -> list[dict[str, str]]:
    """Load recent chat messages from the JSONL log."""
    if not log_path.exists():
        return []

    messages: list[dict[str, str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event_type") != "chat_message":
            continue
        payload = record.get("payload", {})
        role = payload.get("role")
        content = payload.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    return messages[-limit:]
