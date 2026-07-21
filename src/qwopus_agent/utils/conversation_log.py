"""Local JSONL conversation logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_LOG_PATH = Path("logs/conversations.jsonl")
LEGACY_CONVERSATION_ID = "legacy-history"


@dataclass(frozen=True)
class ConversationSummary:
    """Sidebar metadata for one persisted conversation."""

    conversation_id: str
    title: str
    updated_at: str
    message_count: int


def append_conversation_event(
        event_type: str,
        payload: dict[str, Any],
        log_path: Path = DEFAULT_LOG_PATH,
        conversation_id: str | None = None,
) -> None:
    """Append one conversation event to a local JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": uuid4().hex,
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    if conversation_id is not None:
        record["conversation_id"] = conversation_id
    # 原因：用户需要对话留存，但当前阶段不需要数据库。
    # 作用：用 append-only JSONL 保存聊天和分析事件，便于后续升级为持久记忆。
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_conversation(
        title: str = "新对话",
        log_path: Path = DEFAULT_LOG_PATH,
) -> str:
    """Create and persist one empty conversation."""
    conversation_id = uuid4().hex
    append_conversation_event(
        "conversation_created",
        {"title": title.strip() or "新对话"},
        log_path=log_path,
        conversation_id=conversation_id,
    )
    return conversation_id


def rename_conversation(
        conversation_id: str,
        title: str,
        log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Persist a conversation title change."""
    append_conversation_event(
        "conversation_renamed",
        {"title": title.strip() or "新对话"},
        log_path=log_path,
        conversation_id=conversation_id,
    )


def delete_conversation(
        conversation_id: str,
        log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a tombstone so one conversation stays deleted after restart."""
    append_conversation_event(
        "conversation_deleted",
        {},
        log_path=log_path,
        conversation_id=conversation_id,
    )


def conversation_title(content: str, max_length: int = 32) -> str:
    """Build a compact deterministic title from the first user message."""
    title = " ".join(content.split()).strip()
    if not title:
        return "新对话"
    return title if len(title) <= max_length else f"{title[:max_length - 1]}…"


def list_conversations(
        log_path: Path = DEFAULT_LOG_PATH,
) -> list[ConversationSummary]:
    """Rebuild active conversation summaries from the append-only event log."""
    conversations: dict[str, dict[str, Any]] = {}
    deleted: set[str] = set()
    for record in _load_records(log_path):
        event_type = record.get("event_type")
        payload = record.get("payload", {})
        conversation_id = _record_conversation_id(record)
        if conversation_id is None:
            continue
        if event_type == "conversation_deleted":
            deleted.add(conversation_id)
            conversations.pop(conversation_id, None)
            continue
        if conversation_id in deleted:
            continue
        summary = conversations.setdefault(
            conversation_id,
            {
                "title": "历史对话" if conversation_id == LEGACY_CONVERSATION_ID else "新对话",
                "updated_at": str(record.get("timestamp", "")),
                "message_count": 0,
            },
        )
        summary["updated_at"] = str(record.get("timestamp", summary["updated_at"]))
        if event_type in {"conversation_created", "conversation_renamed"}:
            title = payload.get("title")
            if isinstance(title, str) and title.strip():
                summary["title"] = title.strip()
        elif event_type == "chat_message":
            summary["message_count"] += 1
            if summary["title"] == "新对话" and payload.get("role") == "user":
                summary["title"] = conversation_title(str(payload.get("content", "")))

    return sorted(
        (
            ConversationSummary(
                conversation_id=conversation_id,
                title=str(summary["title"]),
                updated_at=str(summary["updated_at"]),
                message_count=int(summary["message_count"]),
            )
            for conversation_id, summary in conversations.items()
        ),
        key=lambda item: item.updated_at,
        reverse=True,
    )


def load_chat_messages(
        log_path: Path = DEFAULT_LOG_PATH,
        limit: int = 50,
        conversation_id: str | None = None,
) -> list[dict[str, str]]:
    """Load recent chat messages from the JSONL log."""
    messages: list[dict[str, str]] = []
    for record in _load_records(log_path):
        if record.get("event_type") != "chat_message":
            continue
        if conversation_id is not None and _record_conversation_id(record) != conversation_id:
            continue
        payload = record.get("payload", {})
        role = payload.get("role")
        content = payload.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    return messages[-limit:]


def _record_conversation_id(record: dict[str, Any]) -> str | None:
    """Resolve new conversation IDs while assigning old chat events to one legacy thread."""
    conversation_id = record.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id:
        return conversation_id
    return LEGACY_CONVERSATION_ID if record.get("event_type") == "chat_message" else None


def _load_records(log_path: Path) -> list[dict[str, Any]]:
    """Read valid JSON objects and ignore interrupted or malformed log lines."""
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records
