"""Tavily provider used by the reusable web-search Skill."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dotenv import dotenv_values


@dataclass(frozen=True)
class TavilySearchConfig:
    """Runtime configuration for Tavily search."""

    api_key: str | None = None
    endpoint: str = "https://api.tavily.com/search"
    max_results: int = 5
    timeout_seconds: int = 20


@dataclass
class TavilySearchProvider:
    """Implement the provider-neutral search contract through Tavily."""

    config: TavilySearchConfig = field(default_factory=TavilySearchConfig)
    progress_callback: Callable[[str], None] | None = None
    _completed_queries: set[str] = field(default_factory=set, init=False, repr=False)

    def search(self, query: str) -> list[str]:
        """Return bounded Markdown evidence for one web query."""
        normalized_query = " ".join(query.split()).casefold()
        if normalized_query in self._completed_queries:
            # 原因：部分模型会在下一步原样重复已成功的搜索，浪费配额和上下文。
            # 作用：Provider 在所有调用入口统一去重，而不是只保护 smolagents Tool。
            return [
                "This exact Tavily query was already completed. Use the previous "
                "search evidence and finish the answer."
            ]

        api_key = resolve_tavily_api_key(self.config.api_key)
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        if self.progress_callback is not None:
            self.progress_callback("searching")

        payload = json.dumps(
            {
                "query": query,
                "search_depth": "basic",
                "max_results": self.config.max_results,
                "topic": "general",
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Qwopus-Agent/0.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.config.timeout_seconds,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        self._completed_queries.add(normalized_query)
        if self.progress_callback is not None:
            self.progress_callback("generating")
        return [format_tavily_results(data, max_results=self.config.max_results)]


def format_tavily_results(payload: dict[str, Any], max_results: int) -> str:
    """Format Tavily response JSON as concise Markdown evidence."""
    sections: list[str] = []
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        sections.append(f"## Tavily Answer\n\n{answer.strip()}")

    result_lines: list[str] = []
    for item in payload.get("results", [])[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title and not content:
            continue
        heading = f"### {title}" if title else "### Search Result"
        if url:
            heading = f"{heading}\n{url}"
        result_lines.append(f"{heading}\n\n{content}".strip())

    if result_lines:
        sections.append("## Tavily Search Results\n\n" + "\n\n".join(result_lines))
    return "\n\n".join(sections) if sections else "No Tavily search results."


def resolve_tavily_api_key(explicit_api_key: str | None) -> str:
    """Resolve Tavily credentials without changing the tracked .env file."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    # 原因：项目的 .env 已被 Git 跟踪，联网密钥不能继续写入该文件。
    # 作用：优先读取被 Git 忽略的本地配置，同时保留进程环境变量部署方式。
    local_api_key = dotenv_values(".env.local").get("TAVILY_API_KEY")
    if isinstance(local_api_key, str) and local_api_key.strip():
        return local_api_key.strip()
    return (os.getenv("TAVILY_API_KEY") or "").strip()
