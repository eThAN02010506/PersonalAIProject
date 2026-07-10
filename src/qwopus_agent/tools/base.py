"""Base interfaces for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    """Execution context passed to tools."""

    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by every tool."""

    success: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for reusable agent tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        """Run the tool with structured arguments."""
