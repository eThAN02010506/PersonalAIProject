"""Knowledge and web Tool assembly for smolagents chat runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from qwopus_agent.memory import (
    DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    conversation_knowledge_path,
)
from qwopus_agent.utils.token_budget import TokenBudgetManager


class LocalKnowledgeTools(list[Any]):
    """List-compatible tools with deterministic source-routing metadata."""

    def __init__(
        self,
        values: list[Any],
        *,
        private_sources: tuple[str, ...] | None,
        global_sources: tuple[str, ...] | None,
        primary_scope: Literal["private", "global", "none"],
    ) -> None:
        super().__init__(values)
        self.private_sources = private_sources
        self.global_sources = global_sources
        self.primary_scope = primary_scope


def build_tavily_search_tool(
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    """Load the Tavily Tool only when web search is enabled."""
    # 原因：普通聊天不需要 Excel、文档或 MiniRAG Tool 依赖。
    # 作用：关闭联网时避免加载完整工具模块，同时保留可测试的工厂注入点。
    from qwopus_agent.integrations.smolagents_tools import (
        build_tavily_search_tool as create_tool,
    )

    return create_tool(progress_callback=progress_callback)


def build_browser_open_tool(
    progress_callback: Callable[[str], None] | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    """Load the Playwright Tool only when browser access is enabled."""
    from qwopus_agent.integrations.smolagents_tools import (
        build_browser_open_tool as create_tool,
    )

    return create_tool(
        progress_callback=progress_callback,
        max_output_tokens=max_output_tokens,
    )


def build_local_knowledge_tools(
    knowledge_scope: str,
    user_message: str = "",
    progress_callback: Callable[[str], None] | None = None,
    min_source_relevance: float = 0.55,
    knowledge_root: Path = DEFAULT_CONVERSATION_KNOWLEDGE_ROOT,
    global_knowledge_path: Path | None = None,
    include_global_knowledge: bool = False,
    budget_manager: TokenBudgetManager | None = None,
) -> LocalKnowledgeTools:
    """Build Tool adapters over private and explicitly authorized global stores."""
    from qwopus_agent.integrations.smolagents_tools import (
        build_graph_search_tool,
        build_minirag_search_tool,
    )
    from qwopus_agent.memory import MiniRAG

    budget = budget_manager or TokenBudgetManager()
    # 原因：聊天运行在独立 spawn 进程，不能安全复用 API 进程中的原生向量对象。
    # 作用：每次启用本地知识时只加载当前 conversation_id 的持久化快照。
    private_minirag = MiniRAG(
        storage_path=conversation_knowledge_path(
            knowledge_scope,
            root=knowledge_root,
        )
    )
    private_sources = _knowledge_sources(private_minirag)
    private_source_hints = _mentioned_sources(user_message, private_sources)
    global_sources: tuple[str, ...] | None = ()
    tools: list[Any] = []
    primary_scope: Literal["private", "global", "none"] = "none"

    if private_sources is None or private_sources:
        primary_scope = "private"
        tools.extend(
            [
                build_minirag_search_tool(
                    private_minirag,
                    min_relevance=min_source_relevance,
                    source_hints=private_source_hints,
                    progress_callback=progress_callback,
                    budget_manager=budget,
                ),
                build_graph_search_tool(
                    private_minirag.graph_index,
                    progress_callback=progress_callback,
                    budget_manager=budget,
                ),
            ]
        )

    if include_global_knowledge:
        if global_knowledge_path is None:
            # 原因：账号模式不能再退回进程级 documents.jsonl，否则会跨账号检索。
            # 作用：只有旧的无账号调用保留原路径；正式 API 总是显式传入账号聚合库。
            global_knowledge_path = Path(knowledge_root).parent / "documents.jsonl"
        global_minirag = MiniRAG(
            storage_path=global_knowledge_path
        )
        global_sources = _knowledge_sources(global_minirag)
        global_source_hints = _mentioned_sources(user_message, global_sources)
        if private_sources is None or private_sources:
            # 原因：私有库有来源时必须优先保持会话隔离，全局库只是显式授权的补充范围。
            # 作用：保留 rag_search/graph_search 语义，并以独立名称追加全局工具。
            tools.extend(
                [
                    build_minirag_search_tool(
                        global_minirag,
                        min_relevance=min_source_relevance,
                        source_hints=global_source_hints,
                        progress_callback=progress_callback,
                        budget_manager=budget,
                        tool_name="global_rag_search",
                        description=(
                            "Search the explicitly authorized global MiniRAG store. Use this only "
                            "when current-conversation evidence is insufficient."
                        ),
                    ),
                    build_graph_search_tool(
                        global_minirag.graph_index,
                        progress_callback=progress_callback,
                        budget_manager=budget,
                        tool_name="global_graph_search",
                        description=(
                            "Search explicit relationships in the authorized global knowledge "
                            "graph."
                        ),
                    ),
                ]
            )
        else:
            # 原因：当前会话私库明确为空时，先调用空工具只会浪费 Agent 步数。
            # 作用：将授权全局库提升为标准工具名，让 Prompt 与已安装能力保持一致。
            primary_scope = "global"
            tools.extend(
                [
                    build_minirag_search_tool(
                        global_minirag,
                        min_relevance=min_source_relevance,
                        source_hints=global_source_hints,
                        progress_callback=progress_callback,
                        budget_manager=budget,
                        tool_name="rag_search",
                        description=(
                            "Search the explicitly authorized global MiniRAG store. The current "
                            "conversation has no indexed sources, so this is the primary semantic "
                            "knowledge search for this turn."
                        ),
                    ),
                    build_graph_search_tool(
                        global_minirag.graph_index,
                        progress_callback=progress_callback,
                        budget_manager=budget,
                        tool_name="graph_search",
                        description=(
                            "Search relationships in the explicitly authorized global knowledge "
                            "graph. The current conversation has no indexed sources."
                        ),
                    ),
                ]
            )

    return LocalKnowledgeTools(
        tools,
        private_sources=private_sources,
        global_sources=global_sources,
        primary_scope=primary_scope,
    )


def _mentioned_sources(
    user_message: str,
    sources: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return only indexed source names explicitly present in the current question."""
    if not sources:
        return ()
    normalized_message = user_message.casefold()
    return tuple(
        source
        for source in sources
        if Path(source).name.casefold() in normalized_message
    )


def _knowledge_sources(memory: Any) -> tuple[str, ...] | None:
    """Read a public source inventory, preserving legacy factories as unknown."""
    list_sources = getattr(memory, "list_sources", None)
    if callable(list_sources):
        return tuple(str(source) for source in list_sources())
    legacy_list_sources = getattr(memory, "_list_sources", None)
    if callable(legacy_list_sources):
        return tuple(str(source) for source in legacy_list_sources())
    # 原因：第三方工厂可能尚未实现来源清单；把 unknown 当 empty 会误切到全局库。
    # 作用：None 保留旧的私库优先行为，只有生产清单明确为空时才 fallback。
    return None
