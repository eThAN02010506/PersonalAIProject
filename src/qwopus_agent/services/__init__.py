"""Lazy application-service exports that keep business logic out of UI layers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentOrchestrator": ("agent_orchestrator", "AgentOrchestrator"),
    "AnswerQualityEvaluator": ("answer_quality", "AnswerQualityEvaluator"),
    "AnswerQualityReport": ("answer_quality", "AnswerQualityReport"),
    "AnswerContract": ("orchestration_models", "AnswerContract"),
    "ContextReference": ("orchestration_models", "ContextReference"),
    "ContextSnapshot": ("orchestration_models", "ContextSnapshot"),
    "ConversationTaskState": ("orchestration_models", "ConversationTaskState"),
    "ConversationTurn": ("orchestration_models", "ConversationTurn"),
    "OrchestrationFile": ("orchestration_models", "OrchestrationFile"),
    "OrchestrationRequest": ("orchestration_models", "OrchestrationRequest"),
    "OrchestrationResult": ("orchestration_models", "OrchestrationResult"),
    "ProcessEvent": ("orchestration_models", "ProcessEvent"),
    "ResolvedIntent": ("orchestration_models", "ResolvedIntent"),
    "SourceCitation": ("orchestration_models", "SourceCitation"),
    "UploadedFileInput": ("analysis_service", "UploadedFileInput"),
    "UploadAnalysisOutcome": ("analysis_service", "UploadAnalysisOutcome"),
    "analyze_uploaded_files": ("analysis_service", "analyze_uploaded_files"),
    "combine_analysis_results": ("analysis_service", "combine_analysis_results"),
    "BackgroundChatTask": ("chat_service", "BackgroundChatTask"),
    "ChatTaskResult": ("chat_service", "ChatTaskResult"),
    "start_chat_task": ("chat_service", "start_chat_task"),
    "KnowledgeGraphService": ("knowledge_graph_service", "KnowledgeGraphService"),
    "IntentResolver": ("intent_resolver", "IntentResolver"),
    "build_context_snapshot": ("intent_resolver", "build_context_snapshot"),
    "KnowledgeMaintenanceService": (
        "knowledge_maintenance_service",
        "KnowledgeMaintenanceService",
    ),
    "SkillGrowthDecision": ("skill_growth_service", "SkillGrowthDecision"),
    "SkillGrowthPolicy": ("skill_growth_service", "SkillGrowthPolicy"),
    "SkillGrowthService": ("skill_growth_service", "SkillGrowthService"),
    "SkillAuthoringService": ("skill_authoring_service", "SkillAuthoringService"),
    "SkillRunTrace": ("skill_growth_service", "SkillRunTrace"),
    "SkillTraceStep": ("skill_growth_service", "SkillTraceStep"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one service export without importing every service dependency."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    # 原因：成长或聊天服务不需要在导入时初始化文档分析与 MiniRAG 依赖。
    # 作用：保留稳定 package 导入接口，同时只加载当前调用链需要的服务模块。
    module = import_module(f"qwopus_agent.services.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value
