"""smolagents runtime integration for Qwopus-Agent.

This module connects Qwopus-Agent with an OpenAI-compatible local LLM server
such as optiq serve / mlx_lm.server.
"""

from __future__ import annotations

from typing import Any

from qwopus_agent.integrations import (
    smolagents_answering,
    smolagents_chat_runner,
    smolagents_debug,
    smolagents_direct_chat,
    smolagents_factory,
    smolagents_file_prompts,
    smolagents_file_runner,
    smolagents_knowledge,
    smolagents_model,
    smolagents_results,
    smolagents_spreadsheets,
)
from qwopus_agent.prompts import smolagents as smolagents_prompts
from qwopus_agent.reports import contract as report_contract
from qwopus_agent.reports import grounded
from qwopus_agent.skills import SkillRegistry

AgentDebugRun = smolagents_debug.AgentDebugRun
ChatAgentRun = smolagents_results.ChatAgentRun
DocumentAnalysisRun = smolagents_results.DocumentAnalysisRun
SmolagentsDependencyError = smolagents_factory.SmolagentsDependencyError
SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings
build_smolagents_code_agent = smolagents_factory.build_smolagents_code_agent
build_smolagents_model = smolagents_factory.build_smolagents_model
build_smolagents_tool_calling_agent = (
    smolagents_factory.build_smolagents_tool_calling_agent
)
check_model_connection = smolagents_model.check_model_connection
run_smolagents_smoke_test = smolagents_factory.run_smolagents_smoke_test
run_smolagents_chat_turn = smolagents_direct_chat.run_smolagents_chat_turn
resolve_model_settings = smolagents_model.resolve_model_settings
ChatMessage = smolagents_prompts.ChatMessage
build_chat_messages = smolagents_prompts.build_chat_messages
format_agent_chat_prompt = smolagents_prompts.format_agent_chat_prompt
format_chat_prompt = smolagents_prompts.format_chat_prompt
format_file_analysis_agent_prompt = (
    smolagents_file_prompts.format_file_analysis_agent_prompt
)
_requires_collection_summary = (
    smolagents_file_prompts.requires_collection_summary_for_prompt
)
LocalKnowledgeTools = smolagents_knowledge.LocalKnowledgeTools
build_local_knowledge_tools = smolagents_knowledge.build_local_knowledge_tools
build_browser_open_tool = smolagents_knowledge.build_browser_open_tool
build_tavily_search_tool = smolagents_knowledge.build_tavily_search_tool


def _sync_chat_runner_dependencies() -> None:
    """Keep legacy runtime monkeypatch points active after moving chat orchestration.

    原因：测试和第三方代码仍可能 patch 本模块导出的 Tool/Agent 工厂。
    作用：业务逻辑已拆到 chat_runner，但旧扩展点不会意外触发真实外部依赖。
    """
    smolagents_chat_runner.configure_runtime_dependencies(
        browser_tool_builder=build_browser_open_tool,
        local_knowledge_tool_builder=build_local_knowledge_tools,
        tavily_search_tool_builder=build_tavily_search_tool,
        tool_calling_agent_builder=build_smolagents_tool_calling_agent,
        skill_registry_discover=SkillRegistry.discover,
    )


def run_agent_chat_turn(*args: Any, **kwargs: Any) -> str:
    """Compatibility wrapper for the split chat runner."""
    _sync_chat_runner_dependencies()
    return smolagents_chat_runner.run_agent_chat_turn(*args, **kwargs)


def run_agent_chat_turn_with_debug(*args: Any, **kwargs: Any) -> ChatAgentRun:
    """Compatibility wrapper for the split chat runner."""
    _sync_chat_runner_dependencies()
    return smolagents_chat_runner.run_agent_chat_turn_with_debug(*args, **kwargs)

_agent_debug_steps = smolagents_debug.agent_debug_steps
_build_agent_debug_run = smolagents_debug.build_agent_debug_run
_extract_collection_covered_file_names = (
    smolagents_debug.extract_collection_covered_file_names
)
_extract_agent_observations = smolagents_debug.extract_agent_observations
_extract_tool_observations = smolagents_debug.extract_tool_observations
_extract_agent_tool_calls = smolagents_debug.extract_agent_tool_calls
_extract_final_answer = smolagents_debug.extract_final_answer
_extract_inspected_file_names = smolagents_debug.extract_inspected_file_names
_extract_successful_agent_tool_calls = (
    smolagents_debug.extract_successful_agent_tool_calls
)
_has_successful_tool_method = smolagents_debug.has_successful_tool_method
_looks_like_tool_observation = smolagents_debug.looks_like_tool_observation
_missing_required_file_tools = smolagents_debug.missing_required_file_tools
_required_file_tools = smolagents_debug.required_file_tools
_unpack_agent_run_result = smolagents_debug.unpack_agent_run_result
_document_evidence_required_answer = (
    smolagents_prompts.document_evidence_required_answer
)
_has_usable_knowledge_evidence = smolagents_prompts.has_usable_knowledge_evidence
_LOCAL_KNOWLEDGE_TOOLS = smolagents_prompts.LOCAL_KNOWLEDGE_TOOLS
_no_knowledge_evidence_answer = smolagents_prompts.no_knowledge_evidence_answer
_requires_document_evidence = smolagents_prompts.requires_document_evidence
_answer_has_grounded_source = smolagents_answering.answer_has_grounded_source
_answer_quality_issues = smolagents_answering.answer_quality_issues
_build_answer_quality_checks = smolagents_answering.build_answer_quality_checks
_role_refinement_prompt = smolagents_answering.role_refinement_prompt
_apply_missing_spreadsheet_fallbacks = (
    smolagents_spreadsheets.apply_missing_spreadsheet_fallbacks
)
_has_required_spreadsheet_method = (
    smolagents_spreadsheets.has_required_spreadsheet_method
)
_remove_markdown_tables = smolagents_spreadsheets.remove_markdown_tables
_required_spreadsheet_method = smolagents_spreadsheets.required_spreadsheet_method
_required_spreadsheet_methods = smolagents_spreadsheets.required_spreadsheet_methods
_sanitize_spreadsheet_narrative = (
    smolagents_spreadsheets.sanitize_spreadsheet_narrative
)
_spreadsheet_computation_summary = (
    smolagents_spreadsheets.spreadsheet_computation_summary
)
_spreadsheet_intent_guidance = smolagents_spreadsheets.spreadsheet_intent_guidance
_spreadsheet_result_tables = smolagents_spreadsheets.spreadsheet_result_tables
_apply_grounded_report_fallbacks = (
    report_contract._apply_grounded_report_fallbacks
)
_collection_grounding_evidence = report_contract._collection_grounding_evidence
_is_model_generation_failure_output = (
    report_contract._is_model_generation_failure_output
)
_merge_numbered_section_refinement = (
    report_contract._merge_numbered_section_refinement
)
_missing_requested_sections = report_contract._missing_requested_sections
_report_quality_issues = report_contract._report_quality_issues
should_use_grounded_report_composer = grounded.should_use_grounded_report_composer


def run_smolagents_file_analysis_with_debug(*args: Any, **kwargs: Any) -> DocumentAnalysisRun:
    """Compatibility wrapper for the split uploaded-file runner."""
    return smolagents_file_runner.run_smolagents_file_analysis_with_debug(*args, **kwargs)
