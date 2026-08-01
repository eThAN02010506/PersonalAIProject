"""Prompt builders for smolagents file-analysis runs."""

from __future__ import annotations

from typing import Literal

from qwopus_agent.analysis.pandas_sandbox import PANDAS_SANDBOX_CODE_GUIDANCE
from qwopus_agent.integrations import smolagents_spreadsheets
from qwopus_agent.prompts import smolagents as smolagents_prompts
from qwopus_agent.reports import grounded

_ALL_SOURCE_REQUEST_PATTERN = grounded._ALL_SOURCE_REQUEST_PATTERN


def format_file_analysis_agent_prompt(
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    analysis_mode: str = "question",
    has_collection_summary: bool = False,
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
) -> str:
    """Build the task prompt for the smolagents uploaded-file driver."""
    question = user_question.strip() or "Summarize the uploaded files."
    requires_collection_summary = requires_collection_summary_for_prompt(
        available=has_collection_summary,
        file_count=len(file_names),
        user_question=question,
        analysis_mode=analysis_mode,
    )
    file_list = "\n".join(f"- {file_name}" for file_name in file_names)
    lines = [
        "You are Qwopus-Agent's uploaded-file analysis agent.",
        "Use the available tools to inspect the current uploaded files before answering.",
        "Never invent file content and never return raw Tool Observation text as the final answer.",
        (
            "The final answer must follow the language of the user's question; "
            "if unclear, follow the files' main language."
        ),
        smolagents_prompts.response_detail_instruction(response_detail),
        "Use rag_search only when previously indexed local knowledge is relevant.",
        (
            "For current documents, use document_search for a specific question. Use "
            "document_outline and document_read_section for chapter tasks, and document_summary "
            "for a whole-document summary. "
            "Never assume that the beginning of a file represents the whole document."
        ),
        "Current uploaded files:",
        file_list,
    ]
    if requires_collection_summary:
        lines.append(
            "For a folder-wide task, call document_collection_summary first so every "
            "selected document contributes evidence before drilling into individual files."
        )
        lines.append(
            "Treat every # File block as an isolated source. Copy lesson titles and scripture "
            "references only from that file's SOURCE_FACTS; never add a second remembered "
            "reference, merge neighboring lessons, or invent quotations. If the collection "
            "marker says no explicit rubric was found, state that fact instead of creating "
            "scores or weights."
        )
        if _ALL_SOURCE_REQUEST_PATTERN.search(question):
            lines.append(
                "The user explicitly requested all sources: the document-understanding section "
                "must name and substantively summarize every listed file. If a complete Draft "
                "is requested, write every lesson subsection in full; phrases such as 'the "
                "remaining sections follow the same format' are a failed answer."
            )
    if analysis_mode == "section":
        # 原因：章节分析必须遵守用户在前端选定的范围，不能退回全文泛化总结。
        # 作用：要求 Agent 优先读取受限章节工具，并围绕章节内容组织最终答案。
        lines.append(
            "Analysis mode: selected sections. Use document_read_section and answer only from "
            "the sections available through that scoped tool."
        )
    elif analysis_mode == "full":
        # 原因：长文档全文不能直接进入模型上下文。
        # 作用：强制使用已分层压缩的 document_summary，再按需检索证据补充细节。
        summary_tool = (
            "document_collection_summary"
            if requires_collection_summary
            else "document_summary"
        )
        lines.append(
            f"Analysis mode: whole document. Call {summary_tool} first, then use "
            "document_search only when details need supporting evidence."
        )
    if spreadsheet_names:
        spreadsheet_list = ", ".join(spreadsheet_names)
        required_methods = smolagents_spreadsheets.required_spreadsheet_methods(question)
        spreadsheet_intent_guidance = smolagents_spreadsheets.spreadsheet_intent_guidance(
            question
        )
        lines.extend(
            [
                f"Spreadsheets: {spreadsheet_list}.",
                (
                    "For a spreadsheet, call excel_schema first. If computation is needed, "
                    "prefer excel_statistics for R summary()-style describe, categorical "
                    "frequency tables, missing values, IQR or Z-score outliers, "
                    "group summaries, correlations, covariance, quantiles, normality tests, "
                    "crosstabs, chi-square independence tests, mean confidence intervals, "
                    "and one- or two-sample t-tests. Treat confidence as a statistical "
                    "confidence level, "
                    "not the probability that one realized interval or conclusion is correct. "
                    "For a question about one row, one field, one character stat, one SKU, "
                    "or one named item, use excel_statistics with method='lookup'. "
                    "Use excel_modeling for linear regression or one-way ANOVA with Tukey HSD. "
                    "Use excel_analysis only for a custom calculation that excel_statistics "
                    "cannot express. When entity rows are mixed "
                    "with regional, income, or total aggregates, use a metadata table and the "
                    "excel_statistics scope arguments so each comparison population is homogeneous."
                ),
                PANDAS_SANDBOX_CODE_GUIDANCE,
                spreadsheet_intent_guidance,
                (
                    "Interpret successful spreadsheet Tool results in prose without copying "
                    "their numbers into a new table; the runtime appends the exact local "
                    "GitHub-Flavored Markdown tables to the final answer."
                ),
                (
                    "For spreadsheet questions with a required statistical computation, do not "
                    "call final_answer until excel_schema and every required computation tool "
                    "has completed successfully."
                    if required_methods
                    else "For spreadsheet questions, do not call final_answer before the file "
                    "schema has been inspected."
                ),
                (
                    "Never describe Tukey HSD as an unequal-variance procedure. If Levene's "
                    "test questions equal variances, state that Tukey is exploratory and that "
                    "an unequal-variance post-hoc test was not computed."
                ),
                (
                    "Only when the user requests a general workbook profile without specifying "
                    "another computation, calculate a per-numeric-column table with count, mean, "
                    "standard deviation, minimum, quartiles, median, maximum, and missing values; "
                    "include relevant categorical item counts."
                ),
                (
                    "For a group or item question, return one row per requested group or item "
                    "with its count and relevant computed aggregates."
                ),
                "Never request or reproduce the entire spreadsheet.",
            ]
        )
    lines.extend(
        [
            "",
            f"User question: {question}",
            "",
            "Produce the final answer after using the needed tools.",
        ]
    )
    return "\n".join(lines)


def requires_collection_summary_for_prompt(
    *,
    available: bool,
    file_count: int,
    user_question: str,
    analysis_mode: str,
) -> bool:
    """Require collection coverage only for exhaustive multi-document tasks."""
    if not available or file_count <= 1:
        return False
    # 原因：具体事实问题可逐文件检索；无条件强制 collection 会浪费步骤并导致弱模型失败。
    # 作用：全文模式和明确要求全部来源时仍保证覆盖，其余任务允许按问题选择文件工具。
    return (
        analysis_mode == "full"
        or _ALL_SOURCE_REQUEST_PATTERN.search(user_question) is not None
    )
