"""Uploaded-file analysis runner for smolagents integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from qwopus_agent.analysis.pandas_sandbox import PANDAS_SANDBOX_CODE_GUIDANCE
from qwopus_agent.integrations import (
    smolagents_debug,
    smolagents_factory,
    smolagents_file_prompts,
    smolagents_model,
    smolagents_spreadsheets,
)
from qwopus_agent.integrations.smolagents_results import DocumentAnalysisRun
from qwopus_agent.prompts import smolagents as smolagents_prompts
from qwopus_agent.reports import contract as report_contract
from qwopus_agent.reports import grounded
from qwopus_agent.utils.token_budget import TokenBudgetManager, truncate_to_tokens

SmolagentsModelSettings = smolagents_model.SmolagentsModelSettings
build_smolagents_tool_calling_agent = smolagents_factory.build_smolagents_tool_calling_agent
format_file_analysis_agent_prompt = smolagents_file_prompts.format_file_analysis_agent_prompt
_requires_collection_summary = smolagents_file_prompts.requires_collection_summary_for_prompt
_requires_document_summary = smolagents_file_prompts.requires_document_summary_for_prompt
_agent_debug_steps = smolagents_debug.agent_debug_steps
_build_agent_debug_run = smolagents_debug.build_agent_debug_run
_extract_collection_covered_file_names = smolagents_debug.extract_collection_covered_file_names
_extract_agent_tool_calls = smolagents_debug.extract_agent_tool_calls
_extract_final_answer = smolagents_debug.extract_final_answer
_extract_inspected_file_names = smolagents_debug.extract_inspected_file_names
_extract_successful_agent_tool_calls = smolagents_debug.extract_successful_agent_tool_calls
_extract_tool_observations = smolagents_debug.extract_tool_observations
_has_successful_tool_method = smolagents_debug.has_successful_tool_method
_looks_like_tool_observation = smolagents_debug.looks_like_tool_observation
_missing_required_file_tools = smolagents_debug.missing_required_file_tools
_required_file_tools = smolagents_debug.required_file_tools
_unpack_agent_run_result = smolagents_debug.unpack_agent_run_result
_document_evidence_required_answer = smolagents_prompts.document_evidence_required_answer
_has_usable_knowledge_evidence = smolagents_prompts.has_usable_knowledge_evidence
_LOCAL_KNOWLEDGE_TOOLS = smolagents_prompts.LOCAL_KNOWLEDGE_TOOLS
_no_knowledge_evidence_answer = smolagents_prompts.no_knowledge_evidence_answer
_requires_document_evidence = smolagents_prompts.requires_document_evidence
_apply_missing_spreadsheet_fallbacks = smolagents_spreadsheets.apply_missing_spreadsheet_fallbacks
_has_required_spreadsheet_method = smolagents_spreadsheets.has_required_spreadsheet_method
_remove_markdown_tables = smolagents_spreadsheets.remove_markdown_tables
_required_spreadsheet_method = smolagents_spreadsheets.required_spreadsheet_method
_required_spreadsheet_methods = smolagents_spreadsheets.required_spreadsheet_methods
_sanitize_spreadsheet_narrative = smolagents_spreadsheets.sanitize_spreadsheet_narrative
_spreadsheet_computation_summary = smolagents_spreadsheets.spreadsheet_computation_summary
_spreadsheet_intent_guidance = smolagents_spreadsheets.spreadsheet_intent_guidance
_spreadsheet_result_tables = smolagents_spreadsheets.spreadsheet_result_tables
_apply_grounded_report_fallbacks = report_contract._apply_grounded_report_fallbacks
_collection_grounding_evidence = report_contract._collection_grounding_evidence
_is_model_generation_failure_output = report_contract._is_model_generation_failure_output
_lesson_slot_manifest = report_contract._lesson_slot_manifest
_merge_numbered_section_refinement = report_contract._merge_numbered_section_refinement
_missing_requested_sections = report_contract._missing_requested_sections
_report_quality_issues = report_contract._report_quality_issues
_SCRIPTURE_REFERENCE_PATTERN = grounded._SCRIPTURE_REFERENCE_PATTERN
_LessonGroundingSpec = grounded._LessonGroundingSpec
_canonical_lesson_heading = grounded._canonical_lesson_heading
_chinese_integer = grounded._chinese_integer
_collection_manifest_sources = grounded._collection_manifest_sources
_collection_source_blocks = grounded._collection_source_blocks
_grounded_application_claim = grounded._grounded_application_claim
_grounded_evidence_claim = grounded._grounded_evidence_claim
_lesson_answer_aliases = grounded._lesson_answer_aliases
_lesson_answer_label = grounded._lesson_answer_label
_lesson_evidence = grounded._lesson_evidence
_lesson_grounding_specs = grounded._lesson_grounding_specs
_lesson_number_from_label = grounded._lesson_number_from_label
_lesson_scripture = grounded._lesson_scripture
_lesson_topic = grounded._lesson_topic
_normalized_fact_text = grounded._normalized_fact_text
_render_deterministic_grounded_report = grounded._render_deterministic_grounded_report
_render_grounded_checklist = grounded._render_grounded_checklist
_render_grounded_draft_review = grounded._render_grounded_draft_review
_render_grounded_examples = grounded._render_grounded_examples
_render_grounded_full_draft = grounded._render_grounded_full_draft
_render_grounded_lesson_fallback = grounded._render_grounded_lesson_fallback
_render_grounded_outline = grounded._render_grounded_outline
_render_grounded_paragraph_guidance = grounded._render_grounded_paragraph_guidance
_render_grounded_source_inventory = grounded._render_grounded_source_inventory
_render_grounded_strategy = grounded._render_grounded_strategy
_render_grounded_understanding = grounded._render_grounded_understanding
_requested_numbered_sections = grounded._requested_numbered_sections
_scripture_reference_is_supported = grounded._scripture_reference_is_supported
_scripture_reference_key = grounded._scripture_reference_key
_source_answer_label = grounded._source_answer_label
_source_application_excerpt = grounded._source_application_excerpt
_source_evidence_excerpt = grounded._source_evidence_excerpt
_source_fact_values = grounded._source_fact_values
_source_tagged_excerpt = grounded._source_tagged_excerpt
_title_is_source_understanding = grounded._title_is_source_understanding
_title_requires_full_draft = grounded._title_requires_full_draft
_topic_payload = grounded._topic_payload
_validated_grounded_collection = grounded._validated_grounded_collection
should_use_grounded_report_composer = grounded.should_use_grounded_report_composer

def run_smolagents_file_analysis_with_debug(
    file_names: list[str],
    spreadsheet_names: list[str],
    user_question: str,
    tools: list[Any],
    settings: SmolagentsModelSettings | None = None,
    analysis_mode: str = "question",
    response_detail: Literal["concise", "balanced", "detailed"] = "detailed",
    spreadsheet_paths: dict[str, Path] | None = None,
) -> DocumentAnalysisRun:
    """Run uploaded-file analysis through the smolagents ToolCallingAgent."""
    if not file_names:
        raise ValueError("file_names must not be empty.")
    if not tools:
        raise ValueError("At least one file-analysis tool is required.")

    effective_settings = settings or SmolagentsModelSettings.from_env()
    budget = TokenBudgetManager(
        context_window=effective_settings.context_window_tokens,
        output_reserve=effective_settings.max_tokens,
    )
    collection_tools = [
        tool
        for tool in tools
        if getattr(tool, "name", "") == "document_collection_summary"
    ]
    if len(collection_tools) > 1:
        raise ValueError("Only one document_collection_summary tool is allowed.")
    has_collection_summary = bool(collection_tools)
    parser_files = set(file_names).difference(spreadsheet_names)
    requires_collection_summary = _requires_collection_summary(
        available=has_collection_summary,
        file_count=len(parser_files),
        user_question=user_question,
        analysis_mode=analysis_mode,
    )
    requires_document_summary = _requires_document_summary(
        file_count=len(parser_files),
        user_question=user_question,
        analysis_mode=analysis_mode,
    )
    requested_sections = _requested_numbered_sections(user_question)
    prompt = format_file_analysis_agent_prompt(
        file_names=file_names,
        spreadsheet_names=spreadsheet_names,
        user_question=user_question,
        analysis_mode=analysis_mode,
        has_collection_summary=has_collection_summary,
        response_detail=response_detail,
    )
    collection_tool = collection_tools[0] if collection_tools else None
    use_grounded_report_composer = should_use_grounded_report_composer(
        file_names=file_names,
        spreadsheet_names=spreadsheet_names,
        user_question=user_question,
        has_collection_summary=collection_tool is not None,
    )
    if use_grounded_report_composer:
        assert collection_tool is not None
        # 原因：一次 8k-token 长生成在本地大模型切换或显存紧张时会超时/重启，
        # 而且弱模型容易把相邻课程压成一个泛化清单。
        # 作用：完整报告先由 collection Tool 确定性覆盖全部来源，再按逐课事实槽位组装；
        # 这一完整性优先路径不依赖模型长生成，也不会使用文件外知识。
        collection_evidence = str(collection_tool.forward())
        lesson_specs = _validated_grounded_collection(
            file_names=file_names,
            collection_evidence=collection_evidence,
        )
        final_answer = _render_deterministic_grounded_report(
            requested=requested_sections,
            file_names=file_names,
            collection_evidence=collection_evidence,
            lesson_specs=lesson_specs,
        )
        missing_sections = _missing_requested_sections(
            final_answer,
            requested_sections,
        )
        quality_issues = _report_quality_issues(
            answer=final_answer,
            requested=requested_sections,
            file_names=file_names,
            user_question=user_question,
            collection_evidence=collection_evidence,
        )
        if missing_sections or quality_issues:
            raise RuntimeError(
                "Grounded report composer produced an invalid report contract."
            )
        debug_steps = [
            "已用 document_collection_summary 读取并核对全部来源。",
            "长篇全来源任务使用逐来源、逐课确定性报告合成，未调用不稳定的单次长生成。",
        ]
        return DocumentAnalysisRun(
            answer=final_answer,
            debug_steps=debug_steps,
            tool_calls=["document_collection_summary"],
            inspected_file_names=tuple(file_names),
            debug_runs=(
                _build_agent_debug_run(
                    label="grounded_report_composer",
                    prompt=prompt,
                    max_steps=0,
                    state="success",
                    output=final_answer,
                    steps=[],
                ),
            ),
            generation_mode="grounded_composer",
        )

    agent = build_smolagents_tool_calling_agent(
        settings=effective_settings,
        tools=tools,
    )
    # 原因：固定至少八步会让不遵循提示的模型反复读取同一文件，显著增加等待时间。
    # 作用：按文件数提供一次读取和收尾预算，遗漏文件由下方精确校验触发补充轮。
    max_steps = (
        4
        if requires_collection_summary
        else min(max(len(file_names) + 2, 4), 12)
    )
    # 原因：上传分析需要由 smolagents 自己选择解析、RAG 或 Excel 沙箱工具。
    # 作用：返回完整运行状态供可选调试区审计，主界面仍只使用最终 answer。
    run_result = agent.run(
        prompt,
        max_steps=max_steps,
        return_full_result=True,
    )
    answer, state, steps = _unpack_agent_run_result(run_result)
    debug_runs = [
        _build_agent_debug_run(
            label="file_analysis",
            prompt=prompt,
            max_steps=max_steps,
            state=state,
            output=answer,
            steps=steps,
        )
    ]
    tool_calls = _extract_agent_tool_calls(steps)
    successful_tool_calls = _extract_successful_agent_tool_calls(steps)
    all_steps = list(steps)
    debug_steps = _agent_debug_steps(state=state, steps=steps, tool_calls=tool_calls)
    required_tools = _required_file_tools(
        spreadsheet_names=spreadsheet_names,
    )
    if requires_collection_summary:
        # 原因：逐文件调用受 Agent 步数限制，提示语不能保证大型文档集合真的全部进入上下文。
        # 作用：多文档任务必须执行一次带 coverage manifest 的平衡证据 Tool。
        required_tools.add("document_collection_summary")
    elif requires_document_summary and parser_files:
        # 原因：整体文档问题如果只用 search，容易只根据局部命中片段回答。
        # 作用：要求先取得分层摘要，再允许用检索补充细节。
        required_tools.add("document_summary")
    if analysis_mode == "section" and parser_files:
        # 原因：章节模式的边界来自用户选择的章节，不应由全文搜索替代。
        # 作用：把 document_read_section 作为验收条件，确保回答来源受 scoped tool 限定。
        required_tools.add("document_read_section")
    missing_tools = _missing_required_file_tools(
        spreadsheet_names=spreadsheet_names,
        required_tools=required_tools,
        successful_tool_calls=successful_tool_calls,
    )
    required_spreadsheet_methods = _required_spreadsheet_methods(user_question)
    for required_spreadsheet_method in required_spreadsheet_methods:
        if (
            spreadsheet_names
            and not _has_required_spreadsheet_method(
                all_steps,
                user_question=user_question,
                required_method=required_spreadsheet_method,
            )
        ):
            missing_tools.add(required_spreadsheet_method[0])
    inspected_files = _extract_inspected_file_names(steps)
    inspected_files.update(_extract_collection_covered_file_names(steps))
    missing_files = set(file_names).difference(inspected_files)

    final_answer = _extract_final_answer(answer)
    if _is_model_generation_failure_output(final_answer):
        # 原因：smolagents 在最终模型请求失败时会把错误包装成普通 text content。
        # 作用：保留完整 debug run 和已检查来源，但不让 transport error 成为用户答案。
        debug_steps.append("模型最终答案生成失败；错误详情仅保留在 Debug Console。")
        return DocumentAnalysisRun(
            answer="",
            debug_steps=debug_steps,
            tool_calls=list(dict.fromkeys(tool_calls)),
            inspected_file_names=tuple(
                file_name for file_name in file_names if file_name in inspected_files
            ),
            debug_runs=tuple(debug_runs),
        )
    missing_sections = _missing_requested_sections(
        final_answer,
        requested_sections,
    )
    collection_evidence = _collection_grounding_evidence(all_steps)
    lesson_specs = _lesson_grounding_specs(file_names, collection_evidence)
    quality_issues = _report_quality_issues(
        answer=final_answer,
        requested=requested_sections,
        file_names=file_names,
        user_question=user_question,
        collection_evidence=collection_evidence,
    )
    refinement_numbers = set(missing_sections).union(quality_issues)
    refinement_sections = {
        number: title
        for number, title in requested_sections.items()
        if number in refinement_numbers
    }
    if (
        not final_answer
        or _looks_like_tool_observation(final_answer)
        or missing_tools
        or missing_files
        or refinement_sections
    ):
        # 原因：少数模型会把最后一次 Tool Observation 当作回答，或者在步数内没有调用 final_answer。
        # 作用：保留同一个 Agent memory 再收敛一轮，禁止原始工具输出进入 Streamlit 主结果。
        section_only_refinement = bool(
            final_answer
            and not _looks_like_tool_observation(final_answer)
            and not missing_tools
            and not missing_files
            and refinement_sections
        )
        if missing_tools:
            missing_names = ", ".join(sorted(missing_tools))
            debug_steps.append(f"Agent 尚未调用必要 Tool：{missing_names}；触发补充执行。")
        elif section_only_refinement:
            debug_steps.append(
                "Agent 报告仅有部分章节缺失或不足；只补齐目标章节并保留其余答案。"
            )
        else:
            debug_steps.append("Agent 尚未形成最终答案，保留工具上下文后触发收敛步骤。")
        missing_tool_instruction = ", ".join(sorted(missing_tools)) or "none"
        required_method_instruction = (
            ", ".join(".".join(method) for method in required_spreadsheet_methods)
            if required_spreadsheet_methods
            else "infer the appropriate reviewed method from the user question"
        )
        spreadsheet_intent_guidance = _spreadsheet_intent_guidance(user_question)
        missing_file_instruction = ", ".join(sorted(missing_files)) or "none"
        missing_section_instruction = (
            ", ".join(
                f"{number}. {title}" for number, title in refinement_sections.items()
            )
            or "none"
        )
        quality_issue_instruction = (
            "; ".join(
                f"{number}. {' | '.join(messages)}"
                for number, messages in quality_issues.items()
            )
            or "none"
        )
        lesson_slot_instruction = _lesson_slot_manifest(lesson_specs)
        if section_only_refinement:
            grounded_context = truncate_to_tokens(
                collection_evidence,
                budget.synthesis_budget,
            )
            retry_prompt = (
                "The previous answer is mostly complete. Do not rewrite, summarize, or repeat "
                "any accepted section because the runtime will retain it verbatim. Return ONLY "
                "the following missing or underdeveloped numbered sections, each with its exact "
                f"Markdown number and title: {missing_section_instruction}. "
                "Fully develop those sections from the existing inspected-file evidence. "
                "Correct every grounded-deliverable defect listed here: "
                f"{quality_issue_instruction}. "
                "Use SOURCE_FACTS as the only authority for lesson titles and scripture "
                "references. If QWOPUS_EXPLICIT_RUBRIC_FOUND is false, explicitly say that no "
                "rubric was supplied and do not invent points, weights, or totals. "
                "Never use placeholders such as '略', 'omitted', or 'to be completed'. "
                "Do not add a preface, conclusion, Observation, Thought, tool log, or any "
                "section not listed above. Follow the language of the user's question.\n\n"
                f"{lesson_slot_instruction}\n\n"
                "Grounding evidence from the completed collection read follows. Treat each "
                "# File block as isolated and use no outside knowledge:\n"
                f"{grounded_context}"
            )
        else:
            retry_prompt = (
                "Continue from the existing tool observations and answer the original "
                "user question now. "
                f"Before answering, call every missing required tool: {missing_tool_instruction}. "
                f"Required spreadsheet computation: {required_method_instruction}. "
                f"{spreadsheet_intent_guidance} "
                "Inspect every missing file with document_summary, document_search, "
                "document_read_section, or document_collection_summary: "
                f"{missing_file_instruction}. "
                "For each missing spreadsheet, call excel_schema, then prefer "
                "excel_statistics for a supported common method, or excel_modeling for "
                "regression and ANOVA; use excel_analysis only for a custom computation. "
                "Generate restricted pandas code only when using "
                "excel_analysis. "
                f"{PANDAS_SANDBOX_CODE_GUIDANCE} "
                "The final answer for spreadsheet work must include at least one GitHub-Flavored "
                "Markdown table, which the runtime will append from successful local Tool "
                "output. Do not transcribe or reformat numeric Tool results into a new table. "
                "Rewrite the complete answer and fully deliver every requested numbered section; "
                f"missing or underdeveloped sections: {missing_section_instruction}. "
                "Correct every grounded-deliverable defect listed here: "
                f"{quality_issue_instruction}. "
                "Never use placeholders such as '略', 'omitted', or 'to be completed'. "
                "Return only a complete natural-language final "
                "answer. Do not repeat Observation, tool output, Thought, code drafts, or "
                "internal steps. Follow the language of the user's question."
            )
        if section_only_refinement and collection_evidence:
            # 原因：原 Agent memory 已累计工具结果、outline 和旧 Draft；弱模型会在长上下文里
            # 重复相邻课次并漏掉某个课次。
            # 作用：只把受控 collection evidence 和精确课次槽位交给无工具的新 Agent，
            # 接受章节仍由确定性 merge 保留。
            retry_agent = build_smolagents_tool_calling_agent(
                settings=effective_settings,
                tools=[],
            )
            retry_max_steps = 2
            retry_result = retry_agent.run(
                retry_prompt,
                max_steps=retry_max_steps,
                return_full_result=True,
            )
        else:
            retry_agent = agent
            retry_max_steps = min(
                max(len(missing_files) + len(missing_tools) + 2, 3),
                12,
            )
            retry_result = retry_agent.run(
                retry_prompt,
                reset=False,
                max_steps=retry_max_steps,
                return_full_result=True,
            )
        retry_answer, retry_state, retry_steps = _unpack_agent_run_result(retry_result)
        debug_runs.append(
            _build_agent_debug_run(
                label=(
                    "file_analysis_section_refinement"
                    if section_only_refinement
                    else "file_analysis_refinement"
                ),
                prompt=retry_prompt,
                max_steps=retry_max_steps,
                state=retry_state,
                output=retry_answer,
                steps=retry_steps,
            )
        )
        retry_tool_calls = _extract_agent_tool_calls(retry_steps)
        tool_calls.extend(retry_tool_calls)
        all_steps.extend(retry_steps)
        successful_tool_calls.extend(
            _extract_successful_agent_tool_calls(retry_steps)
        )
        inspected_files.update(_extract_inspected_file_names(retry_steps))
        inspected_files.update(_extract_collection_covered_file_names(retry_steps))
        debug_steps.extend(
            _agent_debug_steps(
                state=retry_state,
                steps=retry_steps,
                tool_calls=retry_tool_calls,
                prefix="收敛轮",
            )
        )
        retry_final_answer = _extract_final_answer(retry_answer)
        if section_only_refinement:
            if not _looks_like_tool_observation(retry_final_answer):
                final_answer = _merge_numbered_section_refinement(
                    final_answer,
                    retry_final_answer,
                    requested_sections,
                    refinement_sections,
                    lesson_specs,
                )
            final_answer = _apply_grounded_report_fallbacks(
                answer=final_answer,
                refinement=retry_final_answer,
                requested=requested_sections,
                target_sections=refinement_sections,
                quality_issues=quality_issues,
                file_names=file_names,
                collection_evidence=collection_evidence,
                lesson_specs=lesson_specs,
            )
        else:
            final_answer = retry_final_answer
        missing_sections = _missing_requested_sections(
            final_answer,
            requested_sections,
        )
        collection_evidence = _collection_grounding_evidence(all_steps)
        lesson_specs = _lesson_grounding_specs(file_names, collection_evidence)
        quality_issues = _report_quality_issues(
            answer=final_answer,
            requested=requested_sections,
            file_names=file_names,
            user_question=user_question,
            collection_evidence=collection_evidence,
        )

    missing_tools = _missing_required_file_tools(
        spreadsheet_names=spreadsheet_names,
        required_tools=required_tools,
        successful_tool_calls=successful_tool_calls,
    )
    for required_spreadsheet_method in required_spreadsheet_methods:
        if (
            spreadsheet_names
            and not _has_required_spreadsheet_method(
                all_steps,
                user_question=user_question,
                required_method=required_spreadsheet_method,
            )
        ):
            missing_tools.add(required_spreadsheet_method[0])
    fallback_answer = _apply_missing_spreadsheet_fallbacks(
        all_steps,
        spreadsheet_paths=spreadsheet_paths or {},
        spreadsheet_names=spreadsheet_names,
        user_question=user_question,
        missing_tools=missing_tools,
        required_spreadsheet_methods=required_spreadsheet_methods,
        debug_steps=debug_steps,
    )
    if fallback_answer:
        final_answer = fallback_answer
        tool_calls.append("excel_statistics")
        successful_tool_calls.append("excel_statistics")
    if missing_tools:
        missing_names = ", ".join(sorted(missing_tools))
        raise RuntimeError(f"smolagents did not call required file tools: {missing_names}.")
    missing_files = set(file_names).difference(inspected_files)
    if missing_files:
        missing_names = ", ".join(sorted(missing_files))
        raise RuntimeError(f"smolagents did not inspect uploaded files: {missing_names}.")
    if missing_sections:
        missing_names = ", ".join(
            f"{number}. {title}" for number, title in missing_sections.items()
        )
        raise RuntimeError(
            "smolagents did not complete requested report sections: "
            f"{missing_names}."
        )
    if quality_issues:
        issue_names = "; ".join(
            f"{number}. {' | '.join(messages)}"
            for number, messages in quality_issues.items()
        )
        raise RuntimeError(
            "smolagents did not satisfy the grounded report contract: "
            f"{issue_names}."
        )
    if spreadsheet_names:
        computed_tables = _spreadsheet_result_tables(all_steps)
        if not computed_tables:
            raise RuntimeError(
                "smolagents did not include a computed spreadsheet table in the final answer."
            )
        # 原因：较弱模型会在抄写 Tool 数值时改变区间端点、自由度或 p 值。
        # 作用：移除模型重排的表格并呈现本地 Skill 原表；解释仍由 Agent 生成。
        narrative = _sanitize_spreadsheet_narrative(
            _remove_markdown_tables(final_answer),
            required_method=(
                required_spreadsheet_methods[0] if required_spreadsheet_methods else None
            ),
            use_chinese=any("\u4e00" <= character <= "\u9fff" for character in user_question),
        ).strip()
        computation_summary = _spreadsheet_computation_summary(
            all_steps,
            user_question=user_question,
            use_chinese=any("\u4e00" <= character <= "\u9fff" for character in user_question),
        )
        if computation_summary and computation_summary not in narrative:
            narrative = f"{narrative}\n\n{computation_summary}".strip()
        rendered_tables = "\n\n".join(
            f"### Local table {index}\n\n{table}"
            for index, table in enumerate(computed_tables, start=1)
        )
        table_heading = (
            "## 本地计算表格"
            if any("\u4e00" <= character <= "\u9fff" for character in user_question)
            else "## Local calculation table"
        )
        final_answer = (
            f"{narrative}\n\n"
            f"{table_heading}\n\n"
            f"{rendered_tables}"
        ).strip()
    if not final_answer or _looks_like_tool_observation(final_answer):
        raise RuntimeError("smolagents did not produce a final answer after tool execution.")

    return DocumentAnalysisRun(
        answer=final_answer,
        debug_steps=debug_steps,
        tool_calls=list(dict.fromkeys(tool_calls)),
        inspected_file_names=tuple(
            file_name for file_name in file_names if file_name in inspected_files
        ),
        debug_runs=tuple(debug_runs),
    )
