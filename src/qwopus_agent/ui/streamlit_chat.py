"""Streamlit UI for chat and uploaded file analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from qwopus_agent.analysis import AnalysisResult, analyze_uploaded_file
from qwopus_agent.documents import save_uploaded_bytes
from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsDependencyError,
    SmolagentsModelSettings,
    check_model_connection,
    run_smolagents_document_analysis_with_debug,
    run_smolagents_chat_turn,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.utils.conversation_log import append_conversation_event, load_chat_messages


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_messages()
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_debug_steps" not in st.session_state:
        st.session_state.analysis_debug_steps = []
    if "minirag" not in st.session_state:
        st.session_state.minirag = MiniRAG()


def _render_sidebar(settings: SmolagentsModelSettings) -> None:
    with st.sidebar:
        st.header("模型配置")
        st.text(f"模型：{settings.model_id}")
        st.text(f"地址：{settings.base_url}")

        if st.button("检测模型连接", width="stretch"):
            online, message = check_model_connection(settings)
            if online:
                st.success(message)
            else:
                st.error(message)

        if st.button("清空对话", width="stretch"):
            st.session_state.messages = []
            st.rerun()


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_user_input(user_input: str, settings: SmolagentsModelSettings) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    append_conversation_event(
        "chat_message",
        {"role": "user", "content": user_input},
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("正在调用本地大模型..."):
            try:
                online, connection_message = check_model_connection(settings)
                if not online:
                    raise ConnectionError(connection_message)

                history = st.session_state.messages[:-1]
                reply = run_smolagents_chat_turn(
                    user_message=user_input,
                    history=history,
                    settings=settings,
                )

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                append_conversation_event(
                    "chat_message",
                    {"role": "assistant", "content": reply},
                )
            except SmolagentsDependencyError as exc:
                st.error(str(exc))
            except ConnectionError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"对话调用失败：{exc}")


def _render_analysis_result(result: AnalysisResult) -> None:
    if result.llm_analysis:
        st.subheader("分析结果")
        st.markdown(result.llm_analysis)
    else:
        st.warning("尚未生成最终答案。请确认模型服务在线，并输入分析问题后重新分析。")

    if result.llm_analysis:
        for table_name, dataframe in result.tables.items():
            with st.expander(f"表格结果：{table_name}", expanded=False):
                st.markdown(_dataframe_to_safe_html(dataframe), unsafe_allow_html=True)


def _render_debug_steps(debug_steps: list[str]) -> None:
    """Render document-analysis trace only when the user asks for it."""
    if not debug_steps:
        return

    with st.expander("调试过程", expanded=True):
        for index, step in enumerate(debug_steps, start=1):
            st.markdown(f"**Step {index}.** {step}")


def _dataframe_to_safe_html(dataframe) -> str:
    """Render dataframe as HTML without using Streamlit's Arrow dataframe path."""
    safe = dataframe.copy()
    for column in safe.columns:
        if safe[column].dtype == "object":
            # 原因：Streamlit dataframe 会经过 pyarrow，当前环境的 libarrow 会 segfault。
            # 作用：先把 object 列转字符串，再用 pandas HTML 渲染绕开 Arrow native 层。
            safe[column] = safe[column].astype(str)
    html = safe.to_html(index=False, escape=True, border=0)
    html = html.replace('<table border="0" class="dataframe">', '<table class="qwopus-table">')
    return (
        "<div style='overflow-x:auto'>"
        "<style>"
        "table.qwopus-table{border-collapse:collapse;width:100%;font-size:0.9rem;}"
        "table.qwopus-table th,table.qwopus-table td{border:1px solid #e5e7eb;"
        "padding:0.4rem 0.55rem;text-align:left;vertical-align:top;}"
        "table.qwopus-table th{background:#f8fafc;font-weight:600;}"
        "</style>"
        f"{html}"
        f"<p style='color:#64748b;font-size:0.8rem'>Rows: {escape(str(len(safe)))}, "
        f"Columns: {escape(str(len(safe.columns)))}</p>"
        "</div>"
    )


def _format_memory_context(memory_results: list[str], max_chars: int = 4000) -> str:
    """Build bounded MiniRAG context for LLM analysis."""
    if not memory_results:
        return ""

    sections: list[str] = []
    remaining = max_chars
    for index, document in enumerate(memory_results, start=1):
        if remaining <= 0:
            break
        snippet = document[:remaining]
        # 原因：MiniRAG 可能返回长文档，不能无界加入 LLM 上下文。
        # 作用：只附加有限检索片段，让回答能利用知识层但不爆上下文。
        sections.append(f"### MiniRAG Result {index}\n\n{snippet}")
        remaining -= len(snippet)
    return "\n\n".join(sections)


def _merge_analysis_context(document_context: str, memory_context: str) -> str:
    """Merge current file context with MiniRAG search context."""
    if not memory_context:
        return document_context
    return (
        f"{document_context}\n\n"
        "## MiniRAG Search Context\n\n"
        f"{memory_context}"
    )


def _combine_analysis_results(
        results: list[tuple[str, AnalysisResult]],
) -> AnalysisResult:
    """Combine multiple uploaded-file analysis results."""
    markdown_sections: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    metadata_files: list[dict[str, object]] = []
    document_sections: list[str] = []

    for file_name, result in results:
        markdown_sections.append(f"## File: {file_name}\n\n{result.markdown_summary}")
        metadata_files.append(
            {
                "file_name": file_name,
                "metadata": result.metadata,
            }
        )
        if result.markdown_document:
            # 原因：多个上传文件需要合成一个 LLM 上下文，但仍要保留来源。
            # 作用：用文件名分隔每个 Markdown/Excel 安全摘要。
            document_sections.append(f"# File: {file_name}\n\n{result.markdown_document}")
        for table_name, dataframe in result.tables.items():
            safe_file_name = Path(file_name).stem or "file"
            tables[f"{safe_file_name}::{table_name}"] = dataframe

    return AnalysisResult(
        markdown_summary="\n\n".join(markdown_sections),
        tables=tables,
        metadata={
            "source_type": "multi_upload",
            "file_count": len(results),
            "files": metadata_files,
        },
        markdown_document="\n\n".join(document_sections),
    )


def _render_upload_analysis(settings: SmolagentsModelSettings) -> None:
    st.subheader("文档上传与本地分析")
    st.caption("当前阶段：上传文件 → 本地解析/分析 → MiniRAG 入库/检索 → 页面展示。")

    uploaded_files = st.file_uploader(
        "上传 PDF / DOCX / Markdown / TXT / CSV / Excel（可多选）",
        type=["pdf", "docx", "md", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    user_question = st.text_area(
        "分析问题（可选）",
        placeholder="例如：概括文档重点，或查看表格有哪些字段和数值列。",
        height=90,
    )
    show_debug = st.checkbox("显示调试过程", value=True)

    if st.button("开始本地分析", type="primary", width="stretch"):
        if not uploaded_files:
            st.warning("请先上传文件。")
            return

        with st.spinner("正在保存并分析文件..."):
            try:
                debug_steps = []
                analyzed_results: list[tuple[str, AnalysisResult]] = []
                for uploaded_file in uploaded_files:
                    stored = save_uploaded_bytes(uploaded_file.name, uploaded_file.getvalue())
                    debug_steps.extend(
                        [
                            f"文件已保存：{stored.original_name}",
                            f"保存路径：{stored.path}",
                        ]
                    )
                    result = analyze_uploaded_file(stored.path, user_question=user_question)
                    debug_steps.append(f"本地解析完成：{stored.original_name}: {result.metadata}")
                    analyzed_results.append((stored.original_name, result))

                result = _combine_analysis_results(analyzed_results)
                memory_context = ""
                if result.markdown_document:
                    # 原因：上传后的 Markdown/Excel 安全摘要需要进入统一知识层。
                    # 作用：后续分析可以通过 MiniRAG.search(query) 复用已上传内容。
                    st.session_state.minirag.insert(result.markdown_document)
                    debug_steps.append("MiniRAG 入库完成：已插入当前文件的 Markdown/安全摘要。")
                    if user_question.strip():
                        memory_results = st.session_state.minirag.search(user_question)
                        memory_context = _format_memory_context(memory_results)
                        debug_steps.append(f"MiniRAG 检索完成：命中 {len(memory_results)} 条。")
                if user_question.strip() and result.markdown_document:
                    online, connection_message = check_model_connection(settings)
                    debug_steps.append(f"模型连接检测：{connection_message}")
                    if not online:
                        st.warning(f"模型未连接，仅展示本地解析结果：{connection_message}")
                    else:
                        # 原因：当前问题需要定位 Agent 是否停在 Observation。
                        # 作用：拿到最终答案的同时，把每一步模型链路展示给调试面板。
                        analysis_run = run_smolagents_document_analysis_with_debug(
                            document_name=", ".join(file_name for file_name, _ in analyzed_results),
                            content=_merge_analysis_context(
                                result.markdown_document,
                                memory_context,
                            ),
                            user_question=user_question,
                            settings=settings,
                        )
                        debug_steps.extend(analysis_run.debug_steps)
                        result = AnalysisResult(
                            markdown_summary=result.markdown_summary,
                            tables=result.tables,
                            metadata=result.metadata,
                            markdown_document=result.markdown_document,
                            llm_analysis=analysis_run.answer,
                        )
                        append_conversation_event(
                            "analysis",
                            {
                                "files": [file_name for file_name, _ in analyzed_results],
                                "question": user_question,
                                "answer": analysis_run.answer,
                            },
                        )
                elif not user_question.strip():
                    debug_steps.append("未输入分析问题，因此没有调用模型生成最终答案。")
                elif not result.markdown_document:
                    debug_steps.append("本地解析没有得到 Markdown 文档内容，因此没有调用模型。")
                st.session_state.analysis_result = result
                st.session_state.analysis_debug_steps = debug_steps
                st.success(f"已完成分析：{len(analyzed_results)} 个文件")
            except Exception as exc:
                st.error(f"分析失败：{exc}")

    if st.session_state.analysis_result is not None:
        _render_analysis_result(st.session_state.analysis_result)
        if show_debug:
            _render_debug_steps(st.session_state.analysis_debug_steps)


def main() -> None:
    st.set_page_config(page_title="Qwopus-Agent", page_icon="💬", layout="wide")
    st.title("Qwopus-Agent 本地办公助手")
    st.caption("当前阶段：smolagents 对话 + 文档/Excel 上传分析 + MiniRAG 入库检索。报告生成仍为后续模块。")

    _init_session_state()
    settings = SmolagentsModelSettings.from_env()
    _render_sidebar(settings)
    analysis_tab, chat_tab = st.tabs(["文档分析", "对话测试"])

    with analysis_tab:
        _render_upload_analysis(settings)

    with chat_tab:
        _render_history()
        user_input = st.chat_input("输入你的问题...")
        if user_input:
            _handle_user_input(user_input, settings)


if __name__ == "__main__":
    main()
