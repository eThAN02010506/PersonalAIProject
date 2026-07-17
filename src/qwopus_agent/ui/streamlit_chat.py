"""Streamlit UI for chat and uploaded file analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
)
from qwopus_agent.memory import MiniRAG
from qwopus_agent.reports import GeneratedReport, ReportGenerator
from qwopus_agent.services import UploadedFileInput, analyze_uploaded_files, start_chat_task
from qwopus_agent.utils.conversation_log import append_conversation_event, load_chat_messages
from qwopus_agent.utils.logging_config import configure_runtime_logging, get_logger

logger = get_logger("ui.streamlit_chat")


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_messages()
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_debug_steps" not in st.session_state:
        st.session_state.analysis_debug_steps = []
    if "analysis_report" not in st.session_state:
        st.session_state.analysis_report = None
    if "minirag" not in st.session_state:
        st.session_state.minirag = MiniRAG()
    if "chat_task" not in st.session_state:
        st.session_state.chat_task = None
    if "chat_task_notice" not in st.session_state:
        st.session_state.chat_task_notice = None


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


def _start_user_input(
    user_input: str,
    settings: SmolagentsModelSettings,
    enable_web_search: bool = False,
) -> None:
    logger.info("chat_message_received length=%s", len(user_input))
    st.session_state.messages.append({"role": "user", "content": user_input})
    append_conversation_event(
        "chat_message",
        {"role": "user", "content": user_input},
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    history = st.session_state.messages[:-1]
    # 原因：同步 agent.run() 会阻塞 Streamlit，生成期间无法显示进度或响应停止按钮。
    # 作用：Agent 在独立进程运行，UI 保持可重跑并能安全终止当前本地任务。
    st.session_state.chat_task = start_chat_task(
        user_message=user_input,
        history=history,
        settings=settings,
        enable_web_search=enable_web_search,
    )
    st.rerun()


def _render_chat_notice() -> None:
    notice = st.session_state.get("chat_task_notice")
    if not notice:
        return
    level, message = notice
    st.session_state.chat_task_notice = None
    if level == "warning":
        st.warning(message)
    else:
        st.error(message)


@st.fragment(run_every=1.0)
def _render_chat_progress() -> None:
    task = st.session_state.get("chat_task")
    if task is None:
        return

    phase = task.refresh_phase()
    result = task.poll_result()
    if result is not None:
        st.session_state.chat_task = None
        if result.status == "completed":
            st.session_state.messages.append({"role": "assistant", "content": result.content})
            logger.info("chat_message_completed reply_length=%s", len(result.content))
            append_conversation_event(
                "chat_message",
                {"role": "assistant", "content": result.content},
            )
        else:
            logger.error("chat_call_failed error=%s", result.content)
            st.session_state.chat_task_notice = (
                "error",
                f"对话调用失败：{result.content}",
            )
        st.rerun()

    phase_labels = {
        "connecting": "正在启动 Agent",
        "planning": "正在规划工具调用",
        "searching": "正在通过 Tavily 搜索",
        "generating": "正在生成最终答案",
    }
    label = phase_labels.get(phase, "正在处理")
    with st.status(f"{label} · {task.elapsed_seconds:.0f} 秒", expanded=True):
        st.caption("模型较慢时可以停止当前生成，不会删除已有对话记录。")
        if st.button("停止生成", key="stop_chat_generation", type="secondary"):
            task.cancel()
            st.session_state.chat_task = None
            st.session_state.chat_task_notice = ("warning", "已停止本次生成。")
            logger.info("chat_generation_cancelled")
            st.rerun()


def _render_analysis_result(result: AnalysisResult) -> None:
    if result.llm_analysis:
        st.subheader("分析结果")
        st.markdown(result.llm_analysis)
        _render_analysis_status(result)
    else:
        st.warning("尚未生成最终答案。请确认模型服务在线，并输入分析问题后重新分析。")

    if result.llm_analysis:
        for table_name, dataframe in result.tables.items():
            with st.expander(f"表格结果：{table_name}", expanded=False):
                st.markdown(_dataframe_to_safe_html(dataframe), unsafe_allow_html=True)
        _render_report_downloads(result)


def _render_analysis_status(result: AnalysisResult) -> None:
    """Render safe analysis metadata without exposing raw tool observations."""
    metadata = result.metadata
    file_count = metadata.get("file_count", 1)
    hit_count = metadata.get("minirag_search_hits", 0)
    inserted = metadata.get("minirag_inserted") is True

    # 原因：第九步需要让用户看到分析链路状态，而不是只能看到一段答案。
    # 作用：用轻量指标展示文件、表格、知识层参与情况，不暴露原始检索内容。
    columns = st.columns(4)
    columns[0].metric("文件", str(file_count))
    columns[1].metric("表格结果", str(len(result.tables)))
    columns[2].metric("MiniRAG 命中", str(hit_count))
    columns[3].metric("已入库", "是" if inserted else "否")


def _render_debug_steps(debug_steps: list[str]) -> None:
    """Render document-analysis trace only when the user asks for it."""
    if not debug_steps:
        return

    with st.expander("分析过程", expanded=False):
        for index, step in enumerate(debug_steps, start=1):
            st.markdown(f"**Step {index}.** {step}")


def _generate_analysis_report(result: AnalysisResult) -> GeneratedReport:
    """Generate downloadable report artifacts for one analysis result."""
    title = "Qwopus Analysis Report"
    body = result.llm_analysis or result.markdown_summary
    # 原因：报告下载是分析结果的下游能力，不能把生成逻辑散落在按钮回调里。
    # 作用：把 AnalysisResult 转成 ReportGenerator 的统一输入，便于测试和复用。
    return ReportGenerator(output_dir=Path("storage/reports")).generate(
        title=title,
        markdown_body=body,
        tables=result.tables,
        basename="qwopus_analysis_report",
    )


def _render_report_downloads(result: AnalysisResult) -> None:
    """Render report generation and download controls."""
    if st.button("生成报告", width="stretch"):
        try:
            st.session_state.analysis_report = _generate_analysis_report(result)
            st.success("报告已生成。")
        except Exception as exc:
            logger.exception("report_generation_failed")
            st.error(f"报告生成失败：{exc}")
            return

    report = st.session_state.get("analysis_report")
    if report is None:
        return

    for artifact in report.artifacts:
        if not artifact.path.exists():
            continue
        st.download_button(
            label=f"下载 {artifact.kind.upper()}",
            data=artifact.path.read_bytes(),
            file_name=artifact.path.name,
            mime=_report_mime_type(artifact.kind),
            width="stretch",
        )


def _report_mime_type(kind: str) -> str:
    """Return a browser download MIME type for one report kind."""
    return {
        "markdown": "text/markdown",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "charts": "text/markdown",
        "pdf": "application/pdf",
    }.get(kind, "application/octet-stream")


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


def _render_upload_analysis(settings: SmolagentsModelSettings) -> None:
    st.subheader("文档上传与本地分析")
    st.caption("当前阶段：上传文件 → 本地解析/分析 → MiniRAG 入库/检索 → 页面展示。")

    uploaded_files = st.file_uploader(
        "上传 PDF / DOCX / Markdown / TXT / 图片 / CSV / Excel（可多选）",
        # 原因：MinerU pipeline 支持图片 OCR，上传入口需要开放相同格式。
        # 作用：让图片与其他文档共用解析、MiniRAG 入库和模型分析流程。
        type=[
            "pdf",
            "docx",
            "md",
            "txt",
            "png",
            "jpeg",
            "jpg",
            "csv",
            "xlsx",
            "xls",
        ],
        accept_multiple_files=True,
    )
    user_question = st.text_area(
        "分析问题（可选）",
        placeholder="例如：概括文档重点，或查看表格有哪些字段和数值列。",
        height=90,
    )
    # 原因：主界面默认只呈现最终答案，但用户需要时可以查看可审计的分析过程。
    # 作用：把工具调用、检索命中、模型重试等过程信息放到用户可选区域。
    show_debug = st.checkbox("显示分析过程", value=False)

    if st.button("开始本地分析", type="primary", width="stretch"):
        if not uploaded_files:
            st.warning("请先上传文件。")
            return

        with st.spinner("正在保存并分析文件..."):
            try:
                outcome = analyze_uploaded_files(
                    uploaded_files=[
                        UploadedFileInput(
                            name=uploaded_file.name,
                            content=uploaded_file.getvalue(),
                        )
                        for uploaded_file in uploaded_files
                    ],
                    user_question=user_question,
                    settings=settings,
                    minirag=st.session_state.minirag,
                )
                st.session_state.analysis_result = outcome.result
                st.session_state.analysis_debug_steps = outcome.debug_steps
                st.success(f"已完成分析：{len(outcome.analyzed_file_names)} 个文件")
            except Exception as exc:
                logger.exception("analysis_failed")
                st.error(f"分析失败：{exc}")

    if st.session_state.analysis_result is not None:
        _render_analysis_result(st.session_state.analysis_result)
        if show_debug:
            _render_debug_steps(st.session_state.analysis_debug_steps)


def main() -> None:
    configure_runtime_logging()
    logger.info("streamlit_app_started")
    st.set_page_config(page_title="Qwopus-Agent", page_icon="💬", layout="wide")
    st.title("Qwopus-Agent 本地办公助手")
    st.caption(
        "当前阶段：smolagents 对话 + 文档/Excel 上传分析 + MiniRAG 入库检索。报告生成仍为后续模块。"
    )

    _init_session_state()
    # 原因：用户会在同一个服务器地址上频繁切换模型。
    # 作用：Streamlit 每次重跑都从 /models 刷新侧边栏和后续请求使用的模型 id。
    settings = resolve_model_settings(SmolagentsModelSettings.from_env())
    _render_sidebar(settings)
    analysis_tab, chat_tab = st.tabs(["文档分析", "对话测试"])

    with analysis_tab:
        _render_upload_analysis(settings)

    with chat_tab:
        enable_web_search = st.checkbox("联网搜索", value=False)
        _render_history()
        _render_chat_notice()
        _render_chat_progress()
        user_input = st.chat_input(
            "输入你的问题...",
            disabled=st.session_state.chat_task is not None,
        )
        if user_input:
            _start_user_input(
                user_input,
                settings,
                enable_web_search=enable_web_search,
            )


if __name__ == "__main__":
    main()
