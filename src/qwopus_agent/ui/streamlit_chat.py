"""Streamlit UI for chat and uploaded file analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from qwopus_agent.analysis import AnalysisResult
from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
)
from qwopus_agent.llm.openai_compatible import OpenAICompatibleLLM
from qwopus_agent.memory import MiniRAG
from qwopus_agent.memory.graph_extraction import (
    CompositeGraphExtractor,
    LLMGraphExtractor,
    RuleBasedGraphExtractor,
)
from qwopus_agent.reports import GeneratedReport, ReportGenerator

# 原因：Streamlit 热重载期间 services 包可能保留半初始化状态，导致新导出暂时不可见。
# 作用：UI 直接依赖具体服务模块，避免通过 package barrel 读取缓存属性。
from qwopus_agent.services.agent_orchestrator import AgentOrchestrator
from qwopus_agent.services.chat_service import start_chat_task
from qwopus_agent.services.knowledge_graph_service import KnowledgeGraphService
from qwopus_agent.services.knowledge_maintenance_service import KnowledgeMaintenanceService
from qwopus_agent.services.orchestration_models import (
    OrchestrationFile,
    OrchestrationRequest,
)
from qwopus_agent.utils.conversation_log import append_conversation_event, load_chat_messages
from qwopus_agent.utils.logging_config import configure_runtime_logging, get_logger

logger = get_logger("ui.streamlit_chat")


def _init_session_state(settings: SmolagentsModelSettings) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_messages()
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_debug_steps" not in st.session_state:
        st.session_state.analysis_debug_steps = []
    if "analysis_report" not in st.session_state:
        st.session_state.analysis_report = None
    if "minirag" not in st.session_state:
        st.session_state.minirag = MiniRAG(
            graph_extractor=_build_graph_extractor(settings)
        )
    if "chat_task" not in st.session_state:
        st.session_state.chat_task = None
    if "chat_task_notice" not in st.session_state:
        st.session_state.chat_task_notice = None
    if "chat_trace" not in st.session_state:
        st.session_state.chat_trace = []


def _build_graph_extractor(settings: SmolagentsModelSettings) -> CompositeGraphExtractor:
    def create_llm() -> OpenAICompatibleLLM:
        current = resolve_model_settings(settings)
        return OpenAICompatibleLLM(
            model=current.model_id,
            base_url=current.base_url,
            api_key=current.api_key,
            timeout_seconds=current.timeout_seconds,
        )

    # 原因：普通文档需要 LLM 抽取自然语言关系，但模型服务与模型 id 会动态变化。
    # 作用：每批抽取实时解析当前模型，同时保留离线规则抽取作为无网络降级路径。
    return CompositeGraphExtractor(
        extractors=(
            RuleBasedGraphExtractor(),
            LLMGraphExtractor(llm_factory=create_llm),
        )
    )


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
            st.session_state.chat_trace = []
            st.rerun()


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _start_user_input(
    user_input: str,
    settings: SmolagentsModelSettings,
    enable_web_search: bool = False,
    enable_local_knowledge: bool = False,
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
        enable_local_knowledge=enable_local_knowledge,
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
            st.session_state.chat_trace = list(result.trace)
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
            st.session_state.chat_trace = list(result.trace)
        st.rerun()

    phase_labels = {
        "connecting": "正在启动 Agent",
        "planning": "正在规划工具调用",
        "searching": "正在通过 Tavily 搜索",
        "retrieving": "正在检索本地知识库",
        "analyzing": "正在分析上传文件",
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


def _render_chat_trace(show_process: bool) -> None:
    """Render only safe orchestration events when explicitly requested."""
    if not show_process or not st.session_state.chat_trace:
        return
    with st.expander("执行过程", expanded=False):
        for index, event in enumerate(st.session_state.chat_trace, start=1):
            parts = [str(event.get("agent") or "orchestrator"), str(event.get("status") or "")]
            if event.get("tool"):
                parts.append(f"Tool: {event['tool']}")
            if event.get("duration_seconds") is not None:
                parts.append(f"{float(event['duration_seconds']):.2f}s")
            message = str(event.get("message") or "").strip()
            suffix = f" — {message}" if message else ""
            st.markdown(f"**{index}.** {' · '.join(parts)}{suffix}")


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
        "chart_png": "image/png",
        "chart_svg": "image/svg+xml",
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
    analysis_options = st.columns(2)
    show_debug = analysis_options[0].toggle("显示分析过程", value=False)
    generate_report = analysis_options[1].toggle("分析完成后生成报告", value=False)

    if st.button("开始本地分析", type="primary", width="stretch"):
        if not uploaded_files:
            st.warning("请先上传文件。")
            return

        with st.spinner("正在保存并分析文件..."):
            try:
                # 原因：文件入口不能继续绕过统一编排层直接调用 AnalysisService。
                # 作用：单文件保持快速路径，组合/报告请求自动升级为 Supervisor 流程。
                outcome = AgentOrchestrator(
                    settings=settings,
                    minirag=st.session_state.minirag,
                ).run_sync(
                    OrchestrationRequest(
                        objective=user_question,
                        uploaded_files=tuple(
                            OrchestrationFile(
                            name=uploaded_file.name,
                            content=uploaded_file.getvalue(),
                        )
                            for uploaded_file in uploaded_files
                        ),
                        generate_report=generate_report,
                        report_title="Qwopus Analysis Report",
                        report_basename="qwopus_analysis_report",
                    )
                )
                if not outcome.success or outcome.analysis_result is None:
                    raise RuntimeError(outcome.final_answer)
                st.session_state.analysis_result = outcome.analysis_result
                st.session_state.analysis_debug_steps = [
                    event.message
                    for event in outcome.trace
                    if event.message
                ]
                st.session_state.analysis_report = outcome.report
                st.success(f"已完成分析：{len(uploaded_files)} 个文件")
            except Exception as exc:
                logger.exception("analysis_failed")
                st.error(f"分析失败：{exc}")

    if st.session_state.analysis_result is not None:
        _render_analysis_result(st.session_state.analysis_result)
        if show_debug:
            _render_debug_steps(st.session_state.analysis_debug_steps)


def _render_knowledge_graph(minirag: MiniRAG) -> None:
    service = KnowledgeGraphService()
    st.subheader("知识图谱")
    entity_types = service.entity_types()
    controls = st.columns([2, 3])
    selected_type = controls[0].selectbox(
        "实体类型",
        ["全部", *entity_types],
    )
    max_nodes = controls[1].slider("最大节点数", min_value=20, max_value=300, value=100)
    snapshot = service.snapshot(
        entity_type=None if selected_type == "全部" else selected_type,
        max_nodes=max_nodes,
    )
    evidence_rows = service.evidence_rows(snapshot)

    metrics = st.columns(3)
    metrics[0].metric("实体", str(len(snapshot.nodes)))
    metrics[1].metric("关系", str(len(snapshot.edges)))
    metrics[2].metric(
        "来源",
        str(len({str(row["source"]) for row in evidence_rows})),
    )
    _render_knowledge_maintenance(minirag)
    if not snapshot.nodes:
        st.info("当前筛选条件下暂无图谱数据。")
        return

    dot = service.to_dot(snapshot)
    # 原因：图结构必须保留关系方向，普通表格无法直观看出多跳路径。
    # 作用：Graphviz 在前端绘制有向节点/边，同时 DOT 可下载用于外部审计。
    st.graphviz_chart(dot, width="stretch")
    st.download_button(
        "下载 DOT",
        data=dot.encode("utf-8"),
        file_name="qwopus_knowledge_graph.dot",
        mime="text/vnd.graphviz",
    )
    if evidence_rows:
        with st.expander("关系证据", expanded=False):
            st.markdown(
                _dataframe_to_safe_html(pd.DataFrame(evidence_rows)),
                unsafe_allow_html=True,
            )



def _render_knowledge_maintenance(minirag: MiniRAG) -> None:
    """Render destructive knowledge operations independently from graph availability."""
    maintenance = KnowledgeMaintenanceService(minirag)
    with st.expander("知识库维护", expanded=False):
        sources = maintenance.list_sources()
        selected_source = st.selectbox(
            "文件来源",
            sources or ["暂无来源"],
            disabled=not sources,
        )
        confirm_delete = st.checkbox(
            "确认删除所选来源",
            value=False,
            disabled=not sources,
        )
        actions = st.columns(2)
        # 原因：空图可能来自索引损坏，此时也必须保留重建入口。
        # 作用：维护区不依赖可视化节点数量，删除仍要求用户显式确认。
        if actions[0].button(
            "删除来源",
            disabled=not sources or not confirm_delete,
            width="stretch",
        ):
            deleted = maintenance.delete_source(selected_source)
            st.success(f"已删除 {deleted} 条文档记录。")
            st.rerun()
        if actions[1].button("重建索引", width="stretch"):
            with st.spinner("正在从持久化文档重建索引..."):
                maintenance.rebuild_indexes()
            st.success("索引重建完成。")
            st.rerun()


def main() -> None:
    configure_runtime_logging()
    logger.info("streamlit_app_started")
    st.set_page_config(page_title="Qwopus-Agent", page_icon="💬", layout="wide")
    st.title("Qwopus-Agent 本地办公助手")
    st.caption(
        "当前阶段：smolagents 对话 + 文档/Excel 上传分析 + MiniRAG 语义检索 + 报告下载。"
    )

    # 原因：用户会在同一个服务器地址上频繁切换模型。
    # 作用：Streamlit 每次重跑都从 /models 刷新侧边栏和后续请求使用的模型 id。
    settings = resolve_model_settings(SmolagentsModelSettings.from_env())
    _init_session_state(settings)
    _render_sidebar(settings)
    analysis_tab, graph_tab, chat_tab = st.tabs(["文档分析", "知识图谱", "对话测试"])

    with analysis_tab:
        _render_upload_analysis(settings)

    with graph_tab:
        _render_knowledge_graph(st.session_state.minirag)

    with chat_tab:
        chat_options = st.columns(3)
        enable_web_search = chat_options[0].toggle("联网搜索", value=False)
        # 原因：本地文件可能包含敏感信息，聊天不应在用户不知情时自动检索。
        # 作用：用户显式开启后，smolagents 才能选择 MiniRAG 或知识图谱 Tool。
        enable_local_knowledge = chat_options[1].toggle("使用本地知识库", value=False)
        show_chat_process = chat_options[2].toggle("显示执行过程", value=False)
        _render_history()
        _render_chat_notice()
        _render_chat_progress()
        _render_chat_trace(show_chat_process)
        user_input = st.chat_input(
            "输入你的问题...",
            disabled=st.session_state.chat_task is not None,
        )
        if user_input:
            _start_user_input(
                user_input,
                settings,
                enable_web_search=enable_web_search,
                enable_local_knowledge=enable_local_knowledge,
            )


if __name__ == "__main__":
    main()
