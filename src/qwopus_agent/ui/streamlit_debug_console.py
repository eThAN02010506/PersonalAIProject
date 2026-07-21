"""Read-only Streamlit debug console for the Qwopus-Agent backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    check_model_connection,
    resolve_model_settings,
)
from qwopus_agent.utils.debug_store import load_debug_records
from qwopus_agent.utils.logging_config import configure_runtime_logging, get_logger

logger = get_logger("ui.streamlit_debug_console")
RUNTIME_LOG_PATH = Path("logs/qwopus_agent.log")


def _model_name(model_id: str) -> str:
    """Return a compact model name for either Windows or POSIX server paths."""
    return model_id.replace("\\", "/").rstrip("/").rsplit("/", maxsplit=1)[-1]


def _render_trace(trace: list[dict[str, Any]]) -> None:
    """Render the safe orchestration timeline stored beside each raw run."""
    if not trace:
        st.caption("No orchestration events were recorded.")
        return
    for index, event in enumerate(trace, start=1):
        parts = [str(event.get("agent") or "orchestrator"), str(event.get("status") or "")]
        if event.get("phase"):
            parts.insert(0, str(event["phase"]))
        if event.get("tool"):
            parts.append(f"Tool: {event['tool']}")
        if event.get("duration_seconds") is not None:
            parts.append(f"{float(event['duration_seconds']):.2f}s")
        message = str(event.get("message") or "").strip()
        suffix = f" - {message}" if message else ""
        st.markdown(f"**{index}.** {' · '.join(parts)}{suffix}")


def _render_raw_runs(debug_runs: list[dict[str, Any]]) -> None:
    """Render only raw fields actually returned by the Agent runtime."""
    if not debug_runs:
        st.caption("No raw smolagents steps were recorded for this run.")
        return
    for run_index, run in enumerate(debug_runs, start=1):
        label = str(run.get("label") or f"run_{run_index}")
        steps = run.get("steps") if isinstance(run.get("steps"), list) else []
        st.markdown(f"#### Raw run {run_index}: `{label}`")
        st.caption(
            f"state={run.get('state', 'unknown')} · max_steps={run.get('max_steps', '?')} · "
            f"recorded_steps={len(steps)}"
        )
        with st.expander("Complete Prompt", expanded=False):
            st.code(str(run.get("prompt") or ""), language="text", wrap_lines=True)
        with st.expander("Raw Run Output", expanded=False):
            st.code(str(run.get("output") or ""), language="text", wrap_lines=True)
        for step_index, step in enumerate(steps, start=1):
            payload = step if isinstance(step, dict) else {"value": step}
            step_number = payload.get("step_number", step_index)
            with st.expander(f"Step {step_number} · complete raw record", expanded=False):
                if payload.get("model_output"):
                    st.markdown("**Model output / reasoning draft**")
                    st.code(str(payload["model_output"]), language="text", wrap_lines=True)
                if payload.get("tool_calls"):
                    st.markdown("**Tool calls / arguments**")
                    st.json(payload["tool_calls"], expanded=True)
                if payload.get("observations"):
                    st.markdown("**Tool Observation**")
                    st.code(str(payload["observations"]), language="text", wrap_lines=True)
                if payload.get("error"):
                    st.error(str(payload["error"]))
                st.markdown("**All recorded fields**")
                st.json(payload, expanded=False)


def _render_agent_records(records: list[dict[str, Any]]) -> None:
    """Render persisted runs emitted by the formal FastAPI backend."""
    if not records:
        st.info("No backend Agent runs have been recorded yet.")
        return
    # 原因：原始 Observation 可能很长，页面展开不适合完整复制或归档。
    # 作用：下载内容与磁盘记录一致，方便复现问题且不会进入正式前端响应。
    st.download_button(
        "Download displayed traces",
        data=json.dumps(records, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name="qwopus_backend_debug_traces.json",
        mime="application/json",
    )
    for index, record in enumerate(records):
        source = str(record.get("source") or "agent")
        status = str(record.get("status") or "unknown")
        run_id = str(record.get("run_id") or record.get("id") or "unknown")
        with st.expander(
            f"{source} · {status} · {run_id}",
            expanded=index == 0,
        ):
            st.caption(str(record.get("timestamp") or ""))
            result = str(record.get("result") or "")
            if result:
                st.markdown("### Final result")
                st.code(result, language="text", wrap_lines=True)
            st.markdown("### Orchestration trace")
            trace = record.get("trace")
            _render_trace(trace if isinstance(trace, list) else [])
            st.markdown("### Raw Agent data")
            debug_runs = record.get("debug_runs")
            _render_raw_runs(debug_runs if isinstance(debug_runs, list) else [])


def _render_runtime_logs() -> None:
    """Render the rotating backend log without mutating it."""
    if not RUNTIME_LOG_PATH.is_file():
        st.info("Runtime log has not been created yet.")
        return
    log_text = RUNTIME_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    st.download_button(
        "Download complete runtime log",
        data=log_text.encode("utf-8"),
        file_name=RUNTIME_LOG_PATH.name,
        mime="text/plain",
    )
    st.caption("Showing the latest 500 lines.")
    st.code("\n".join(log_text.splitlines()[-500:]), language="text", wrap_lines=True)


def _render_sidebar(settings: SmolagentsModelSettings) -> None:
    """Render read-only runtime identity and a connection probe."""
    with st.sidebar:
        st.header("Runtime Target")
        st.text(f"Model: {_model_name(settings.model_id)}")
        st.text(f"Address: {settings.base_url}")
        if st.button(
            "Check model connection",
            icon=":material/network_check:",
            width="stretch",
        ):
            online, message = check_model_connection(settings)
            if online:
                st.success(message)
            else:
                st.error(message)
        st.link_button(
            "Open formal frontend",
            "http://127.0.0.1:8010/",
            icon=":material/open_in_new:",
            width="stretch",
        )


def main() -> None:
    configure_runtime_logging()
    logger.info("streamlit_debug_console_started")
    st.set_page_config(
        page_title="Qwopus-Agent Debug Console",
        page_icon=":material/bug_report:",
        layout="wide",
    )
    settings = resolve_model_settings(SmolagentsModelSettings.from_env())
    records = load_debug_records(limit=50)
    _render_sidebar(settings)

    title_column, refresh_column = st.columns([5, 1])
    title_column.title("Qwopus-Agent Debug Console")
    # 原因：Console 不再拥有任务输入，后端完成新运行后不会触发 Streamlit 自动重跑。
    # 作用：显式刷新只重新读取磁盘诊断数据，不会启动或修改任何 Agent 任务。
    if refresh_column.button(
        "Refresh",
        icon=":material/refresh:",
        width="stretch",
    ):
        st.rerun()

    st.warning(
        "Local-only diagnostics may contain complete prompts, document excerpts, Tool Observations "
        "and model-provided reasoning drafts. Hidden reasoning not returned by the model is not "
        "shown."
    )
    latest = records[0] if records else {}
    metrics = st.columns(4)
    metrics[0].metric("Recorded Runs", str(len(records)))
    metrics[1].metric("Latest Source", str(latest.get("source") or "none"))
    metrics[2].metric("Latest Status", str(latest.get("status") or "idle"))
    metrics[3].metric("Current Model", _model_name(settings.model_id))

    runs_tab, logs_tab = st.tabs(["Backend Agent Runs", "Runtime Logs"])
    with runs_tab:
        _render_agent_records(records)
    with logs_tab:
        _render_runtime_logs()


if __name__ == "__main__":
    main()
