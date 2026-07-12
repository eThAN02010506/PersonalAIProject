"""Minimal Streamlit chat UI for smolagents local model testing."""
from __future__ import annotations
import sys

print("PYTHON:", sys.executable)
print("PATH:")
print("\n".join(sys.path))


import streamlit as st

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsDependencyError,
    SmolagentsModelSettings,
    check_model_connection,
    run_smolagents_chat_turn,
)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_sidebar(settings: SmolagentsModelSettings) -> None:
    with st.sidebar:
        st.header("模型配置")
        st.text(f"模型：{settings.model_id}")
        st.text(f"地址：{settings.base_url}")

        if st.button("检测模型连接", use_container_width=True):
            online, message = check_model_connection(settings)
            if online:
                st.success(message)
            else:
                st.error(message)

        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_user_input(user_input: str, settings: SmolagentsModelSettings) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input})
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

                st.write("DEBUG:")
                st.write(type(reply))
                st.write(repr(reply))

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except SmolagentsDependencyError as exc:
                st.error(str(exc))
            except ConnectionError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"对话调用失败：{exc}")


def main() -> None:
    st.set_page_config(page_title="Qwopus-Agent Chat", page_icon="💬", layout="wide")
    st.title("Qwopus-Agent 本地对话测试")
    st.caption("第一阶段：验证 smolagents → 本地 MLX 大模型 → Streamlit 多轮对话")

    _init_session_state()
    settings = SmolagentsModelSettings.from_env()
    _render_sidebar(settings)
    _render_history()

    user_input = st.chat_input("输入你的问题...")
    if user_input:
        _handle_user_input(user_input, settings)


if __name__ == "__main__":
    main()
