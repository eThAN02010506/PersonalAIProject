import sys
import types
import unittest
from unittest.mock import patch

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    build_chat_messages,
    build_smolagents_code_agent,
    build_smolagents_model,
    check_model_connection,
    format_chat_prompt,
    run_smolagents_document_analysis,
    run_smolagents_document_analysis_with_debug,
    run_smolagents_chat_turn,
)


class FakeOpenAIModel:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = None
        self.calls = []
        self.call_kwargs = []
        FakeOpenAIModel.last_instance = self

    def generate(self, messages, **kwargs):
        self.messages = messages
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        if "请直接给出中文最终答案" in messages[-1]["content"]:
            return types.SimpleNamespace(
                content=(
                    "这是一份较完整的中文分析。\n\n"
                    "整体概览：文件围绕作业内容展开。\n\n"
                    "关键内容：文档包含主要任务、计算步骤和结果说明。\n\n"
                    "结论与建议：可以继续补充更细的结果解释。"
                )
            )
        if "TRIGGER_EMPTY" in messages[-1]["content"]:
            return types.SimpleNamespace(content="")
        if "TRIGGER_REASONING_ONLY" in messages[-1]["content"]:
            message = types.SimpleNamespace(
                content="",
                reasoning_content="We need to produce a structured analysis in Chinese.",
                tool_calls=None,
            )
            choice = types.SimpleNamespace(message=message, finish_reason="length")
            raw = types.SimpleNamespace(choices=[choice])
            return types.SimpleNamespace(content="", raw=raw)
        if "请直接给出中文总结" in messages[-1]["content"]:
            return types.SimpleNamespace(content='final_answer("空返回后的重试总结。")')
        if "上一步只是工具 Observation" in messages[-1]["content"]:
            return types.SimpleNamespace(content='final_answer("这是最终总结。")')
        if "TRIGGER_OBSERVATION" in messages[-1]["content"]:
            return types.SimpleNamespace(content="Observation:\nDocument Analysis: raw preview")
        return types.SimpleNamespace(content=f"reply: {messages[-1]['content']}")


class FakeCodeAgent:
    def __init__(self, tools, model, **kwargs):
        self.tools = tools
        self.model = model
        self.kwargs = kwargs

    def run(self, prompt):
        return f"ok: {prompt}"


class SmolagentsRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_module = sys.modules.get("smolagents")
        fake_module = types.ModuleType("smolagents")
        fake_module.OpenAIModel = FakeOpenAIModel
        fake_module.CodeAgent = FakeCodeAgent
        sys.modules["smolagents"] = fake_module

    def tearDown(self) -> None:
        if self.previous_module is None:
            sys.modules.pop("smolagents", None)
        else:
            sys.modules["smolagents"] = self.previous_module

    def test_build_smolagents_model_uses_local_openai_compatible_settings(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="gemma-4-12B-it-qat-OptiQ-4bit",
            base_url="http://127.0.0.1:8080/v1",
            api_key="local_token",
            timeout_seconds=120,
            temperature=0.2,
            max_tokens=128,
        )

        model = build_smolagents_model(settings)

        self.assertEqual(model.kwargs["model_id"], "gemma-4-12B-it-qat-OptiQ-4bit")
        self.assertEqual(model.kwargs["api_base"], "http://127.0.0.1:8080/v1")
        self.assertEqual(model.kwargs["api_key"], "local_token")
        self.assertNotIn("max_tokens", model.kwargs)

    def test_build_smolagents_code_agent_starts_without_tools(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        agent = build_smolagents_code_agent(settings=settings)

        self.assertEqual(agent.tools, [])
        self.assertEqual(agent.run("hello"), "ok: hello")

    def test_format_chat_prompt_includes_history_and_latest_user_message(self) -> None:
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是本地助手。"},
        ]

        prompt = format_chat_prompt(history=history, user_message="上一句你说了什么？")

        self.assertIn("用户：你好", prompt)
        self.assertIn("助手：你好，我是本地助手。", prompt)
        self.assertIn("用户：上一句你说了什么？", prompt)
        self.assertTrue(prompt.endswith("助手："))

    def test_run_smolagents_chat_turn_uses_formatted_prompt(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        history = [{"role": "user", "content": "你好"}]

        result = run_smolagents_chat_turn(
            user_message="请继续",
            history=history,
            settings=settings,
        )

        self.assertEqual(result, "reply: 请继续")

    def test_build_chat_messages_uses_plain_chat_roles(self) -> None:
        history = [{"role": "user", "content": "你好"}]

        messages = build_chat_messages(history=history, user_message="请继续")

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], {"role": "user", "content": "你好"})
        self.assertEqual(messages[2], {"role": "user", "content": "请继续"})

    def test_run_smolagents_document_analysis_sends_file_context(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        result = run_smolagents_document_analysis(
            document_name="assignment.pdf",
            content="This homework asks for vectorized R functions.",
            user_question="总结",
            settings=settings,
        )

        self.assertIn("reply:", result)
        self.assertIn("assignment.pdf", result)
        self.assertIn("总结", result)

    def test_run_smolagents_document_analysis_continues_after_observation(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        result = run_smolagents_document_analysis(
            document_name="assignment.pdf",
            content="TRIGGER_OBSERVATION",
            user_question="总结",
            settings=settings,
        )

        self.assertEqual(result, "这是最终总结。")

    def test_document_analysis_debug_steps_show_observation_retry(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        result = run_smolagents_document_analysis_with_debug(
            document_name="assignment.pdf",
            content="TRIGGER_OBSERVATION",
            user_question="总结",
            settings=settings,
        )

        self.assertEqual(result.answer, "这是最终总结。")
        self.assertTrue(
            any("触发第二轮最终答案生成" in step for step in result.debug_steps)
        )

    def test_document_analysis_retries_after_empty_model_response(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        result = run_smolagents_document_analysis_with_debug(
            document_name="assignment.pdf",
            content="TRIGGER_EMPTY",
            user_question="总结",
            settings=settings,
        )

        self.assertIn("整体概览", result.answer)
        self.assertIn("关键内容", result.answer)
        self.assertTrue(any("第一次模型未生成完整 content" in step for step in result.debug_steps))

    def test_document_analysis_does_not_show_reasoning_content(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        result = run_smolagents_document_analysis_with_debug(
            document_name="assignment.docx",
            content="TRIGGER_REASONING_ONLY",
            user_question="摘要",
            settings=settings,
        )

        self.assertIn("整体概览", result.answer)
        self.assertNotIn("We need to", result.answer)

    def test_document_analysis_retry_rebuilds_messages_without_extra_user_turn(self) -> None:
        settings = SmolagentsModelSettings(
            model_id="any-model",
            base_url="http://127.0.0.1:8080/v1",
        )

        run_smolagents_document_analysis_with_debug(
            document_name="assignment.docx",
            content="TRIGGER_REASONING_ONLY",
            user_question="摘要",
            settings=settings,
        )
        model = FakeOpenAIModel.last_instance

        self.assertEqual(len(model.calls[-1]), 2)
        self.assertEqual(model.calls[-1][0]["role"], "system")
        self.assertEqual(model.calls[-1][1]["role"], "user")
        self.assertIn("TRIGGER_REASONING_ONLY", model.calls[-1][1]["content"])
        self.assertGreater(model.call_kwargs[-1]["max_tokens"], settings.max_tokens)

    @patch("qwopus_agent.integrations.smolagents_runtime.urllib.request.urlopen")
    def test_check_model_connection_reports_online(self, mock_urlopen) -> None:
        mock_response = mock_urlopen.return_value.__enter__.return_value
        mock_response.status = 200

        online, message = check_model_connection(
            SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            )
        )

        self.assertTrue(online)
        self.assertIn("模型服务在线", message)

    @patch("qwopus_agent.integrations.smolagents_runtime.urllib.request.urlopen")
    def test_check_model_connection_reports_offline(self, mock_urlopen) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        online, message = check_model_connection(
            SmolagentsModelSettings(
                model_id="any-model",
                base_url="http://127.0.0.1:8080/v1",
            )
        )

        self.assertFalse(online)
        self.assertIn("无法连接模型服务", message)
