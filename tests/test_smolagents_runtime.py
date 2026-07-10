import sys
import types
import unittest
from unittest.mock import patch

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    build_smolagents_code_agent,
    build_smolagents_model,
    check_model_connection,
    format_chat_prompt,
    run_smolagents_chat_turn,
)


class FakeInferenceClientModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCodeAgent:
    def __init__(self, tools, model):
        self.tools = tools
        self.model = model

    def run(self, prompt):
        return f"ok: {prompt}"


class SmolagentsRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_module = sys.modules.get("smolagents")
        fake_module = types.ModuleType("smolagents")
        fake_module.InferenceClientModel = FakeInferenceClientModel
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
        self.assertEqual(model.kwargs["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(model.kwargs["api_key"], "local_token")

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

        self.assertIn("用户：你好", result)
        self.assertIn("用户：请继续", result)

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
