import sys
import types
import unittest

from qwopus_agent.integrations.smolagents_runtime import (
    SmolagentsModelSettings,
    build_smolagents_code_agent,
    build_smolagents_model,
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
