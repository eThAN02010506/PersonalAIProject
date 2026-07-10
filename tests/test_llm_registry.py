import unittest

from qwopus_agent.llm import (
    BaseLLM,
    LLMConfig,
    LLMRegistry,
    LocalMLXLLM,
    OpenAICompatibleLLM,
    create_default_llm_registry,
)


class LLMRegistryTests(unittest.TestCase):
    def test_default_registry_creates_local_mlx_from_config(self) -> None:
        registry = create_default_llm_registry()

        llm = registry.create(
            LLMConfig(
                provider="local_mlx",
                model="gemma-4-12B-it-qat-OptiQ-4bit",
            )
        )

        self.assertIsInstance(llm, LocalMLXLLM)
        self.assertEqual(llm.model, "gemma-4-12B-it-qat-OptiQ-4bit")

    def test_default_registry_creates_openai_compatible_from_config(self) -> None:
        registry = create_default_llm_registry()

        llm = registry.create(
            LLMConfig(
                provider="openai_compatible",
                model="any-model-name",
                base_url="http://127.0.0.1:9999/v1",
            )
        )

        self.assertIsInstance(llm, OpenAICompatibleLLM)
        self.assertEqual(llm.model, "any-model-name")
        self.assertEqual(llm.base_url, "http://127.0.0.1:9999/v1")

    def test_registry_accepts_custom_provider_factory(self) -> None:
        registry = LLMRegistry()

        registry.register(
            "custom_backend",
            lambda config: OpenAICompatibleLLM(
                model=config.model,
                base_url=config.base_url or "http://localhost:8000/v1",
            ),
        )
        llm = registry.create(LLMConfig(provider="custom_backend", model="plugged-model"))

        self.assertIsInstance(llm, BaseLLM)
        self.assertEqual(llm.model, "plugged-model")
