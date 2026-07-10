import unittest

from qwopus_agent.llm import BaseLLM, ChatMessage, LLMResponse


class FakeLLM(BaseLLM):
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=messages[-1].content.upper(), model="fake")


class BaseLLMContractTests(unittest.TestCase):
    def test_base_llm_contract_can_be_implemented(self) -> None:
        llm = FakeLLM()

        response = llm.generate([ChatMessage(role="user", content="hello")])

        self.assertEqual(response.content, "HELLO")
        self.assertEqual(response.model, "fake")
