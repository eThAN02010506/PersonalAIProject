import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from qwopus_agent.llm import ChatMessage, OpenAICompatibleLLM
from qwopus_agent.llm.openai_compatible import OpenAICompatibleLLMError


class OpenAICompatibleReliabilityTests(unittest.TestCase):
    def test_transient_http_failure_retries_once_then_returns_response(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"model":"live","choices":[{"message":{"content":"ready"}}]}'
        )
        transient = HTTPError(
            "http://local/v1/chat/completions",
            503,
            "busy",
            {},
            BytesIO(b"busy"),
        )
        llm = OpenAICompatibleLLM(
            model="test",
            base_url="http://local/v1",
            max_retries=1,
        )

        with (
            patch(
                "qwopus_agent.llm.openai_compatible.urlopen",
                side_effect=(transient, response),
            ) as request,
            patch("qwopus_agent.llm.openai_compatible._wait_before_retry") as wait,
        ):
            result = llm.generate([ChatMessage(role="user", content="hello")])

        self.assertEqual(result.content, "ready")
        self.assertEqual(request.call_count, 2)
        wait.assert_called_once()

    def test_non_transient_client_error_is_not_retried(self) -> None:
        invalid = HTTPError(
            "http://local/v1/chat/completions",
            400,
            "invalid",
            {},
            BytesIO(b"invalid request"),
        )
        llm = OpenAICompatibleLLM(
            model="test",
            base_url="http://local/v1",
            max_retries=3,
        )

        with (
            patch(
                "qwopus_agent.llm.openai_compatible.urlopen",
                side_effect=invalid,
            ) as request,
            self.assertRaises(OpenAICompatibleLLMError),
        ):
            llm.generate([ChatMessage(role="user", content="hello")])

        request.assert_called_once()

    def test_invalid_retry_and_timeout_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_retries"):
            OpenAICompatibleLLM(
                model="test",
                base_url="http://local/v1",
                max_retries=4,
            )
        with self.assertRaisesRegex(ValueError, "timeout_seconds"):
            OpenAICompatibleLLM(
                model="test",
                base_url="http://local/v1",
                timeout_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
