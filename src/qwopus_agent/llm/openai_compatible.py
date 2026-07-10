"""Generic OpenAI-compatible LLM adapter.

Many local and remote runtimes expose `/v1/chat/completions`. This adapter lets Qwopus-Agent use
those models through one implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qwopus_agent.llm.base import BaseLLM, ChatMessage, LLMResponse


class OpenAICompatibleLLMError(RuntimeError):
    """Raised when an OpenAI-compatible server fails or returns invalid data."""


@dataclass(frozen=True)
class OpenAICompatibleLLM(BaseLLM):
    """LLM adapter for any backend that implements OpenAI-compatible chat completions."""

    # Reason: The adapter should accept any model id instead of encoding Gemma, Qwen, or Qwopus.
    model: str

    # Role: Base URL for a local or remote `/v1` compatible endpoint.
    base_url: str

    # Role: Optional bearer token for hosted providers; local MLX can leave this empty.
    api_key: str | None = None

    # Role: Prevents HTTP calls from hanging forever.
    timeout_seconds: float = 120.0

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text through a provider-neutral chat request."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai_dict() for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._post_json("/chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleLLMError(
                "OpenAI-compatible server returned an unexpected response shape."
            ) from exc

        return LLMResponse(
            content=content,
            model=str(response.get("model", self.model)),
            raw=response,
            usage=dict(response.get("usage", {})),
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the configured provider endpoint."""
        url = f"{self.base_url.rstrip('/')}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAICompatibleLLMError(
                f"OpenAI-compatible server returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise OpenAICompatibleLLMError(f"Could not reach LLM server at {url}: {exc.reason}") from exc

        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleLLMError("OpenAI-compatible server returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise OpenAICompatibleLLMError(
                "OpenAI-compatible server returned a non-object JSON response."
            )
        return decoded
