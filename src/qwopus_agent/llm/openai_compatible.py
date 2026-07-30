"""Generic OpenAI-compatible LLM adapter.

Many local and remote runtimes expose `/v1/chat/completions`. This adapter lets Qwopus-Agent use
those models through one implementation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qwopus_agent.llm.base import BaseLLM, ChatMessage, LLMResponse


class OpenAICompatibleLLMError(RuntimeError):
    """Raised when an OpenAI-compatible server fails or returns invalid data."""


_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
logger = logging.getLogger(__name__)


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

    # Role: Retry only transient transport failures; invalid requests and auth errors stop.
    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3.")

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
        data = ""
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in _RETRYABLE_HTTP_STATUS or attempt >= self.max_retries:
                    raise OpenAICompatibleLLMError(
                        f"OpenAI-compatible server returned HTTP {exc.code}: {detail}"
                    ) from exc
                _wait_before_retry(url, attempt, f"HTTP {exc.code}")
            except (TimeoutError, URLError) as exc:
                if attempt >= self.max_retries:
                    reason = getattr(exc, "reason", str(exc))
                    raise OpenAICompatibleLLMError(
                        f"Could not reach LLM server at {url}: {reason}"
                    ) from exc
                _wait_before_retry(url, attempt, type(exc).__name__)

        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleLLMError(
                "OpenAI-compatible server returned invalid JSON."
            ) from exc
        if not isinstance(decoded, dict):
            raise OpenAICompatibleLLMError(
                "OpenAI-compatible server returned a non-object JSON response."
            )
        return decoded


def _wait_before_retry(url: str, attempt: int, reason: str) -> None:
    """Apply one short exponential delay and expose the attempt in runtime logs."""
    delay_seconds = 0.25 * (2**attempt)
    # 原因：立即重复请求会放大繁忙模型服务的负载，并让多个并发用户同步重试。
    # 作用：只对瞬态失败做短指数退避；总次数仍由 max_retries 硬限制。
    logger.warning(
        "llm_request_retry url=%s attempt=%s delay_seconds=%.2f reason=%s",
        url,
        attempt + 2,
        delay_seconds,
        reason,
    )
    time.sleep(delay_seconds)
