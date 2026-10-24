"""Google Gemini provider, using the Interactions API.

Gemini is hosted by Google: it needs network access and an API key and may incur cost.
Structured output is requested with ``response_format`` (a JSON schema built from the
task's Pydantic model) and then re-validated with Pydantic, so a malformed answer is an
error, never silent data. Community text is minimised and scraped text is wrapped as
untrusted by the agents *before* it reaches this layer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from newsroom.llm.base import LLMError, T

logger = logging.getLogger(__name__)


class GeminiLLM:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        retries: int = 2,
        thinking_level: str = "low",
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        from google import genai  # imported lazily so the offline mode needs no key

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout_seconds
        self._retries = retries
        self._key = api_key
        self._thinking_level = thinking_level

    def _scrub(self, message: str) -> str:
        """Never let the API key reach a log line or an exception message."""

        return message.replace(self._key, "***")

    @staticmethod
    def _is_permanent(exc: Exception) -> bool:
        text = str(exc)
        return any(
            marker in text
            for marker in (
                "no longer available",
                "not_found",
                "API key not valid",
                "PERMISSION_DENIED",
                "invalid_api_key",
            )
        ) or text.startswith("Error code: 404")

    def check(self) -> tuple[bool, str]:
        """A tiny real call, so a missing model or a bad key is found at start-up and shown
        to editors, instead of surfacing later as stories that mysteriously fail."""

        try:
            self._client.interactions.create(
                model=self._model,
                input="Reply with the single word: ok",
                generation_config={"thinking_level": self._thinking_level},
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            return False, self._scrub(str(exc))[:240]
        return True, "ok"

    def generate(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        payload: dict[str, Any] | None = None,
    ) -> T:
        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema.model_json_schema(),
        }
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                interaction = self._client.interactions.create(
                    model=self._model,
                    input=prompt,
                    system_instruction=system,
                    response_format=response_format,
                    generation_config={"thinking_level": self._thinking_level},
                    timeout=self._timeout,
                )
                return schema.model_validate_json(str(interaction.output_text or ""))
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Gemini answer for %s failed validation: %s", task, self._scrub(str(exc))
                )
            except Exception as exc:  # network, quota, safety blocks
                last_error = exc
                if self._is_permanent(exc):
                    # A missing model or a bad key will not fix itself on retry.
                    raise LLMError(self._scrub(f"Gemini cannot run task '{task}': {exc}")) from exc
                logger.warning(
                    "Gemini call for %s failed (attempt %s): %s",
                    task,
                    attempt + 1,
                    self._scrub(str(exc)),
                )
            time.sleep(min(2**attempt, 8))
        raise LLMError(self._scrub(f"Gemini could not complete task '{task}': {last_error}"))
