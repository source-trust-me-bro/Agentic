"""Provider for OpenAI-compatible chat endpoints (for example the Hugging Face router).

It uses plain HTTP, so no extra dependency is needed. The model is asked for one JSON object that
matches the task's schema, and the answer is validated with Pydantic: a malformed answer is an
error, never silent data. Permanent errors (a bad token, a missing model, no credit) fail at once;
temporary ones are retried. The token never appears in a log line or an exception message.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from pydantic import ValidationError

from newsroom.llm.base import LLMError, T

logger = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_PERMANENT = {400, 401, 402, 403, 404}


def extract_json(text: str) -> str:
    """The JSON object inside a model reply, tolerating reasoning blocks and code fences."""

    cleaned = _FENCE.sub("", _THINK.sub("", text or "").strip()).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start : end + 1] if start != -1 and end > start else cleaned


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        name: str = "huggingface",
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 120.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("The model token is not set.")
        self.name = name
        self._key = api_key
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout_seconds
        self._retries = retries
        self._transport = transport

    def _scrub(self, message: str) -> str:
        return message.replace(self._key, "***")

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        timeout_seconds: float | None = None,
    ) -> str:
        timeout = (
            min(self._timeout, timeout_seconds) if timeout_seconds is not None else self._timeout
        )
        with httpx.Client(timeout=timeout, transport=self._transport) as client:
            response = client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
        if response.status_code in _PERMANENT:
            raise LLMError(
                self._scrub(
                    f"Model request refused ({response.status_code}): {response.text[:200]}"
                )
            )
        response.raise_for_status()
        try:
            return str(response.json()["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, ValueError) as exc:
            raise httpx.HTTPError("unexpected response shape") from exc

    def check(self) -> tuple[bool, str]:
        """A tiny real call, so a bad token or a missing model shows up at start-up."""

        try:
            self._complete(
                [{"role": "user", "content": 'Reply with the JSON {"ok": true}.'}], max_tokens=64
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
        timeout_seconds: float | None = None,
    ) -> T:
        contract = (
            "Reply with ONLY one JSON object, no prose and no code fences, that matches this "
            "JSON schema:\n" + json.dumps(schema.model_json_schema())
        )
        messages = [
            {"role": "system", "content": f"{system}\n\n{contract}"},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            if timeout_seconds is not None and timeout_seconds <= 0:
                break
            try:
                reply = self._complete(messages, timeout_seconds=timeout_seconds)
                return schema.model_validate_json(extract_json(reply))
            except LLMError:
                raise  # permanent: a retry would fail the same way
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Answer for %s did not match the schema: %s", task, self._scrub(str(exc))[:200]
                )
                messages = [
                    *messages[:2],
                    {
                        "role": "user",
                        "content": "That was not valid JSON for the schema. Reply again with "
                        "ONLY the corrected JSON object.",
                    },
                ]
            except (httpx.HTTPError, OSError) as exc:  # network, rate limits, server errors
                last_error = exc
                logger.warning(
                    "Model call for %s failed (attempt %s): %s",
                    task,
                    attempt + 1,
                    self._scrub(str(exc))[:200],
                )
            time.sleep(min(2**attempt, 8))
        raise LLMError(self._scrub(f"The model could not complete task '{task}': {last_error}"))
