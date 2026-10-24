"""Deterministic offline provider.

It lets the whole newsroom run, and be tested, without a network or an API key. It
is *not* a language model: each agent module registers small rule-based handlers for
its tasks (``FAKE_HANDLERS``). Handlers receive the structured ``payload`` and return
an instance of the task's schema. Output is clearly labelled as an offline stand-in
wherever it could be mistaken for real translation or analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from newsroom.llm.base import LLMError, T

Handler = Callable[[dict[str, Any]], BaseModel]


class FakeLLM:
    name = "fake"

    def __init__(self, handlers: dict[str, Handler] | None = None) -> None:
        self._handlers: dict[str, Handler] = dict(handlers or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []  # lets tests inspect behaviour

    def register(self, task: str, handler: Handler) -> None:
        self._handlers[task] = handler

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
        handler = self._handlers.get(task)
        if handler is None:
            raise LLMError(f"FakeLLM has no handler for task '{task}'.")
        data = payload or {}
        self.calls.append((task, data))
        result = handler(data)
        if not isinstance(result, schema):
            result = schema.model_validate(result.model_dump())
        return result
