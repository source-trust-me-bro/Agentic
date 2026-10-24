"""A model wrapper that enforces the cycle limits on every call."""

from __future__ import annotations

from typing import Any

from newsroom.limits import CycleLimits
from newsroom.llm.base import LLM, LLMError, T


class BudgetedLLM:
    """Counts model calls and refuses them once the cycle is out of time, out of budget or
    stopped. Everything else (``calls``, ``check``...) passes through to the real model."""

    def __init__(self, inner: LLM, limits: CycleLimits) -> None:
        self._inner = inner
        self._limits = limits
        self.name = inner.name

    def generate(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        payload: dict[str, Any] | None = None,
    ) -> T:
        reason = self._limits.stop_reason()
        if reason:
            raise LLMError(f"Cycle limit: {reason}.")
        if self._limits.active:
            self._limits.calls += 1
        return self._inner.generate(
            task=task, system=system, prompt=prompt, schema=schema, payload=payload
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
