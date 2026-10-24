"""A model wrapper that enforces the cycle limits on every call."""

from __future__ import annotations

import threading
import time
from typing import Any

from newsroom.limits import CycleLimits
from newsroom.llm.base import LLM, LLMError, T


class BudgetedLLM:
    """Counts model calls and refuses them once the cycle is out of time, out of budget or
    stopped. Everything else (``calls``, ``check``...) passes through to the real model."""

    def __init__(self, inner: LLM, limits: CycleLimits, max_concurrent: int = 8) -> None:
        self._inner = inner
        self._limits = limits
        # Stories, languages and claims run side by side; this is the most model calls at once.
        self._slots = threading.BoundedSemaphore(max(1, max_concurrent))
        self._max_concurrent = max(1, max_concurrent)
        self._stats_lock = threading.Lock()
        self.name = inner.name
        self.reset_stats()

    def reset_stats(self) -> None:
        """Start counting afresh, at the beginning of a cycle."""

        with self._stats_lock:
            self._in_flight = 0
            self._peak = 0
            self._finished = 0
            self._seconds = 0.0
            self._by_task: dict[str, int] = {}

    def stats(self) -> dict[str, Any]:
        """How hard the model is being used: calls against the budget, calls running now, and
        how long a call takes on average. Shown live on the Newsroom desk."""

        with self._stats_lock:
            top = sorted(self._by_task.items(), key=lambda item: -item[1])[:6]
            return {
                "calls": self._limits.calls,
                "budget": self._limits.max_model_calls,
                "in_flight": self._in_flight,
                "peak_in_flight": self._peak,
                "max_concurrent": self._max_concurrent,
                "avg_seconds": round(self._seconds / self._finished, 2) if self._finished else None,
                "by_task": dict(top),
            }

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
        self._limits.count_call()
        with self._slots:
            with self._stats_lock:
                self._in_flight += 1
                self._peak = max(self._peak, self._in_flight)
                self._by_task[task] = self._by_task.get(task, 0) + 1
            started = time.monotonic()
            try:
                return self._inner.generate(
                    task=task, system=system, prompt=prompt, schema=schema, payload=payload
                )
            finally:
                with self._stats_lock:
                    self._in_flight -= 1
                    self._finished += 1
                    self._seconds += time.monotonic() - started

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
