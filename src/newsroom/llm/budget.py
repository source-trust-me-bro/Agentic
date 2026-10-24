"""A model wrapper that enforces the cycle limits and the model's input budget.

Two limits matter for a newsroom cycle:

* calls and wall-clock time (the cycle budget), and
* the size of one prompt (the model's context window).

The second is checked before the call and fails with a normal ``LLMError``. Agents already
degrade safely on that error, so an oversized prompt costs one refused call instead of a
provider timeout, three retries and a blown cycle.

Every call also records its input size and latency on ``CycleLimits.usage``; the service
writes that to the audit trail at the end of a cycle.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from newsroom.limits import CycleLimits
from newsroom.llm.base import LLM, LLMError, T

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """A cheap, provider-independent estimate. Four characters per token is close enough
    for mixed Finnish/English/Ukrainian text and errs on the safe side for prompts."""

    return (len(text) + 3) // 4


class BudgetedLLM:
    """Counts model calls, refuses them once the cycle is out of time, out of budget or
    stopped, and refuses prompts that would exceed the context-window budget. Everything
    else (``calls``, ``check``...) passes through to the real model."""

    def __init__(self, inner: LLM, limits: CycleLimits, *, max_input_tokens: int = 30000) -> None:
        self._inner = inner
        self._limits = limits
        self._max_input_tokens = max_input_tokens
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

        input_text = f"{system}\n{prompt}"
        input_chars = len(input_text)
        input_tokens = estimate_tokens(input_text)
        if input_tokens > self._max_input_tokens:
            self._limits.record_model_call(
                task, input_chars=input_chars, input_tokens=input_tokens, seconds=0.0
            )
            raise LLMError(
                f"Input budget: task '{task}' needs about {input_tokens} tokens, "
                f"over the {self._max_input_tokens}-token limit ({input_chars} chars). "
                "The caller must send fewer articles."
            )

        # A call may not outlive the cycle: give the provider the smaller of its own timeout
        # and the time actually left, so a slow or retrying call cannot overrun the budget.
        remaining = self._limits.remaining_seconds()
        timeout = remaining if remaining is not None else None
        if timeout is not None and timeout <= 0:
            raise LLMError("Cycle limit: time limit reached.")

        logger.debug(
            "model call %s: %s chars, ~%s tokens, timeout=%s",
            task,
            input_chars,
            input_tokens,
            f"{timeout:.1f}s" if timeout is not None else "provider default",
        )
        started = time.monotonic()
        try:
            return self._inner.generate(
                task=task,
                system=system,
                prompt=prompt,
                schema=schema,
                payload=payload,
                timeout_seconds=timeout,
            )
        finally:
            self._limits.record_model_call(
                task,
                input_chars=input_chars,
                input_tokens=input_tokens,
                seconds=time.monotonic() - started,
            )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
