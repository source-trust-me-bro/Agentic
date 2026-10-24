"""Hard limits on one newsroom cycle, so it can never run away.

A cycle is stopped by whichever comes first: an editor pressing Stop, the time budget, or the
model-call budget. Agents check the limits between steps and the model layer refuses calls once
a limit is hit, so every agent degrades safely (an unfinished story is blocked, never published).
Anything not processed stays saved in the database and is picked up by the next cycle.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class CycleLimits:
    max_model_calls: int = 120
    max_seconds: float = 420.0
    cancel: threading.Event = field(default_factory=threading.Event)
    calls: int = 0
    deadline: float | None = None
    active: bool = False
    triggered: str | None = None  # why the cycle was cut short, if it was
    # Per-task model usage, recorded so a cycle can report prompt size and time. This is
    # what makes a context-window regression visible instead of silent.
    usage: dict[str, dict[str, float]] = field(default_factory=dict)

    def start(self) -> None:
        self.calls = 0
        self.usage = {}
        self.cancel.clear()
        self.deadline = time.monotonic() + self.max_seconds
        self.triggered = None
        self.active = True

    def finish(self) -> None:
        self.active = False

    def remaining_seconds(self) -> float | None:
        """Seconds left in the cycle, or None when no cycle deadline is enforced."""

        if not self.active or self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def record_model_call(
        self, task: str, *, input_chars: int, input_tokens: int, seconds: float
    ) -> None:
        entry = self.usage.setdefault(
            task,
            {"calls": 0.0, "input_chars": 0.0, "input_tokens": 0.0, "seconds": 0.0},
        )
        entry["calls"] += 1
        entry["input_chars"] += input_chars
        entry["input_tokens"] += input_tokens
        entry["seconds"] += seconds

    def stop_reason(self) -> str | None:
        """Why the cycle must stop now, or None. Only enforced while a cycle is running."""

        if not self.active:
            return None
        reason = None
        if self.cancel.is_set():
            reason = "stopped by an editor"
        elif self.deadline is not None and time.monotonic() >= self.deadline:
            reason = "time limit reached"
        elif self.calls >= self.max_model_calls:
            reason = "model-call limit reached"
        if reason and self.triggered is None:
            self.triggered = reason
        return reason

    @classmethod
    def unlimited(cls) -> CycleLimits:
        return cls(max_model_calls=10**9, max_seconds=10**9)
