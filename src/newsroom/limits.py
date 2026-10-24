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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def count_call(self) -> None:
        """Several stories call the model at once, so the counter is guarded."""

        with self._lock:
            if self.active:
                self.calls += 1

    def start(self) -> None:
        self.calls = 0
        self.cancel.clear()
        self.deadline = time.monotonic() + self.max_seconds
        self.triggered = None
        self.active = True

    def finish(self) -> None:
        self.active = False

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
