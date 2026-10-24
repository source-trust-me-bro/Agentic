"""What a newsroom cycle is doing right now, so the editor can see it.

A cycle moves through stages (collect news, group it into events, choose stories, write them,
link topics), and each story moves through its own steps. The agents report here as they go; the
Newsroom desk reads a snapshot: the percentage, the time left, each stage and story, a live feed
of what just happened, and how much of the model budget has been used. Everything is guarded by a
lock because stories, languages and feeds are handled side by side.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

# (key, label, share of the whole cycle). Writing stories is the long part.
STAGES: list[tuple[str, str, float]] = [
    ("collect", "Collecting news", 0.15),
    ("group", "Grouping the same news into events", 0.10),
    ("select", "Choosing stories for each community", 0.05),
    ("write", "Writing, translating and checking stories", 0.60),
    ("link", "Linking related topics", 0.10),
]

# The steps of one story, in order. Reaching a step means the ones before it are done.
STORY_STEPS: list[tuple[str, str]] = [
    ("investigate", "Checking the facts"),
    ("explain", "Writing the story"),
    ("translate", "Translating"),
    ("standards", "Standards check"),
    ("review", "Ready for a person to review"),
]
_STEP_INDEX = {key: i for i, (key, _) in enumerate(STORY_STEPS)}

ACTIVITY_KEPT = 80  # events remembered
ACTIVITY_SHOWN = 25  # events sent to the desk, newest first
MIN_PERCENT_FOR_ETA = 6  # too early to say how long it will take before this


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CycleProgress:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Set by the newsroom: returns model usage (calls, budget, in flight, average time).
        self.usage_source: Callable[[], dict[str, Any]] | None = None
        self._reset()

    def _reset(self) -> None:
        self.running = False
        self.outcome = ""  # "" | done | stopped | failed
        self.note = ""
        self._t0: float | None = None
        self._t1: float | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self._stages: dict[str, dict[str, Any]] = {
            key: {
                "label": label,
                "status": "pending",
                "done": 0,
                "total": 0,
                "detail": "",
                "_t0": None,
                "_t1": None,
            }
            for key, label, _ in STAGES
        }
        self._stories: dict[str, dict[str, Any]] = {}
        self._activity: deque[dict[str, Any]] = deque(maxlen=ACTIVITY_KEPT)

    def _elapsed(self) -> float:
        if self._t0 is None:
            return 0.0
        return (self._t1 if self._t1 is not None else time.monotonic()) - self._t0

    # ------------------------------------------------------------------ the cycle

    def start(self) -> None:
        with self._lock:
            self._reset()
            self.running = True
            self._t0 = time.monotonic()
            self.started_at = _now()
            self.log("newsroom", "The cycle started")

    def finish(self, outcome: str = "done", note: str = "") -> None:
        with self._lock:
            self.running = False
            self.outcome = outcome
            self.note = note
            self._t1 = time.monotonic()
            self.finished_at = _now()
            for stage in self._stages.values():
                if stage["status"] == "running":
                    stage["status"] = "done" if outcome == "done" else "stopped"
                    stage["_t1"] = self._t1
            text = {
                "done": "The cycle finished",
                "stopped": f"The cycle was stopped{': ' + note if note else ''}",
                "failed": f"The cycle failed{': ' + note if note else ''}",
            }.get(outcome, "The cycle ended")
            self.log("newsroom", text, "error" if outcome == "failed" else "info")

    # ----------------------------------------------------------------------- feed

    def log(self, agent: str, text: str, level: str = "info") -> None:
        """One line in the live feed: who did what, and how far into the cycle it happened."""

        with self._lock:
            self._activity.append(
                {"t": round(self._elapsed(), 1), "agent": agent, "text": text, "level": level}
            )

    # ---------------------------------------------------------------------- stages

    def begin(self, key: str, total: int = 0, detail: str = "") -> None:
        with self._lock:
            stage = self._stages[key]
            stage.update(status="running", done=0, total=total, detail=detail, _t0=time.monotonic())
            self.log("newsroom", f"{stage['label']} began" + (f": {detail}" if detail else ""))

    def advance(self, key: str, by: int = 1, detail: str | None = None) -> None:
        with self._lock:
            stage = self._stages[key]
            if stage["status"] == "pending":
                stage["status"] = "running"
                stage["_t0"] = time.monotonic()
            stage["done"] += by
            if detail is not None:
                stage["detail"] = detail
                self.log(key, detail)

    def set_total(self, key: str, total: int) -> None:
        with self._lock:
            self._stages[key]["total"] = total

    def finish_stage(self, key: str, status: str = "done", detail: str | None = None) -> None:
        with self._lock:
            stage = self._stages[key]
            stage["status"] = status
            stage["_t1"] = time.monotonic()
            if stage["_t0"] is None:
                stage["_t0"] = stage["_t1"]
            if status == "done" and stage["total"]:
                stage["done"] = stage["total"]
            if detail is not None:
                stage["detail"] = detail
            self.log(
                "newsroom", f"{stage['label']}: {detail}" if detail else f"{stage['label']} done"
            )

    # --------------------------------------------------------------------- stories

    def story_started(self, pitch_id: str, title: str, community: str) -> None:
        with self._lock:
            self._stories[pitch_id] = {
                "pitch_id": pitch_id,
                "title": title,
                "community": community,
                "step": STORY_STEPS[0][0],
                "steps_done": 0,
                "status": "running",
                "languages_done": 0,
                "languages_total": 0,
            }
            self.log("editor", f"Started “{title}” for {community}")

    def story_step(self, pitch_id: str, step: str) -> None:
        with self._lock:
            story = self._stories.get(pitch_id)
            if story is None or step not in _STEP_INDEX:
                return
            if story["step"] != step:
                self.log("reporter", f"“{story['title']}”: {dict(STORY_STEPS)[step].lower()}")
            story["step"] = step
            # Never move backwards when a revision sends a story round again.
            story["steps_done"] = max(story["steps_done"], _STEP_INDEX[step])

    def story_languages(
        self,
        pitch_id: str,
        *,
        total: int | None = None,
        add_done: int = 0,
        language: str = "",
    ) -> None:
        """How many of a story's languages are translated, and which one just finished."""

        with self._lock:
            story = self._stories.get(pitch_id)
            if story is None:
                return
            if total is not None:
                story["languages_total"] = total
                story["languages_done"] = 0
            story["languages_done"] += add_done
            if language:
                self.log("translator", f"{language} ready for “{story['title']}”")

    def story_finished(self, pitch_id: str, status: str) -> None:
        with self._lock:
            story = self._stories.get(pitch_id)
            if story is None:
                return
            story["status"] = status
            if status in ("awaiting_review", "published"):
                story["step"] = STORY_STEPS[-1][0]
                story["steps_done"] = len(STORY_STEPS)
            words = {
                "awaiting_review": "is ready for a person to review",
                "published": "was published",
                "blocked": "is blocked and needs a person",
                "failed": "failed",
            }.get(status, f"ended as {status}")
            self.log(
                "editor", f"“{story['title']}” {words}", "error" if status == "failed" else "info"
            )

    # -------------------------------------------------------------------- snapshot

    def _fraction(self, key: str) -> float:
        stage = self._stages[key]
        if stage["status"] in ("done", "skipped"):
            return 1.0
        if key == "write" and self._stories:
            total = stage["total"] or len(self._stories)
            steps = sum(s["steps_done"] / len(STORY_STEPS) for s in self._stories.values())
            return min(1.0, steps / max(1, total))
        if stage["total"]:
            return min(1.0, stage["done"] / stage["total"])
        return 0.0

    def _stage_seconds(self, stage: dict[str, Any]) -> float | None:
        if stage["_t0"] is None:
            return None
        end = stage["_t1"] if stage["_t1"] is not None else time.monotonic()
        return round(end - stage["_t0"], 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            weights = {key: share for key, _, share in STAGES}
            percent = round(100 * sum(weights[k] * self._fraction(k) for k in weights))
            if not self.running and self.outcome == "done":
                percent = 100
            elapsed = self._elapsed()
            # A rough time left: how long it took to get this far, scaled to what remains.
            eta = None
            if self.running and percent >= MIN_PERCENT_FOR_ETA:
                eta = min(6 * 3600.0, elapsed * (100 - percent) / percent)
            current = next(
                (k for k, _, _ in STAGES if self._stages[k]["status"] == "running"), ""
            )
            usage: dict[str, Any] = {}
            if self.usage_source is not None:
                try:
                    usage = self.usage_source()
                except Exception:  # noqa: BLE001 - usage is a nicety, never break the page
                    usage = {}
            return {
                "running": self.running,
                "outcome": self.outcome,
                "note": self.note,
                "percent": percent,
                "stage": current,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": None if eta is None else round(eta, 0),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "stages": [
                    {
                        "key": key,
                        "label": label,
                        **{k: v for k, v in self._stages[key].items() if not k.startswith("_")},
                        "percent": round(100 * self._fraction(key)),
                        "seconds": self._stage_seconds(self._stages[key]),
                    }
                    for key, label, _ in STAGES
                ],
                "story_steps": [{"key": k, "label": label} for k, label in STORY_STEPS],
                "stories": [
                    {
                        **story,
                        "step_label": dict(STORY_STEPS)[story["step"]],
                        "steps_total": len(STORY_STEPS),
                    }
                    for story in self._stories.values()
                ],
                "activity": list(self._activity)[-ACTIVITY_SHOWN:][::-1],
                "usage": usage,
            }
