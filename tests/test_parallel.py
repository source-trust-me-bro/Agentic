"""The agents do their slow, independent work side by side, without changing the result."""

from __future__ import annotations

import threading
import time

from newsroom.agents.base import AgentContext
from newsroom.agents.monitor import MonitorAgent
from newsroom.config import Settings
from newsroom.domain import Article, SourceConfig, new_id
from newsroom.limits import CycleLimits
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.llm.budget import BudgetedLLM
from newsroom.parallel import run_parallel, start_in_background
from newsroom.service import Newsroom
from newsroom.store.memory import MemoryStore


def test_run_parallel_keeps_the_order_and_really_overlaps():
    def slow(n: int) -> int:
        time.sleep(0.2 - n * 0.02)  # the first items finish last
        return n * n

    start = time.perf_counter()
    out = run_parallel(slow, range(8), workers=8)
    took = time.perf_counter() - start

    assert out == [n * n for n in range(8)]  # results in the order the items went in
    assert took < 0.6  # eight 0.1-0.2 s jobs in a fraction of the 1.1 s one after another
    assert run_parallel(lambda n: n, [], workers=4) == []
    assert run_parallel(lambda n: n + 1, [1, 2], workers=1) == [2, 3]


def test_start_in_background_runs_now_and_passes_on_errors():
    ran = threading.Event()
    wait = start_in_background(lambda: (ran.set(), "done")[1])
    assert wait() == "done" and ran.is_set()

    def boom():
        raise ValueError("went wrong")

    failing = start_in_background(boom)
    try:
        failing()
    except ValueError as exc:
        assert "went wrong" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("the error should be re-raised")


def test_the_cap_on_model_calls_in_flight_holds_however_many_threads_ask():
    class Inner:
        name = "fake"

        def __init__(self):
            self.now = 0
            self.most = 0
            self.lock = threading.Lock()

        def generate(self, **kwargs):
            with self.lock:
                self.now += 1
                self.most = max(self.most, self.now)
            time.sleep(0.05)
            with self.lock:
                self.now -= 1
            return "ok"

    inner = Inner()
    llm = BudgetedLLM(inner, CycleLimits.unlimited(), max_concurrent=3)

    out = run_parallel(
        lambda _: llm.generate(task="t", system="", prompt="", schema=object),
        range(24),
        workers=12,
    )

    assert out == ["ok"] * 24
    assert 1 < inner.most <= 3  # overlapping, but never more than the cap


def _source(sid: str, official: bool = False) -> SourceConfig:
    return SourceConfig(id=sid, name=sid, feed_url=f"https://example.test/{sid}", official=official)


def test_feeds_and_pages_are_downloaded_side_by_side_and_stored_in_priority_order(monkeypatch):
    def fake_fetch(source, *, limit, settings=None, client=None):
        time.sleep(0.25)  # a slow website
        return [
            Article(
                id=new_id("art"),
                source_id=source.id,
                source_name=source.name,
                url=f"https://example.test/{source.id}/{n}",
                title=f"{source.id} story {n}",
                language="fi",
            )
            for n in range(2)
        ]

    def fake_page(url, *, settings=None, client=None):
        time.sleep(0.1)
        return f"full text of {url}", None

    monkeypatch.setattr("newsroom.agents.monitor.fetch_feed", fake_fetch)
    monkeypatch.setattr("newsroom.agents.monitor.extract_page", fake_page)
    sources = [_source("bbc"), _source("yle"), _source("kaleva"), _source("city", True)]

    def scan(workers: int):
        ctx = AgentContext(
            llm=FakeLLM(),
            store=MemoryStore(),
            settings=Settings(mongodb_uri="", fetch_workers=workers),
        )
        start = time.perf_counter()
        fresh = MonitorAgent(ctx).scan(sources, limit_per_source=2)
        return time.perf_counter() - start, fresh

    in_turn, expected = scan(1)
    side_by_side, got = scan(8)

    assert [a.url for a in got] == [a.url for a in expected]  # same articles, same order
    assert got[0].source_id == "city"  # the official body is still stored first
    assert all(a.content and a.content.startswith("full text") for a in got)
    assert in_turn > 1.5  # 4 feeds x 0.25 s, then 8 pages x 0.1 s, one after another
    assert side_by_side < in_turn / 2


class SlowFake(FakeLLM):
    """The offline model plus a delay per call, standing in for network latency."""

    delay = 0.03

    def generate(self, **kwargs):
        time.sleep(self.delay)
        return super().generate(**kwargs)


def _cycle(**workers):
    nr = Newsroom(
        settings=Settings(llm_provider="fake", mongodb_uri="", **workers),
        store=MemoryStore(),
        llm=SlowFake(collect_fake_handlers()),
        persistent_checkpoints=False,
    )
    start = time.perf_counter()
    report = nr.run_cycle(None, use_fixtures=True)
    took = time.perf_counter() - start
    return took, nr, report


def test_a_whole_cycle_runs_side_by_side_with_the_same_result_only_faster():
    in_turn_time, in_turn, report_a = _cycle(
        story_workers=1,
        translation_workers=1,
        claim_workers=1,
        fetch_workers=1,
        max_concurrent_model_calls=1,
    )
    parallel_time, parallel, report_b = _cycle(
        story_workers=6,
        translation_workers=6,
        claim_workers=4,
        fetch_workers=8,
        max_concurrent_model_calls=16,
    )

    summary = lambda report: [(r.community_id, r.working_title, r.status) for r in report.runs]  # noqa: E731
    assert summary(report_b) == summary(report_a)  # the same stories, in the same order
    assert sorted(t for t, _ in parallel.ctx.llm.calls) == sorted(t for t, _ in in_turn.ctx.llm.calls)
    assert report_b.topic_links == report_a.topic_links
    # One story per event leaves fewer (and so less parallel) stories in the fixture cycle.
    assert parallel_time < in_turn_time / 1.5, (in_turn_time, parallel_time)
