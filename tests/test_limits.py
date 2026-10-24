"""A cycle is bounded and stoppable. This is a demo-sized newsroom, so it must never run away."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from newsroom.agents.base import AgentContext
from newsroom.agents.monitor import MonitorAgent
from newsroom.api.app import create_app
from newsroom.config import Settings
from newsroom.domain import Article, Publication, SourceConfig, new_id
from newsroom.limits import CycleLimits
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.llm.base import LLMError
from newsroom.llm.budget import BudgetedLLM
from newsroom.service import Newsroom
from newsroom.store.memory import MemoryStore


def build(**overrides) -> Newsroom:
    return Newsroom(
        settings=Settings(llm_provider="fake", mongodb_uri="", **overrides),
        store=MemoryStore(),
        llm=FakeLLM(collect_fake_handlers()),
        persistent_checkpoints=False,
    )


# ------------------------------------------------------------------ the limits themselves


def test_limits_are_only_enforced_while_a_cycle_runs():
    limits = CycleLimits(max_model_calls=1, max_seconds=0)

    assert limits.stop_reason() is None  # idle: nothing is enforced
    limits.start()
    assert limits.stop_reason() == "time limit reached"
    limits.finish()
    assert limits.stop_reason() is None


def test_stop_reasons_are_reported_in_order_and_remembered():
    limits = CycleLimits(max_model_calls=2, max_seconds=1000)
    limits.start()
    assert limits.stop_reason() is None

    limits.calls = 2
    assert limits.stop_reason() == "model-call limit reached"
    limits.cancel.set()
    assert limits.stop_reason() == "stopped by an editor"
    assert limits.triggered == "model-call limit reached"  # the first reason is kept


def test_the_model_refuses_calls_once_the_budget_is_spent():
    limits = CycleLimits(max_model_calls=2, max_seconds=1000)
    fake = FakeLLM({"echo": lambda payload: payload["schema"](answer="x")})
    llm = BudgetedLLM(fake, limits)

    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    def ask():
        return llm.generate(
            task="echo", system="s", prompt="p", schema=Out, payload={"schema": Out}
        )

    limits.start()
    ask()
    ask()
    with pytest.raises(LLMError, match="model-call limit"):
        ask()
    assert len(fake.calls) == 2  # the third call never reached the model
    assert llm.calls is fake.calls  # everything else passes through


# ------------------------------------------------------------------ a whole cycle


def test_articles_per_cycle_are_capped_and_the_rest_wait_in_the_database():
    newsroom = build(max_articles_per_cycle=3)

    first = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert first.articles_processed == 3
    assert first.articles_deferred == 5
    saved = newsroom.store.find(Article)
    assert len(saved) == 8  # nothing was thrown away
    assert sum(1 for a in saved if a.event_id) == 3  # only 3 were clustered

    second = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    assert second.articles_processed == 3  # the next three, picked up from the database
    assert sum(1 for a in newsroom.store.find(Article) if a.event_id) == 6

    third = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    assert third.articles_deferred == 0
    assert sum(1 for a in newsroom.store.find(Article) if a.event_id) == 8


def test_the_model_call_budget_ends_a_cycle_safely():
    newsroom = build(cycle_max_model_calls=4)

    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert report.stopped_reason == "model-call limit reached"
    assert len(newsroom.ctx.llm.calls) <= 4
    assert newsroom.store.find(Publication) == []  # an unfinished story is never published
    assert len(newsroom.store.find(Article)) == 8  # what was collected is kept

    # The next cycle starts with a fresh budget.
    newsroom.settings.cycle_max_model_calls = 400
    newsroom.limits.max_model_calls = 400
    assert newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True).stopped_reason is None


def test_the_time_budget_ends_a_cycle_at_once():
    newsroom = build(cycle_max_seconds=0)

    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert report.stopped_reason == "time limit reached"
    assert report.runs == []  # no story was started after the deadline
    assert newsroom.ctx.llm.calls == []


def test_an_editor_can_stop_a_running_cycle():
    newsroom = build()
    original_scan = newsroom.monitor.scan

    def scan_then_stop(*args, **kwargs):
        articles = original_scan(*args, **kwargs)
        newsroom.limits.cancel.set()  # the editor presses Stop while the cycle is working
        return articles

    newsroom.monitor.scan = scan_then_stop  # type: ignore[method-assign]
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert report.stopped_reason == "stopped by an editor"
    assert report.runs == []
    assert len(newsroom.store.find(Article)) == 8  # already collected, so kept


def test_the_stop_endpoint():
    newsroom = build()
    with TestClient(create_app(newsroom)) as client:
        assert client.post("/cycle/stop").json() == {"status": "idle"}

        client.app.state.cycle_lock.acquire()  # pretend a cycle is running
        try:
            assert client.post("/cycle/stop").json() == {"status": "stopping"}
            assert newsroom.limits.cancel.is_set()
        finally:
            client.app.state.cycle_lock.release()


def test_defaults_are_demo_sized():
    settings = Settings(llm_provider="fake", mongodb_uri="")

    assert settings.max_items_per_source <= 3
    assert settings.max_articles_per_cycle <= 30
    assert settings.max_pitches_per_cycle <= 2
    assert settings.cycle_max_seconds <= 600
    assert settings.cycle_max_model_calls <= 150


# ------------------------------------------------------------------ the monitor


def _source(sid: str, official: bool) -> SourceConfig:
    return SourceConfig(id=sid, name=sid, feed_url=f"https://example.test/{sid}", official=official)


def test_official_sources_are_read_first_and_the_cap_stops_the_rest(monkeypatch):
    seen: list[tuple[str, int]] = []

    def fake_fetch(source, *, limit, settings=None, client=None):
        seen.append((source.id, limit))
        return [
            Article(
                id=new_id("art"),
                source_id=source.id,
                source_name=source.name,
                url=f"https://example.test/{source.id}/{n}",
                title=f"{source.id} story {n}",
                language="fi",
            )
            for n in range(limit)
        ]

    monkeypatch.setattr("newsroom.agents.monitor.fetch_feed", fake_fetch)
    monkeypatch.setattr("newsroom.agents.monitor.extract_content", lambda *a, **k: None)
    ctx = AgentContext(llm=FakeLLM(), store=MemoryStore(), settings=Settings(mongodb_uri=""))
    sources = [
        _source("bbc", False),
        _source("kaleva", False),
        _source("oulu-city", True),
        _source("migri", True),
    ]

    fresh = MonitorAgent(ctx).scan(sources, limit_per_source=2, max_new=4)

    assert [sid for sid, _ in seen] == ["oulu-city", "migri"]  # official first, then stopped
    assert all(limit == 2 for _, limit in seen)  # a few items per feed
    assert {a.source_id for a in fresh} == {"oulu-city", "migri"}
