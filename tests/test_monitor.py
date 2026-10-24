from __future__ import annotations

import pytest

from helpers_agents import art, decisions, make_ctx
from newsroom.agents import monitor as monitor_module
from newsroom.agents.monitor import MonitorAgent, syndication_key
from newsroom.domain import SourceConfig
from newsroom.registry import load_fixture_articles


def test_fixture_scan_stores_everything_and_dedupes_on_rescan():
    ctx = make_ctx()
    agent = MonitorAgent(ctx)
    fixtures = load_fixture_articles()
    first = agent.scan([], fixtures=fixtures)
    assert len(first) == len(fixtures)
    assert agent.scan([], fixtures=load_fixture_articles()) == []
    assert len(decisions(ctx, "duplicate")) == len(fixtures)


def test_injection_article_is_flagged_but_kept_for_humans():
    ctx = make_ctx()
    fresh = MonitorAgent(ctx).scan([], fixtures=load_fixture_articles())
    flagged = next(a for a in fresh if a.id == "fx-injection-1")
    assert "ignore all previous instructions" in flagged.injection_flags
    assert ctx.store.get(type(flagged), flagged.id) is not None
    assert decisions(ctx, "flagged_injection")
    clean = next(a for a in fresh if a.id == "fx-kaleva-sports-1")
    assert clean.injection_flags == []


def test_metadata_only_source_never_fetches_the_page(monkeypatch):
    ctx = make_ctx()
    source = SourceConfig(
        id="kaleva-x",
        name="Kaleva",
        feed_url="https://kaleva.test/rss",
        access="metadata_only",
    )
    paywalled = art("p1", "Kaleva", "Pääotsikko jutusta", summary="Lyhyt tiivistelmä uutisesta.")

    def fake_fetch(src, **_kwargs):
        return [paywalled.model_copy(update={"content": "FULL TEXT WE MUST NOT KEEP"})]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a paywalled page must never be fetched")

    monkeypatch.setattr(monitor_module, "fetch_feed", fake_fetch)
    monkeypatch.setattr(monitor_module, "extract_content", forbidden)
    fresh = MonitorAgent(ctx).scan([source])
    assert len(fresh) == 1
    assert fresh[0].access == "metadata_only"
    assert fresh[0].content is None
    assert fresh[0].summary == "Lyhyt tiivistelmä uutisesta."
    assert decisions(ctx, "metadata_only")


def test_full_access_source_fetches_text(monkeypatch):
    ctx = make_ctx()
    source = SourceConfig(id="yle", name="Yle News", feed_url="https://yle.test/rss", language="en")
    item = art("y1", "Yle News", "Council decides on winter buses", language="en")
    monkeypatch.setattr(monitor_module, "fetch_feed", lambda src, **kw: [item])
    monkeypatch.setattr(monitor_module, "extract_content", lambda url, **kw: "Body text here.")
    fresh = MonitorAgent(ctx).scan([source])
    assert fresh[0].content == "Body text here."
    assert fresh[0].access == "full"


def test_metadata_only_fixture_drops_body_text():
    ctx = make_ctx()
    leaked = art("k1", "Kaleva", "Otsikko", content="Paywalled body", access="metadata_only")
    fresh = MonitorAgent(ctx).scan([], fixtures=[leaked])
    assert fresh[0].content is None


def test_one_broken_source_does_not_stop_the_cycle(monkeypatch):
    ctx = make_ctx()
    good = SourceConfig(id="g", name="Good", feed_url="https://g.test/rss")
    bad = SourceConfig(id="b", name="Bad", feed_url="https://b.test/rss")
    disabled = SourceConfig(id="d", name="Off", feed_url="https://d.test/rss", enabled=False)

    def fake_fetch(src, **_kwargs):
        if src.id == "b":
            raise RuntimeError("boom")
        return [art("g1", "Good", "A good story from a good source")]

    monkeypatch.setattr(monitor_module, "fetch_feed", fake_fetch)
    monkeypatch.setattr(monitor_module, "extract_content", lambda *a, **k: None)
    fresh = MonitorAgent(ctx).scan([bad, disabled, good])
    assert [a.id for a in fresh] == ["g1"]
    assert decisions(ctx, "source_failed")


def test_syndication_key_groups_reposts_not_unrelated_stories():
    a = syndication_key("Seasonal workers struggle to find housing in Levi")
    b = syndication_key("Seasonal Workers Struggle to Find Housing in Levi!")
    c = syndication_key("Bus timetables change in Oulu from November")
    assert a == b and a != c
    assert syndication_key("Short title") is None


@pytest.mark.parametrize("title", ["Kiekko-Kaleva voitti kotiottelun jatkoajalla"])
def test_content_hash_is_set(title):
    ctx = make_ctx()
    fresh = MonitorAgent(ctx).scan([], fixtures=[art("h1", "Kaleva", title, content="x " * 20)])
    assert fresh[0].content_hash and fresh[0].syndication_key
