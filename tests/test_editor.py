from __future__ import annotations

from helpers_agents import KALEVA_PERMIT, MIGRI_PERMIT, art, community, decisions, make_ctx
from newsroom.agents.editor import NewsEditorAgent
from newsroom.agents.monitor import MonitorAgent
from newsroom.domain import RiskTier, UserNeed
from newsroom.registry import load_fixture_articles


def select_from_fixtures(community_id: str, max_pitches: int = 5):
    ctx = make_ctx()
    fresh = MonitorAgent(ctx).scan([], fixtures=load_fixture_articles(paywalled=False))
    pitches = NewsEditorAgent(ctx).select(community(community_id), fresh, max_pitches=max_pitches)
    return ctx, pitches


def test_sports_story_is_rejected_for_newcomers_with_a_reason():
    ctx, pitches = select_from_fixtures("oulu-international")
    covered = {aid for p in pitches for aid in p.article_ids}
    assert "fx-kaleva-sports-1" not in covered
    rejected = decisions(ctx, "rejected")
    assert any("fx-kaleva-sports-1" in t.refs["article_ids"] for t in rejected)
    assert all(t.reason for t in rejected)


def test_injection_article_is_never_pitched():
    ctx, pitches = select_from_fixtures("lapland-seasonal")
    assert all("fx-injection-1" not in p.article_ids for p in pitches)
    assert decisions(ctx, "excluded_injection")


def test_permit_story_appears_once_even_in_two_languages():
    _, pitches = select_from_fixtures("oulu-international")
    permit = [
        p for p in pitches if {"fx-kaleva-permit-1", "fx-migri-permit-1"} & set(p.article_ids)
    ]
    assert len(permit) == 1
    assert permit[0].risk_tier == RiskTier.HIGH
    assert permit[0].origin == "monitor"


def test_pitches_have_needs_score_and_are_saved():
    ctx, pitches = select_from_fixtures("lapland-seasonal")
    assert pitches
    for pitch in pitches:
        assert pitch.needs and all(isinstance(n, UserNeed) for n in pitch.needs)
        assert 0.0 <= pitch.score <= 1.0
        assert ctx.store.get(type(pitch), pitch.id) is not None
    scores = [p.score for p in pitches]
    assert scores == sorted(scores, reverse=True)


def test_sami_story_matches_the_sami_community_only():
    _, sami = select_from_fixtures("sapmi-north")
    assert any("fx-kaleva-sami-1" in p.article_ids for p in sami)
    _, oulu = select_from_fixtures("oulu-international")
    assert all("fx-kaleva-sami-1" not in p.article_ids for p in oulu)


def test_event_id_groups_kaleva_paywalled_and_yle_into_one_pitch():
    ctx = make_ctx()
    kaleva = art(
        "k1",
        "Kaleva",
        "Oleskeluluvan käsittelyajat pitenevät",
        summary=KALEVA_PERMIT,
        access="metadata_only",
        event_id="ev1",
    )
    yle = art(
        "y1",
        "Yle News",
        "Work permit processing takes longer",
        content=MIGRI_PERMIT,
        language="en",
        event_id="ev1",
        source_id="yle-news-en",
    )
    yle_fi = art(
        "y2",
        "Yle Pohjois-Pohjanmaa",
        "Työlupien käsittely hidastuu",
        summary=KALEVA_PERMIT,
        event_id="ev1",
        source_id="yle-pp",
    )
    pitches = NewsEditorAgent(ctx).select(
        community("oulu-international"), [kaleva, yle, yle_fi], max_pitches=3
    )
    assert len(pitches) == 1
    assert set(pitches[0].article_ids) == {"k1", "y1", "y2"}


def test_pitch_backed_by_more_outlets_scores_higher():
    ctx = make_ctx()
    text = "Seasonal workers struggle to find housing in {place} before the winter season."
    two_outlets = [
        art(
            "a1",
            "Yle News",
            "Seasonal workers housing Levi",
            summary=text.format(place="Levi"),
            language="en",
            event_id="evA",
        ),
        art(
            "a2",
            "BBC News",
            "Seasonal workers housing in Levi",
            summary=text.format(place="Levi"),
            language="en",
            event_id="evA",
        ),
    ]
    one_outlet = [
        art(
            "b1",
            "Yle News",
            "Seasonal workers housing Ylläs",
            summary=text.format(place="Ylläs"),
            language="en",
            event_id="evB",
        ),
    ]
    pitches = NewsEditorAgent(ctx).select(
        community("lapland-seasonal"), two_outlets + one_outlet, max_pitches=2
    )
    by_event = {p.article_ids[0][0]: p.score for p in pitches}
    assert set(by_event) == {"a", "b"}
    assert by_event["a"] > by_event["b"]


def test_diversity_cap_keeps_one_source_from_filling_every_slot():
    ctx = make_ctx()
    housing = "Seasonal workers struggle to find housing in {place}."
    articles = [
        art(
            f"s1-{i}",
            "Yle News",
            f"Housing story number {i} for {place}",
            summary=housing.format(place=place),
            language="en",
            source_id="src1",
            published=f"2026-10-22T1{i}:00:00+00:00",
        )
        for i, place in enumerate(["Levi", "Ylläs", "Rovaniemi"])
    ] + [
        art(
            "s2-0",
            "Kaleva",
            "Kuljetus ja bussiaikataulut Levillä muuttuvat",
            summary="Joukkoliikenteen aikataulut Levillä muuttuvat talvikaudella.",
            source_id="src2",
            access="metadata_only",
        ),
    ]
    pitches = NewsEditorAgent(ctx).select(community("lapland-seasonal"), articles, max_pitches=2)
    by_id = {a.id: a for a in articles}
    sources = {by_id[p.article_ids[0]].source_id for p in pitches}
    assert sources == {"src1", "src2"}


def test_model_outage_degrades_to_deterministic_pitches():
    ctx = make_ctx(empty_llm=True)
    fresh = MonitorAgent(ctx).scan([], fixtures=load_fixture_articles(paywalled=False))
    pitches = NewsEditorAgent(ctx).select(community("lapland-seasonal"), fresh, max_pitches=2)
    assert pitches and all(p.working_title and p.needs for p in pitches)
    assert decisions(ctx, "model_unavailable")


def test_paywalled_kaleva_and_other_outlets_become_one_pitch_through_the_architect():
    """The real path: Kaleva gives only a headline and summary, the architect links it to
    the other outlets' reports of the same news, and the editor pitches the event once."""

    from newsroom.agents.architect import KnowledgeArchitectAgent
    from newsroom.agents.base import AgentContext
    from newsroom.domain import Article
    from newsroom.llm import FakeLLM, collect_fake_handlers
    from newsroom.registry import load_communities
    from newsroom.store.memory import MemoryStore

    store = MemoryStore()
    ctx = AgentContext(llm=FakeLLM(collect_fake_handlers()), store=store)
    articles = MonitorAgent(ctx).scan([], fixtures=load_fixture_articles())
    kaleva = next(a for a in articles if a.id == "fx-kaleva-permit-1")
    assert kaleva.access == "metadata_only" and not kaleva.content

    KnowledgeArchitectAgent(ctx).process(articles)
    articles = [store.get(Article, a.id) for a in articles]
    community = next(c for c in load_communities() if c.id == "oulu-international")

    pitches = NewsEditorAgent(ctx).select(community, articles, max_pitches=5)

    permit = [p for p in pitches if "fx-kaleva-permit-1" in p.article_ids]
    assert len(permit) == 1
    assert "fx-migri-permit-1" in permit[0].article_ids  # same event, one pitch
