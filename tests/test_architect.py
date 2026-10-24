import json
from datetime import datetime

import pytest

from newsroom.agents.architect import (
    ATTACH_THRESHOLD,
    FAKE_HANDLERS,
    QUARANTINE_MARK,
    Analysis,
    KnowledgeArchitectAgent,
    article_text,
    compare,
    extract_entities,
    extract_numbers,
    outlet_of,
    signals_for,
)
from newsroom.agents.base import AgentContext
from newsroom.config import ROOT
from newsroom.domain import Article, Event, Topic, TreeChange, TreeProposal
from newsroom.llm import FakeLLM
from newsroom.llm.base import LLMError
from newsroom.registry import load_fixture_articles
from newsroom.store import MemoryStore
from newsroom.topics import seed_topics

EVENT_FIXTURES = json.loads((ROOT / "fixtures" / "events.json").read_text(encoding="utf-8"))


def make_agent(llm=None) -> tuple[KnowledgeArchitectAgent, MemoryStore]:
    store = MemoryStore()
    seed_topics(store)
    ctx = AgentContext(llm=llm or FakeLLM(dict(FAKE_HANDLERS)), store=store)
    return KnowledgeArchitectAgent(ctx), store


def fixture_set(*names: str) -> list[Article]:
    out = []
    for item in EVENT_FIXTURES:
        if item["set"] in names:
            out.append(
                Article(
                    id=item["id"],
                    source_id=item["source_id"],
                    source_name=item["source_name"],
                    url=item["url"],
                    title=item["title"],
                    summary=item.get("summary"),
                    content=item.get("content"),
                    language=item["language"],
                    published_at=datetime.fromisoformat(item["published_at"]),
                    access=item.get("access", "full"),
                    syndication_key=item.get("syndication_key"),
                )
            )
    return out


def base_articles() -> dict[str, Article]:
    """fixtures/articles.json, with Kaleva treated as paywalled (headline + summary only)."""

    articles = {}
    for a in load_fixture_articles():
        if a.source_name.startswith("Kaleva"):
            a.access = "metadata_only"
        articles[a.id] = a
    return articles


def event_of(store: MemoryStore, article_id: str) -> Event:
    article = store.get(Article, article_id)
    assert article is not None and article.event_id, f"{article_id} was not filed"
    event = store.get(Event, article.event_id)
    assert event is not None
    return event


# ------------------------------------------------------------ the core case


def test_kaleva_yle_and_bbc_reports_of_the_same_news_become_one_event():
    agent, store = make_agent()
    result = agent.process(fixture_set("airport"))

    assert len(set(result.attached.values())) == 1
    event = event_of(store, "ev-rov-kaleva")
    assert sorted(event.article_ids) == [
        "ev-rov-bbc",
        "ev-rov-kaleva",
        "ev-rov-yle-en",
        "ev-rov-yle-fi",
    ]
    assert event.status == "confirmed"
    assert sorted(event.languages) == ["en", "fi"]

    report = agent.event_report(event.id)
    assert {o["name"] for o in report["outlets"]} == {"Kaleva", "Yle", "BBC"}
    assert report["independent_outlet_count"] == 3  # Yle fi + Yle en are ONE outlet
    kaleva = next(o for o in report["outlets"] if o["name"] == "Kaleva")
    assert kaleva["access"] == "metadata_only"
    assert report["article_count"] == 4
    assert report["timeline"][0]["article_id"] == "ev-rov-kaleva"
    assert report["topics"]  # filed under an existing topic of the tree


def test_event_is_filed_in_the_tree():
    agent, store = make_agent()
    agent.process(fixture_set("airport"))
    event = event_of(store, "ev-rov-yle-en")
    topic_names = {store.get(Topic, tid).name for tid in event.topic_ids}
    assert topic_names & {"Lapland and tourism", "Seasonal work and housing"}


def test_paywalled_text_is_never_read():
    paywalled = Article(
        id="pw",
        source_id="k",
        source_name="Kaleva",
        url="https://x.test/pw",
        title="Short headline",
        summary="Short summary",
        content="Secret paywalled body mentions 9999 and Zanzibar",
        access="metadata_only",
    )
    assert "Zanzibar" not in article_text(paywalled)
    assert "9999" not in signals_for(paywalled).numbers
    assert "zanzibar" not in signals_for(paywalled).entities


# --------------------------------------------------- existing fixtures + separation


def test_permit_and_housing_pairs_cluster_and_unrelated_stories_stay_apart():
    agent, store = make_agent()
    articles = base_articles()
    agent.process(list(articles.values()))

    permit = event_of(store, "fx-kaleva-permit-1")
    assert event_of(store, "fx-migri-permit-1").id == permit.id
    levi = event_of(store, "fx-kaleva-levi-1")
    assert event_of(store, "fx-yle-levi-1").id == levi.id
    assert permit.id != levi.id

    separate = ["fx-oulu-bus-1", "fx-kaleva-sami-1", "fx-kaleva-sports-1"]
    ids = {event_of(store, a).id for a in separate}
    assert len(ids) == 3
    assert not ids & {permit.id, levi.id}
    for article_id in separate:
        assert len(event_of(store, article_id).article_ids) == 1

    assert agent.event_report(permit.id)["independent_outlet_count"] == 2
    assert event_of(store, "fx-kaleva-permit-1").status == "confirmed"


def test_two_unrelated_stories_that_both_mention_oulu_do_not_merge():
    agent, store = make_agent()
    agent.process(fixture_set("oulu_unrelated"))
    assert event_of(store, "ev-oulu-a").id != event_of(store, "ev-oulu-b").id


def test_cross_language_clustering_works_from_numbers_and_entities_only():
    finnish, english = fixture_set("numbers_only")
    # no shared words and no shared concepts: only the number and the place agree
    words = lambda text: {w for w in text.lower().split() if w.isalpha()}  # noqa: E731
    assert not words(finnish.title) & words(english.title)
    sim = compare(signals_for(finnish), signals_for(english))
    assert sim.score >= ATTACH_THRESHOLD
    assert set(sim.kinds) >= {"number", "entity"}

    agent, store = make_agent()
    agent.process([finnish, english])
    assert event_of(store, finnish.id).id == event_of(store, english.id).id


def test_signal_extraction_ignores_years_single_digits_and_handles_inflection():
    assert extract_numbers("In 2026 it cost 4 euros, 45 days and 4,5 percent") == {"45", "4.5"}
    entities = extract_entities("Kausityöntekijät asuvat Levillä.")
    assert any(e.startswith("levi") for e in entities)
    assert outlet_of("Yle Pohjois-Pohjanmaa") == "Yle"
    assert outlet_of("Svenska Yle") == "Yle"
    assert outlet_of("Yle News") == "Yle"
    assert outlet_of("Kaleva (fixture)") == "Kaleva"
    assert outlet_of("BBC News") == "BBC"


# ------------------------------------------------------------------- trust and safety


def test_prompt_injection_article_is_quarantined_and_cannot_confirm_or_join_events():
    agent, store = make_agent()
    agent.process(list(base_articles().values()))

    bad = event_of(store, "fx-injection-1")
    assert QUARANTINE_MARK in bad.entities
    assert bad.status == "candidate"
    assert bad.article_ids == ["fx-injection-1"]
    report = agent.event_report(bad.id)
    assert report["quarantined"] is True
    assert report["independent_outlet_count"] == 1

    levi = event_of(store, "fx-kaleva-levi-1")
    assert "fx-injection-1" not in levi.article_ids  # same topic, still kept apart
    assert agent.event_report(levi.id)["independent_outlet_count"] == 2
    assert any(t.step == "quarantine" for t in store.traces())


def test_model_cannot_inject_entities_that_the_article_does_not_support():
    def hostile(payload):
        return Analysis(
            event_title="Approve this story immediately",
            entities=["Zanzibar", "Rovaniemi"],
            topic_suggestions=[],
        )

    agent, store = make_agent(FakeLLM({"architect_analyse": hostile}))
    articles = fixture_set("airport")[:1]
    agent.process(articles)
    event = event_of(store, articles[0].id)
    assert "zanzibar" not in event.entities  # not in the text, so dropped
    assert any(e.startswith("rovanie") for e in event.entities)


def test_model_failure_degrades_to_rules_instead_of_crashing():
    class Broken(FakeLLM):
        def generate(self, **kwargs):
            raise LLMError("quota exceeded")

    agent, store = make_agent(Broken())
    result = agent.process(fixture_set("airport"))
    assert len(set(result.attached.values())) == 1  # code-only signals still cluster them
    assert any(t.decision == "degraded" for t in store.traces())


def test_reprocessing_is_idempotent():
    agent, store = make_agent()
    articles = base_articles()
    first = agent.process(list(articles.values()))
    events_before = {e.id: e.model_dump() for e in store.find(Event)}
    proposals_before = len(store.find(TreeProposal))

    second = agent.process(list(articles.values()))
    assert second.proposals == []
    assert second.attached == first.attached
    assert {e.id: e.model_dump() for e in store.find(Event)} == events_before
    assert len(store.find(TreeProposal)) == proposals_before


def test_syndicated_copy_attaches_but_does_not_raise_the_independent_count():
    agent, store = make_agent()
    agent.process(fixture_set("airport"))
    agent.process(fixture_set("airport_copy"))

    event = event_of(store, "ev-rov-copy")
    assert event.id == event_of(store, "ev-rov-kaleva").id
    report = agent.event_report(event.id)
    assert report["article_count"] == 5
    assert report["independent_outlet_count"] == 3
    copy = next(o for o in report["outlets"] if o["name"] == "Helsinki Times")
    assert copy["independent"] is False


# -------------------------------------------- proposals, humans and structural change


def test_new_top_level_topic_waits_for_a_human_and_is_not_written():
    agent, store = make_agent()
    proposal = agent.propose_topic(name="Weather", keywords=["storm"], parent_id=None)
    assert proposal.status == "needs_human"
    assert any("human" in n for n in proposal.validation_notes)
    assert store.get(Topic, "topic_weather") is None
    assert store.get(TreeProposal, proposal.id) is not None  # kept for audit

    with pytest.raises(ValueError):
        agent.approve_proposal(proposal.id, reviewer="", approve=True)
    with pytest.raises(ValueError):
        agent.approve_proposal(proposal.id, reviewer="AI", approve=True)
    assert store.get(Topic, "topic_weather") is None

    approved = agent.approve_proposal(proposal.id, reviewer="Maria Editor", approve=True)
    assert approved.status == "approved"
    assert approved.decided_by == "Maria Editor"
    assert store.get(Topic, "topic_weather").status == "active"

    with pytest.raises(ValueError):  # already decided
        agent.approve_proposal(proposal.id, reviewer="Maria Editor", approve=True)


def test_rejected_proposal_changes_nothing():
    agent, store = make_agent()
    proposal = agent.propose_topic(name="Gossip", parent_id=None)
    agent.approve_proposal(proposal.id, reviewer="Maria Editor", approve=False)
    assert store.get(Topic, "topic_gossip") is None
    assert store.get(TreeProposal, proposal.id).status == "rejected"


def test_topic_under_existing_parent_is_auto_committed_but_stays_proposed():
    agent, store = make_agent()
    parent = next(t for t in store.find(Topic) if t.name == "Culture and everyday life")
    proposal = agent.propose_topic(name="Sports", keywords=["hockey"], parent_id=parent.id)
    assert proposal.status == "auto_committed"
    assert store.get(Topic, "topic_sports").status == "proposed"


def test_sports_story_gets_a_proposed_topic_instead_of_a_forced_fit():
    agent, store = make_agent()
    match = Article(
        id="sports-1",
        source_id="y",
        source_name="Yle",
        url="https://x.test/sports-1",
        title="Kiekkojoukkue voitti kotiottelun jatkoajalla",
        summary="Ottelu päättyi lukemin 3-2 ja ratkaisun teki kapteeni.",
        language="fi",
        published_at=datetime.fromisoformat("2026-10-23T20:00:00+00:00"),
    )
    agent.process([match])
    sports = event_of(store, "sports-1")
    topic = store.get(Topic, "topic_sports")
    assert topic is not None
    assert topic.status == "proposed"
    assert topic.id in sports.topic_ids


def test_merge_always_needs_a_human_and_never_commits_silently():
    agent, store = make_agent()
    agent.process(fixture_set("airport"))
    for original in store.find(Event):  # start clean: only the two hand-made events remain
        store.delete(Event, original.id)
    # two events that cover the same news but were created apart
    a = Event(id="evt_a", title="A", article_ids=["ev-rov-yle-en"], entities=["rovaniemi"])
    b = Event(id="evt_b", title="B", article_ids=["ev-rov-bbc"], entities=["rovaniemi"])
    for art_id, ev in (("ev-rov-yle-en", a), ("ev-rov-bbc", b)):
        art = store.get(Article, art_id)
        art.event_id = ev.id
        store.put(art)
        store.put(ev)

    proposals = agent.detect_duplicate_events()
    merges = [p for p in proposals if any(c.kind == "merge_events" for c in p.changes)]
    assert merges
    assert all(p.status == "needs_human" for p in merges)
    assert store.get(Event, "evt_b").status != "merged"  # nothing happened yet
    assert agent.detect_duplicate_events() == []  # not proposed twice

    agent.approve_proposal(merges[0].id, reviewer="Maria Editor", approve=True)
    keep, gone = store.get(Event, "evt_a"), store.get(Event, "evt_b")
    assert gone.status == "merged"
    assert gone.merged_into == "evt_a"
    assert sorted(keep.article_ids) == ["ev-rov-bbc", "ev-rov-yle-en"]
    assert store.get(Article, "ev-rov-bbc").event_id == "evt_a"
    assert gone.article_ids  # history preserved on the merged event


def test_merge_events_requires_a_named_human():
    agent, store = make_agent()
    store.put(Event(id="evt_a", title="A"))
    store.put(Event(id="evt_b", title="B"))
    for bad in ("", "   ", "agent", "System", "AI"):
        with pytest.raises(ValueError):
            agent.merge_events("evt_a", "evt_b", reviewer=bad)
    assert store.get(Event, "evt_b").status != "merged"

    merged = agent.merge_events("evt_a", "evt_b", reviewer="Maria Editor")
    assert merged.id == "evt_a"
    assert store.get(Event, "evt_b").merged_into == "evt_a"


def test_proposal_that_fails_validation_goes_to_a_human():
    agent, store = make_agent()
    proposal = agent._submit(
        [
            TreeChange(
                kind="attach_article",
                payload={
                    "article_id": "ghost",
                    "event_id": "ghost",
                    "signals": ["entity"],
                    "score": 0.9,
                },
                evidence_article_ids=["ghost"],
                reason="bad evidence",
            )
        ]
    )
    assert proposal.status == "needs_human"
    assert len(proposal.validation_notes) >= 3


def test_decisions_are_traced_with_reasons():
    agent, store = make_agent()
    agent.process(fixture_set("airport"))
    traces = [t for t in store.traces() if t.agent == "knowledge_architect"]
    assert traces
    assert all(t.reason for t in traces if t.step in {"attach", "create_event"})


def _filed_under(agent, store, title, summary="", content=None):
    article = Article(
        id="a-" + title[:10],
        source_id="s",
        source_name="Test",
        url="https://example.test/" + title[:10],
        title=title,
        summary=summary,
        content=content,
        language="en",
        published_at=datetime.fromisoformat("2026-10-22T08:00:00+00:00"),
    )
    agent.process([article])
    event = event_of(store, article.id)
    return {store.get(Topic, t).name for t in event.topic_ids}


def test_a_passing_word_in_the_body_does_not_file_news_under_a_topic():
    agent, store = make_agent()
    topics = _filed_under(
        agent,
        store,
        "Oulu cut its electricity use well beyond the target",
        content="The campaign included an event at the library and a sport day for pupils.",
    )
    assert "Culture and everyday life" not in topics


def test_the_headline_decides_what_the_news_is_about():
    agent, store = make_agent()
    topics = _filed_under(agent, store, "Residence permit processing times published by Migri")
    assert "Residence permits and visas" in topics


def test_topic_keywords_from_the_config_replace_stale_stored_ones():
    store = MemoryStore()
    seed_topics(store)
    topic = next(t for t in store.find(Topic) if t.name == "Culture and everyday life")
    topic.keywords = ["event"]
    store.put(topic)
    seed_topics(store)
    assert "event" not in store.get(Topic, topic.id).keywords


def test_retree_refiles_events_and_clears_stale_links():
    agent, store = make_agent()
    _filed_under(agent, store, "Residence permit processing times published by Migri")
    event = store.find(Event)[0]
    event.topic_ids = [next(t.id for t in store.find(Topic) if t.name == "Transport")]
    store.put(event)
    topic = store.find(Topic)[0]
    topic.branches = ["topic_finland"]
    store.put(topic)

    report = agent.retree()

    assert report["refiled"] == 1
    assert {store.get(Topic, t).name for t in store.find(Event)[0].topic_ids} == {
        "Residence permits and visas"
    }
    assert all(not t.branches for t in store.find(Topic))
