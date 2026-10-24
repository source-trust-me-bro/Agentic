"""Clustering must be quick: no model call per article, and few database round trips."""

from __future__ import annotations

from newsroom.config import Settings
from newsroom.domain import Article, Event, Story, TraceEvent, new_id
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.registry import load_fixture_articles
from newsroom.service import Newsroom
from newsroom.store.cache import ReadCache
from newsroom.store.memory import MemoryStore


class CountingStore(MemoryStore):
    """A store that counts how often it is asked, to stand in for a remote database."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0
        self.writes = 0
        self.trace_writes = 0

    def get(self, cls, doc_id):
        self.reads += 1
        return super().get(cls, doc_id)

    def find(self, cls, **equals):
        self.reads += 1
        return super().find(cls, **equals)

    def put(self, doc):
        self.writes += 1
        super().put(doc)

    def trace(self, event):
        self.trace_writes += 1
        super().trace(event)


def article(**kw) -> Article:
    return Article(
        id=new_id("art"),
        source_id="s",
        source_name="S",
        url=f"https://example.test/{new_id('u')}",
        title="T",
        **kw,
    )


# ------------------------------------------------------------------ the cache


def test_reads_are_served_from_memory_and_copies_are_returned():
    inner = CountingStore()
    a = article()
    inner.put(a)
    cache = ReadCache(inner)
    inner.reads = 0

    first = cache.get(Article, a.id)
    second = cache.get(Article, a.id)
    cache.find(Article)
    cache.find(Article)

    assert inner.reads == 2  # one get and one find, however often they are repeated
    first.title = "changed by a caller"
    assert cache.get(Article, a.id).title == "T"  # callers never change the cached copy
    assert second.title == "T"


def test_writes_pass_through_and_drop_the_reads_they_affect():
    inner = CountingStore()
    cache = ReadCache(inner)
    assert cache.find(Event) == []

    event = Event(id="e1", title="An event")
    cache.put(event)

    assert inner.get(Event, "e1") is not None  # written for real
    assert [e.id for e in cache.find(Event)] == ["e1"]  # the cached list was refreshed
    assert cache.delete(Event, "e1") is True
    assert cache.find(Event) == [] and cache.get(Event, "e1") is None


def test_audit_entries_are_buffered_in_order_and_written_on_flush():
    inner = CountingStore()
    cache = ReadCache(inner)
    for n in range(3):
        cache.trace(TraceEvent(id=f"t{n}", run_id="r", agent="a", step=str(n), decision="d"))

    assert inner.trace_writes == 0  # nothing yet
    cache.flush()

    assert [t.step for t in inner.traces("r")] == ["0", "1", "2"]
    assert inner.trace_writes == 3


# ------------------------------------------------------------------ the cycle


def newsroom(store=None, **settings) -> Newsroom:
    return Newsroom(
        settings=Settings(llm_provider="fake", mongodb_uri="", **settings),
        store=store or MemoryStore(),
        llm=FakeLLM(collect_fake_handlers()),
        persistent_checkpoints=False,
    )


def test_clustering_is_rule_based_unless_the_model_is_switched_on():
    assert Settings.model_fields["architect_uses_model"].default is False  # the shipped default
    nr = newsroom(architect_uses_model=False)

    nr.architect.process(nr.monitor.scan([], fixtures=load_fixture_articles(paywalled=False)))

    asked = [task for task, _ in nr.ctx.llm.calls]
    assert "architect_analyse" not in asked


def test_rule_based_clustering_still_joins_the_same_news_across_outlets_and_languages():
    nr = newsroom(architect_uses_model=False)
    articles = nr.monitor.scan([], fixtures=load_fixture_articles(paywalled=False))

    nr.architect.process(articles)

    by_id = {a.id: nr.store.get(Article, a.id) for a in articles}
    permit = {by_id["fx-kaleva-permit-1"].event_id, by_id["fx-migri-permit-1"].event_id}
    assert len(permit) == 1 and None not in permit  # Finnish Kaleva + English Migri: one event
    housing = {by_id["fx-kaleva-levi-1"].event_id, by_id["fx-yle-levi-1"].event_id}
    assert len(housing) == 1 and None not in housing
    assert by_id["fx-kaleva-sports-1"].event_id not in permit | housing  # unrelated stays apart
    assert nr.store.get(Event, next(iter(permit))).status == "confirmed"  # 2 outlets


def test_the_model_can_still_be_switched_on_for_clustering():
    nr = newsroom(architect_uses_model=True)

    nr.architect.process(nr.monitor.scan([], fixtures=load_fixture_articles(paywalled=False)))

    assert "architect_analyse" in [task for task, _ in nr.ctx.llm.calls]


def test_a_batch_uses_few_database_round_trips_per_article():
    store = CountingStore()
    nr = newsroom(store=store, architect_uses_model=False)
    articles = nr.monitor.scan([], fixtures=load_fixture_articles(paywalled=False))
    store.reads = store.writes = store.trace_writes = 0

    nr.architect.process(articles)

    per_article = (store.reads + store.writes + store.trace_writes) / len(articles)
    assert per_article <= 14, per_article  # it was about 28 before the cache


# ------------------------------------------------------------------ side-by-side work


def test_the_call_counter_is_exact_when_many_threads_call_at_once():
    import threading

    from newsroom.limits import CycleLimits

    limits = CycleLimits(max_model_calls=10**6, max_seconds=60)
    limits.start()

    def hammer():
        for _ in range(2000):
            limits.count_call()

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert limits.calls == 16000


def test_running_stories_and_languages_side_by_side_gives_the_same_result():
    def cycle(**workers):
        nr = newsroom(**workers)
        report = nr.run_cycle(None, use_fixtures=True)
        calls = sorted(task for task, _ in nr.ctx.llm.calls)
        return (
            [(r.community_id, r.working_title, r.status) for r in report.runs],
            calls,
            report.pitches,
        )

    one_at_a_time = cycle(story_workers=1, translation_workers=1)
    side_by_side = cycle(story_workers=4, translation_workers=4)

    assert side_by_side == one_at_a_time  # same stories, same order, same model calls


def test_back_translation_is_only_for_risky_stories():
    from newsroom.domain import RiskTier

    nr = newsroom()
    nr.run_cycle(["arabic-speakers-finland"], use_fixtures=True)
    lanes = nr.communities["arabic-speakers-finland"].languages
    expected = 0
    for story in nr.store.find(Story):
        machine = [
            lane for lane in lanes if lane.code != story.language and lane.human_review != "human_only"
        ]
        if story.risk_tier is RiskTier.HIGH:
            expected += len(machine)  # every machine translation is checked
        elif story.risk_tier is RiskTier.ELEVATED:
            expected += sum(1 for lane in machine if lane.tier == "B")
        # LOW: no second call at all, even for a tier-B language such as Arabic

    asked = [task for task, _ in nr.ctx.llm.calls]
    assert asked.count("translator.backtranslate") == expected
