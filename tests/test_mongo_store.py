"""Everything the newsroom produces must survive in the database, not just in memory."""

from __future__ import annotations

import mongomock
import pytest
from tests.conftest import approve_everything

from newsroom.config import Settings
from newsroom.domain import (
    Article,
    Claim,
    Event,
    Pitch,
    Publication,
    Story,
    Topic,
    TraceEvent,
    new_id,
)
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.registry import load_fixture_articles
from newsroom.service import Newsroom
from newsroom.store.mongo import MongoStore


@pytest.fixture
def mongo_db():
    return mongomock.MongoClient(tz_aware=True)["agentic_newsroom_test"]


@pytest.fixture
def mongo_newsroom(mongo_db) -> Newsroom:
    return Newsroom(
        settings=Settings(llm_provider="fake", mongodb_uri=""),
        store=MongoStore(mongo_db),
        llm=FakeLLM(collect_fake_handlers()),
        persistent_checkpoints=False,
    )


def test_an_article_is_saved_once_and_read_back_intact(mongo_db):
    store = MongoStore(mongo_db)
    article = load_fixture_articles(paywalled=False)[0]

    saved, created = store.upsert_article(article)
    again, created_again = store.upsert_article(article.model_copy(update={"id": new_id("art")}))

    assert created is True and created_again is False  # the same URL is never saved twice
    assert again.id == saved.id
    assert mongo_db["articles"].count_documents({}) == 1
    loaded = store.get(Article, saved.id)
    assert loaded is not None
    assert loaded.title == article.title and loaded.language == article.language
    assert loaded.published_at == article.published_at


def test_paywalled_articles_keep_only_what_the_feed_gives(mongo_db):
    store = MongoStore(mongo_db)
    kaleva = next(a for a in load_fixture_articles() if a.id == "fx-kaleva-permit-1")

    store.upsert_article(kaleva)

    raw = mongo_db["articles"].find_one({"_id": kaleva.id})
    assert raw["access"] == "metadata_only"
    assert not raw.get("content")  # nothing behind the paywall is ever stored


def test_a_whole_cycle_is_persisted_in_the_database(mongo_newsroom, mongo_db):
    report = mongo_newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    run = next(r for r in report.runs if r.pending)

    counts = {name: mongo_db[name].count_documents({}) for name in mongo_db.list_collection_names()}
    assert counts["articles"] == 8  # every collected article, including flagged ones
    assert counts["events"] >= 1 and counts["topics"] >= 1
    assert counts["pitches"] >= 1 and counts["claims"] >= 1
    assert counts["stories"] >= 1 and counts["translations"] >= 1
    assert counts["standards_reports"] >= 1
    assert counts["agent_runs"] >= 1  # the audit trail is saved too
    assert counts["communities"] == 3

    # Articles are linked to their event and the event to a topic, all in the database.
    linked = mongo_db["articles"].count_documents({"event_id": {"$ne": None}})
    assert linked >= 4
    assert mongo_db["events"].find_one({"topic_ids": {"$ne": []}}) is not None

    # A published story and who approved it are saved as well.
    done = approve_everything(mongo_newsroom, run)
    assert done.status == "published"
    assert mongo_db["publications"].count_documents({}) == 1
    publication = mongo_db["publications"].find_one({})
    assert publication["approval_ids"]
    assert mongo_db["reviews"].count_documents({}) >= len(publication["approval_ids"])


def test_the_data_survives_a_restart(mongo_db):
    first = Newsroom(
        settings=Settings(llm_provider="fake", mongodb_uri=""),
        store=MongoStore(mongo_db),
        llm=FakeLLM(collect_fake_handlers()),
        persistent_checkpoints=False,
    )
    first.run_cycle(["lapland-seasonal"], use_fixtures=True)
    articles_before = mongo_db["articles"].count_documents({})

    restarted_store = MongoStore(mongo_db)  # a new process connecting to the same database

    assert len(restarted_store.find(Article)) == articles_before
    assert restarted_store.find(Event) and restarted_store.find(Topic)
    assert (
        restarted_store.find(Story) and restarted_store.find(Pitch) and restarted_store.find(Claim)
    )
    assert restarted_store.traces()


def test_documents_round_trip_with_every_collection_type(mongo_db):
    store = MongoStore(mongo_db)
    store.trace(TraceEvent(id=new_id("t"), run_id="r1", agent="a", step="s", decision="d"))

    assert [t.run_id for t in store.traces("r1")] == ["r1"]
    assert store.get(Publication, "missing") is None


def test_search_finds_saved_articles_across_languages(mongo_db):
    store = MongoStore(mongo_db)
    for article in load_fixture_articles(paywalled=False):
        store.upsert_article(article)

    hits = store.search_articles("Migri residence permit processing days", limit=5)

    assert hits and any(
        "Migri" in (a.summary or a.title) or "permit" in a.title.lower() for a in hits
    )
