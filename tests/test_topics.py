from newsroom.domain import Event, Topic
from newsroom.store import MemoryStore
from newsroom.topics import get_tree, seed_topics, slugify, unfiled_events


def test_slugify_is_ascii_and_stable():
    assert slugify("Local life · Oulu region") == "local-life-oulu-region"
    assert slugify("Sámi languages and education") == "sami-languages-and-education"


def test_seed_topics_is_idempotent():
    store = MemoryStore()
    first = seed_topics(store)
    snapshot = {t.id: t.model_dump() for t in store.find(Topic)}
    again = seed_topics(store)
    assert [t.id for t in first] == [t.id for t in again]
    assert {t.id: t.model_dump() for t in store.find(Topic)} == snapshot
    assert len(snapshot) == len(first) > 10


def test_get_tree_is_nested_and_files_events_under_topics():
    store = MemoryStore()
    seed_topics(store)
    permits = next(t for t in store.find(Topic) if t.name == "Residence permits and visas")
    event = Event(id="evt_1", title="Permit times", topic_ids=[permits.id])
    merged = Event(id="evt_2", title="Old", topic_ids=[permits.id], status="merged")
    loose = Event(id="evt_3", title="No topic yet")
    for e in (event, merged, loose):
        store.put(e)

    roots = get_tree(store)
    names = [r.topic.name for r in roots]
    assert "Immigration and rights" in names
    root = next(r for r in roots if r.topic.name == "Immigration and rights")
    child = next(c for c in root.children if c.topic.id == permits.id)
    assert [e.id for e in child.events] == ["evt_1"]  # merged events are hidden
    assert [e.id for e in unfiled_events(store)] == ["evt_3"]


# ---------------------------------------------------- emptying the database while running


def test_an_emptied_database_gets_its_topics_back_without_a_restart(newsroom):
    from fastapi.testclient import TestClient

    from newsroom.api.app import create_app
    from newsroom.domain import Community, Topic

    store = newsroom.store
    for topic in store.find(Topic):
        store.delete(Topic, topic.id)
    for community in store.find(Community):
        store.delete(Community, community.id)
    assert store.find(Topic) == []

    with TestClient(create_app(newsroom)) as client:
        tree = client.get("/tree").json()
        overview = client.get("/overview").json()

    assert tree, "reading the tree puts the configured topics back"
    assert len(overview["topics"]) >= len(tree)
    assert len(store.find(Community)) == len(newsroom.communities)


def test_a_cycle_restores_wiped_topics_before_it_starts(newsroom):
    from newsroom.domain import Topic

    for topic in newsroom.store.find(Topic):
        newsroom.store.delete(Topic, topic.id)

    newsroom.run_cycle(["new-in-finland"], use_fixtures=True)

    assert newsroom.store.find(Topic)


# ------------------------------------------------ topics the agents make on their own


def _proposed(store, slug, parent_id, events):
    from newsroom.domain import Event, Topic

    topic = Topic(
        id=f"topic_{slug}", name=slug, slug=slug, parent_id=parent_id, status="proposed"
    )
    store.put(topic)
    for n, sources in enumerate(events):
        store.put(
            Event(id=f"ev_{slug}_{n}", title=f"event {n}", topic_ids=[topic.id], source_names=sources)
        )
    return topic


def _active_parent(store):
    from newsroom.domain import Topic

    return next(t for t in store.find(Topic) if t.status == "active")


def test_a_well_supported_agent_topic_becomes_active_by_itself(newsroom):
    from newsroom.domain import Topic

    store = newsroom.store
    topic = _proposed(store, "bus-timetables", _active_parent(store).id, [["Yle"], ["Kaleva"]])

    promoted = newsroom.architect.promote_corroborated_topics()

    assert [t.id for t in promoted] == [topic.id]
    assert store.get(Topic, topic.id).status == "active"
    assert any(t.step == "topic" and t.decision == "activated" for t in store.traces())


def test_a_thinly_supported_topic_stays_proposed(newsroom):
    from newsroom.domain import Topic

    store = newsroom.store
    parent = _active_parent(store).id
    one_event = _proposed(store, "one-event", parent, [["Yle", "Kaleva"]])
    one_outlet = _proposed(store, "one-outlet", parent, [["Yle"], ["Yle"]])

    assert newsroom.architect.promote_corroborated_topics() == []

    assert store.get(Topic, one_event.id).status == "proposed"
    assert store.get(Topic, one_outlet.id).status == "proposed"


def test_a_new_top_level_topic_or_one_under_an_inactive_parent_still_needs_a_human(newsroom):
    from newsroom.domain import Topic

    store = newsroom.store
    top_level = _proposed(store, "brand-new-area", None, [["Yle"], ["Kaleva"]])
    inactive_parent = _proposed(store, "waiting-parent", _active_parent(store).id, [])
    orphan = _proposed(store, "child-of-waiting", inactive_parent.id, [["Yle"], ["Kaleva"]])

    assert newsroom.architect.promote_corroborated_topics() == []

    assert store.get(Topic, top_level.id).status == "proposed"
    assert store.get(Topic, orphan.id).status == "proposed"


def test_merged_events_do_not_count_toward_a_topic(newsroom):
    from newsroom.domain import Event, Topic

    store = newsroom.store
    topic = _proposed(store, "merged-news", _active_parent(store).id, [["Yle"], ["Kaleva"]])
    event = store.get(Event, "ev_merged-news_1")
    event.status = "merged"
    store.put(event)

    assert newsroom.architect.promote_corroborated_topics() == []
    assert store.get(Topic, topic.id).status == "proposed"


def test_the_topic_export_lists_only_active_topics_and_includes_promoted_ones(newsroom):
    from fastapi.testclient import TestClient

    from newsroom.api.app import create_app

    store = newsroom.store
    waiting = _proposed(store, "still-waiting", _active_parent(store).id, [["Yle"]])
    made = _proposed(store, "made-by-agent", _active_parent(store).id, [["Yle"], ["Kaleva"]])
    newsroom.architect.promote_corroborated_topics()

    with TestClient(create_app(newsroom)) as client:
        ids = {t["id"] for t in client.get("/export/topics").json()["topics"]}

    assert made.id in ids and waiting.id not in ids
    assert len(ids) >= 22  # the configured topics are there too
