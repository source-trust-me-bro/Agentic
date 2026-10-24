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
