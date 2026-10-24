"""Storage helpers for the persistent news tree (topics → events → articles).

No model is involved here: seeding and reading the tree is plain, deterministic code.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from newsroom.config import ROOT
from newsroom.domain import Event, Topic
from newsroom.store.base import Store

TOPICS_FILE = ROOT / "config" / "topics.yaml"


def slugify(name: str) -> str:
    """ASCII-friendly stable slug: 'Local life · Oulu region' -> 'local-life-oulu-region'."""

    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "topic"


def topic_id_for(slug: str) -> str:
    return f"topic_{slug}"


class TopicNode(BaseModel):
    topic: Topic
    events: list[Event] = Field(default_factory=list)
    children: list[TopicNode] = Field(default_factory=list)


TopicNode.model_rebuild()


@lru_cache
def expected_topic_ids(path: str | None = None) -> frozenset[str]:
    """The ids of every topic in config/topics.yaml. Read once, so checking that the database
    has them all costs one query instead of one per topic."""

    raw = yaml.safe_load(Path(path or TOPICS_FILE).read_text(encoding="utf-8")) or {}
    ids: set[str] = set()

    def walk(items: list[dict]) -> None:
        for item in items:
            ids.add(topic_id_for(slugify(item["name"])))
            walk(item.get("children", []) or [])

    walk(raw.get("topics", []))
    return frozenset(ids)


def seed_topics(store: Store, path: str | Path | None = None) -> list[Topic]:
    """Load config/topics.yaml into Topic docs. Idempotent: existing topics are kept as is."""

    raw = yaml.safe_load(Path(path or TOPICS_FILE).read_text(encoding="utf-8")) or {}
    out: list[Topic] = []

    def walk(items: list[dict], parent_id: str | None) -> None:
        for item in items:
            slug = slugify(item["name"])
            doc_id = topic_id_for(slug)
            existing = store.get(Topic, doc_id)
            if existing is None:
                existing = Topic(
                    id=doc_id,
                    name=item["name"],
                    slug=slug,
                    parent_id=parent_id,
                    description=item.get("description", ""),
                    keywords=[str(k) for k in item.get("keywords", [])],
                )
                store.put(existing)
            keywords = [str(k) for k in item.get("keywords", [])]
            if existing.keywords != keywords and existing.status == "active":
                # The config is the source of truth for the topics it defines.
                existing.keywords = keywords
                store.put(existing)
            out.append(existing)
            walk(item.get("children", []) or [], doc_id)

    walk(raw.get("topics", []), None)
    return out


def get_tree(store: Store, *, include_proposed: bool = True) -> list[TopicNode]:
    """Nested tree of topics with the (non-merged) events filed under each."""

    topics = store.find(Topic)
    if not include_proposed:
        topics = [t for t in topics if t.status == "active"]
    events = [e for e in store.find(Event) if e.status != "merged"]

    nodes = {t.id: TopicNode(topic=t) for t in topics}
    for event in sorted(events, key=lambda e: e.last_updated, reverse=True):
        for topic_id in event.topic_ids:
            if topic_id in nodes:
                nodes[topic_id].events.append(event)

    roots: list[TopicNode] = []
    for node in sorted(nodes.values(), key=lambda n: n.topic.name.casefold()):
        parent = nodes.get(node.topic.parent_id) if node.topic.parent_id else None
        (parent.children if parent else roots).append(node)
    return roots


def unfiled_events(store: Store) -> list[Event]:
    """Events not attached to any topic yet (waiting for a branch or a human)."""

    return [e for e in store.find(Event) if e.status != "merged" and not e.topic_ids]
