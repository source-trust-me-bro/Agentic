"""Connector agent: builds the semantic relationship graph between topics.

It goes through the topics, decides which two relate to each other and adds each one's id to
the other's ``branches``. Two signals are used:

* Evidence from the news itself: an event filed under two topics links them. This needs no
  model and is always applied.
* Meaning: the model reads every topic (name, keywords, recent events) and proposes related
  pairs with a reason. Code validates each pair: both ids must exist, no self links, and no
  parent/child links (the tree already holds those).

Links are only ever added, never removed, and always in both directions.
"""

from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel, Field

from newsroom.agents.base import Agent
from newsroom.domain import Event, Topic
from newsroom.parallel import run_parallel

MAX_TOPICS_PER_CALL = 120
MAX_EVENTS_PER_TOPIC = 5


class TopicLink(BaseModel):
    a: str  # short key of one topic, e.g. "t3"
    b: str
    reason: str = ""


class TopicLinks(BaseModel):
    links: list[TopicLink] = Field(default_factory=list)


SYSTEM = (
    "You connect topics in a news knowledge graph. Two topics are related when news in one "
    "often matters for, causes, or is read together with news in the other (for example "
    "'Residence permits' and 'Work and employment', or 'Energy prices' and 'Housing and rent'). "
    "Do not link a topic to itself. Do not link topics only because both are broad. Use only the "
    "keys given. Give a short reason for each link."
)


class ConnectorAgent(Agent):
    name = "connector"

    def connect(self) -> int:
        """Link related topics. Returns how many new links (pairs) were added."""

        topics = {t.id: t for t in self.store.find(Topic)}
        if len(topics) < 2:
            return 0
        events = [e for e in self.store.find(Event) if e.status != "merged"]

        pairs: dict[tuple[str, str], str] = {}
        for event in events:
            ids = sorted({tid for tid in event.topic_ids if tid in topics})
            for a, b in combinations(ids, 2):
                pairs.setdefault((a, b), f"both cover the event '{event.title[:80]}'")
        for (a, b), reason in self._model_pairs(topics, events).items():
            pairs.setdefault((a, b), reason)

        added = 0
        changed: set[str] = set()
        for (a, b), reason in pairs.items():
            first, second = topics[a], topics[b]
            if first.parent_id == b or second.parent_id == a:
                continue  # the tree already says how a parent and its child relate
            if b in first.branches and a in second.branches:
                continue
            if b not in first.branches:
                first.branches.append(b)
            if a not in second.branches:
                second.branches.append(a)
            changed.update((a, b))
            added += 1
            self.trace("link", "linked", reason[:300], topic_a=a, topic_b=b)
        for topic_id in changed:
            self.store.put(topics[topic_id])
        self.trace("connect", "done", f"{added} new link(s) between {len(topics)} topics")
        return added

    def _model_pairs(
        self, topics: dict[str, Topic], events: list[Event]
    ) -> dict[tuple[str, str], str]:
        """Related pairs proposed by the model, keyed by sorted topic ids."""

        recent: dict[str, list[str]] = {}
        for event in sorted(events, key=lambda e: e.last_updated, reverse=True):
            for tid in event.topic_ids:
                titles = recent.setdefault(tid, [])
                if len(titles) < MAX_EVENTS_PER_TOPIC:
                    titles.append(event.title[:100])

        # The busiest topics first, so a very large tree still links what matters most.
        ordered = sorted(topics.values(), key=lambda t: len(recent.get(t.id, [])), reverse=True)
        batches = [
            ordered[start : start + MAX_TOPICS_PER_CALL]
            for start in range(0, len(ordered), MAX_TOPICS_PER_CALL)
        ]

        def ask(batch: list[Topic]) -> tuple[dict[str, str], TopicLinks] | None:
            if self.ctx.limits.stop_reason():
                return None
            keys = {f"t{n}": topic.id for n, topic in enumerate(batch, 1)}
            lines = []
            for key, topic_id in keys.items():
                topic = topics[topic_id]
                parent = topics.get(topic.parent_id or "")
                lines.append(
                    f"{key}: {topic.name}"
                    + (f" (under {parent.name})" if parent else "")
                    + (f" | keywords: {', '.join(topic.keywords[:12])}" if topic.keywords else "")
                    + (f" | recent: {'; '.join(recent[topic_id])}" if recent.get(topic_id) else "")
                )
            try:
                result = self.llm.generate(
                    task="connector.link_topics",
                    system=SYSTEM,
                    prompt="Topics:\n" + "\n".join(lines),
                    schema=TopicLinks,
                    payload={"keys": list(keys)},
                )
            except Exception as exc:  # noqa: BLE001 - the graph from events still stands
                self.trace("model", "unavailable", str(exc)[:300])
                return None
            return keys, result

        out: dict[tuple[str, str], str] = {}
        # The batches are independent questions, so they are asked side by side.
        for answered in run_parallel(ask, batches, self.settings.claim_workers, "connector"):
            if answered is None:
                continue
            keys, result = answered
            for link in result.links:
                a, b = keys.get(link.a.strip()), keys.get(link.b.strip())
                if not a or not b or a == b:
                    self.trace("model", "rejected", f"unknown or self link {link.a}-{link.b}")
                    continue
                out.setdefault(tuple(sorted((a, b))), link.reason or "related in meaning")
        return out
