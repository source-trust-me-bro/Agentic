"""Source librarian: attaches the basic official reference to a story automatically.

Editors then check it and add more. This agent uses a curated registry and the policy
topics, not a model, so what it attaches is predictable and reviewable. An official
reference tells readers where to confirm the facts; it is not proof of any claim.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from newsroom import guards
from newsroom.agents.base import Agent, AgentContext
from newsroom.domain import Community, SourceRef, Story, utcnow
from newsroom.officials import OfficialSource, load_official_sources, match_official_sources

FAKE_HANDLERS: dict[str, Callable[[dict[str, Any]], BaseModel]] = {}  # rule-based: no model task

REFERENCE_NOTE = (
    "Official reference added by the newsroom so you can confirm the current rules. "
    "It does not verify the claims in this story."
)


class SourceLibrarianAgent(Agent):
    name = "source_librarian"

    def __init__(self, ctx: AgentContext, registry: list[OfficialSource] | None = None) -> None:
        super().__init__(ctx)
        self._registry = registry if registry is not None else load_official_sources()

    def topics_for(self, story: Story) -> list[str]:
        text = f"{story.headline} {story.standfirst} {story.body_md}"
        _tier, topics = guards.classify_risk(text, self.policy)
        topics = list(topics)
        # Elevated topics (tax, housing) deserve a reference too, even when a higher
        # risk topic already matched.
        cleaned = guards.sanitize(text)
        for name, rule in self.policy.elevated_topics.items():
            if name not in topics and guards._matches(cleaned, rule.keywords):
                topics.append(name)
        return topics

    def attach(self, story: Story, community: Community) -> Story:
        """Attach matching official references that are not already linked."""

        topics = self.topics_for(story)
        matches = match_official_sources(topics, community.regions, registry=self._registry)
        existing = {ref.url.rstrip("/") for ref in story.sources}
        added: list[str] = []
        for source in matches:
            if source.url.rstrip("/") in existing:
                continue
            story.sources.append(
                SourceRef(
                    name=source.name,
                    url=source.url,
                    official=True,
                    origin="agent_reference",
                    note=REFERENCE_NOTE,
                )
            )
            added.append(source.id)
        if added:
            story.updated_at = utcnow()
            self.store.put(story)
        self.trace(
            "attach_official_sources",
            "attached" if added else "none",
            f"Topics {topics or 'none'}; attached {added or 'nothing'}. Editors review these.",
            story_id=story.id,
            topics=topics,
            attached=added,
        )
        return story
