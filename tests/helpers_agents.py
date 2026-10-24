"""Builders shared by the agent tests (fork A). No network, no real model."""

from __future__ import annotations

from datetime import UTC, datetime

from newsroom.agents.base import AgentContext
from newsroom.domain import Article, Community, Pitch, RiskTier, UserNeed, new_id
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.registry import load_communities
from newsroom.store.memory import MemoryStore


def make_ctx(*, empty_llm: bool = False) -> AgentContext:
    llm = FakeLLM() if empty_llm else FakeLLM(collect_fake_handlers())
    return AgentContext(llm=llm, store=MemoryStore())


def community(community_id: str) -> Community:
    return {c.id: c for c in load_communities()}[community_id]


def art(
    article_id: str,
    source_name: str,
    title: str,
    *,
    content: str | None = None,
    summary: str | None = None,
    language: str = "fi",
    official: bool = False,
    access: str = "full",
    published: str = "2026-10-22T10:00:00+00:00",
    source_id: str | None = None,
    event_id: str | None = None,
    syndication_key: str | None = None,
    url: str | None = None,
) -> Article:
    return Article(
        id=article_id,
        source_id=source_id or source_name.lower().replace(" ", "-"),
        source_name=source_name,
        url=url or f"https://example.test/{article_id}",
        title=title,
        summary=summary,
        content=content,
        language=language,
        published_at=datetime.fromisoformat(published).astimezone(UTC),
        official=official,
        access=access,  # type: ignore[arg-type]
        event_id=event_id,
        syndication_key=syndication_key,
    )


def store_articles(ctx: AgentContext, *articles: Article) -> None:
    for article in articles:
        ctx.store.upsert_article(article)


def pitch_for(community_id: str, *articles: Article, risk: RiskTier = RiskTier.LOW) -> Pitch:
    return Pitch(
        id=new_id("pitch"),
        community_id=community_id,
        article_ids=[a.id for a in articles],
        working_title=articles[0].title,
        rationale="test",
        needs=[UserNeed.INFORMATION],
        risk_tier=risk,
    )


def decisions(ctx: AgentContext, decision: str) -> list:
    return [t for t in ctx.store.traces(ctx.run_id) if t.decision == decision]


# Reusable report texts ---------------------------------------------------------------

KALEVA_PERMIT = (
    "Maahanmuuttovirasto Migri kertoo, että työperusteisten oleskelulupien käsittely kestää "
    "nyt keskimäärin 45 päivää. Viime vuonna käsittelyaika oli 30 päivää. Migrin mukaan "
    "hakemus kannattaa jättää vähintään kaksi kuukautta ennen työn alkamista."
)
MIGRI_PERMIT = (
    "The Finnish Immigration Service Migri says the average processing time for work-based "
    "residence permits is now 45 days. Last year the average was 30 days. Migri recommends "
    "applying at least two months before the work starts."
)
