"""Shared builders for the explainer / translator / standards / distributor tests."""

from __future__ import annotations

from newsroom.agents.base import AgentContext
from newsroom.agents.explainer import ExplainerAgent
from newsroom.agents.standards import StandardsEditorAgent
from newsroom.domain import (
    Article,
    Claim,
    ClaimStatus,
    Community,
    Decision,
    EvidenceLink,
    Pitch,
    ReviewDecision,
    ReviewRole,
    StandardsReport,
    Story,
    UserNeed,
    new_id,
)
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.registry import load_communities
from newsroom.store import MemoryStore


def make_ctx() -> AgentContext:
    return AgentContext(llm=FakeLLM(collect_fake_handlers()), store=MemoryStore())


def community(community_id: str) -> Community:
    return next(c for c in load_communities() if c.id == community_id)


def article(
    art_id: str, source: str, url: str, title: str, content: str, *, official: bool = False
) -> Article:
    return Article(
        id=art_id,
        source_id=source.lower(),
        source_name=source,
        url=url,
        title=title,
        content=content,
        language="en",
        official=official,
    )


def permit_articles() -> list[Article]:
    return [
        article(
            "a-migri",
            "Migri",
            "https://example.test/migri/processing-times",
            "Processing times for work-based residence permits",
            "Migri says the average processing time for work-based residence permits is 45 days.",
            official=True,
        ),
        article(
            "a-kaleva",
            "Kaleva",
            "https://example.test/kaleva/permits",
            "Residence permit processing slows in Oulu",
            "Processing now takes 45 days on average, up from 30 days last year.",
        ),
    ]


def link(art: Article, relation: str = "supports") -> EvidenceLink:
    return EvidenceLink(
        article_id=art.id, url=art.url, source_name=art.source_name, relation=relation
    )  # type: ignore[arg-type]


def permit_claims(pitch_id: str, articles: list[Article]) -> list[Claim]:
    migri, kaleva = articles
    employers = article(
        "a-fed", "Employers' federation", "https://example.test/fed/rent", "Rent", "Rent is fair."
    )
    return [
        Claim(
            id=new_id("claim"),
            pitch_id=pitch_id,
            text="The average processing time for work-based residence permits is 45 days",
            status=ClaimStatus.VERIFIED,
            evidence=[link(migri), link(kaleva)],
        ),
        Claim(
            id=new_id("claim"),
            pitch_id=pitch_id,
            text="Applicants should apply at least two months before the work starts",
            status=ClaimStatus.ATTRIBUTED,
            attribution="Migri",
            source_article_id=migri.id,
            evidence=[link(migri)],
        ),
        Claim(
            id=new_id("claim"),
            pitch_id=pitch_id,
            text="Some employers charge more for accommodation than the collective agreement allows",
            status=ClaimStatus.DISPUTED,
            evidence=[link(kaleva), link(employers, "contradicts")],
        ),
        Claim(
            id=new_id("claim"),
            pitch_id=pitch_id,
            text="Processing times will fall back to 30 days in spring",
            status=ClaimStatus.UNKNOWN,
            unknowns=["No source says when processing times will improve."],
        ),
    ]


def pitch_for(
    community_id: str, articles: list[Article], title: str = "Residence permit delays"
) -> Pitch:
    return Pitch(
        id=new_id("pitch"),
        community_id=community_id,
        article_ids=[a.id for a in articles],
        working_title=title,
        rationale="Affects newcomers' start dates.",
        needs=[UserNeed.USEFULNESS],
    )


def write_story(
    ctx: AgentContext,
    community_id: str = "oulu-international",
    *,
    articles: list[Article] | None = None,
    claims: list[Claim] | None = None,
    title: str = "Residence permit delays",
) -> tuple[Story, list[Claim], Community, list[Article]]:
    comm = community(community_id)
    arts = articles if articles is not None else permit_articles()
    pitch = pitch_for(community_id, arts, title)
    cls = claims if claims is not None else permit_claims(pitch.id, arts)
    story = ExplainerAgent(ctx).write(pitch, cls, comm, arts)
    return story, cls, comm, arts


def review(ctx: AgentContext, story, claims, comm, translations=()) -> StandardsReport:
    return StandardsEditorAgent(ctx).review(story, claims, list(translations), comm)


def approvals_for(
    story: Story, report: StandardsReport, *, reviewer: str = "Anna Virtanen"
) -> list[ReviewDecision]:
    roles = list(dict.fromkeys([ReviewRole.EDITOR_IN_CHIEF, *report.required_reviews]))
    return [
        ReviewDecision(
            id=new_id("rev"),
            story_id=story.id,
            story_version=story.version,
            role=role,
            reviewer=f"{reviewer} ({role.value})",
            decision=Decision.APPROVE,
        )
        for role in roles
    ]
