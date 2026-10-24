from __future__ import annotations

from _b_helpers import make_ctx, permit_articles, permit_claims, pitch_for, write_story
from newsroom.agents.explainer import (
    ContextNoteDraft,
    ExplainerAgent,
    StoryDraft,
    verify_story_claims,
)
from newsroom.domain import ClaimStatus, RiskTier, Story, StoryStatus, UserNeed
from newsroom.llm.base import LLMError


def _text(story: Story) -> str:
    return f"{story.headline}\n\n{story.standfirst}\n\n{story.body_md}"


def test_story_follows_claim_rules():
    ctx = make_ctx()
    story, claims, comm, _ = write_story(ctx)
    text = _text(story)

    assert verify_story_claims(text, claims) == []
    assert "According to Migri" in text  # attributed claim keeps its attribution
    assert "Sources disagree" in text and "Employers' federation" in text  # two-sided
    assert "Not yet confirmed: Processing times will fall back to 30 days" in text
    assert "What we don't know yet" in story.body_md
    assert story.language == "en" and story.status is StoryStatus.DRAFT
    assert story.claim_ids == [c.id for c in claims]


def test_unknown_claim_is_never_stated_as_fact():
    ctx = make_ctx()
    story, claims, _, _ = write_story(ctx)
    unknown = next(c for c in claims if c.status is ClaimStatus.UNKNOWN)
    flat = f"{unknown.text}."
    lines = [line.strip(" -") for line in story.body_md.splitlines()]
    assert flat not in lines  # never a bare statement of fact...
    assert f"Not yet confirmed: {flat}" in lines  # ...only the hedged form
    assert verify_story_claims(_text(story), [unknown]) == []


def test_code_repairs_a_model_that_overstates_evidence():
    ctx = make_ctx()
    articles = permit_articles()
    pitch = pitch_for("oulu-international", articles)
    claims = permit_claims(pitch.id, articles)
    unknown = next(c for c in claims if c.status is ClaimStatus.UNKNOWN)
    attributed = next(c for c in claims if c.status is ClaimStatus.ATTRIBUTED)
    disputed = next(c for c in claims if c.status is ClaimStatus.DISPUTED)

    def overconfident(payload):
        body = (
            f"{unknown.text}. {attributed.text}. {disputed.text}.\n\nPlan around the new timeline."
        )
        return StoryDraft(headline="Permit delays", standfirst="Delays.", body_md=body)

    ctx.llm.register("explainer.write_story", overconfident)
    from newsroom.agents.explainer import ExplainerAgent as Agent

    story = Agent(ctx).write(
        pitch, claims, __import__("_b_helpers").community("oulu-international"), articles
    )

    assert verify_story_claims(_text(story), claims) == []
    assert f"{unknown.text}." not in [line.strip(" -") for line in story.body_md.splitlines()]
    assert "According to Migri" in story.body_md
    assert "Sources disagree" in story.body_md
    steps = [t.step for t in ctx.store.traces()]
    assert "repair" in steps


def test_ungrounded_context_becomes_human_needed():
    ctx = make_ctx()
    articles = permit_articles()
    pitch = pitch_for("oulu-international", articles)
    claims = permit_claims(pitch.id, articles)
    comm = __import__("_b_helpers").community("oulu-international")
    real_note = comm.cultural_context[0]

    def handler(payload):
        return StoryDraft(
            headline="Residence permit delays",
            standfirst="What delays mean.",
            body_md="The average processing time for work-based residence permits is 45 days.",
            needs=[UserNeed.UNDERSTANDING],
            context_notes=[
                ContextNoteDraft(
                    text="Permits are always approved within a week in Sweden.",
                    kind="from_story",
                ),
                ContextNoteDraft(
                    text="The ministry guarantees a decision.",
                    kind="official_source",
                    source_url="https://invented.test/ministry",
                ),
                ContextNoteDraft(text="Finns never queue for permits.", kind="community_knowledge"),
                ContextNoteDraft(text=real_note, kind="community_knowledge"),
            ],
        )

    ctx.llm.register("explainer.write_story", handler)
    story = ExplainerAgent(ctx).write(pitch, claims, comm, articles)

    kinds = {n.text: n.kind for n in story.context_notes}
    assert kinds["Permits are always approved within a week in Sweden."] == "human_needed"
    assert kinds["The ministry guarantees a decision."] == "human_needed"
    assert kinds["Finns never queue for permits."] == "human_needed"
    assert kinds[real_note] == "community_knowledge"
    assert all(n.source_url is None for n in story.context_notes if n.kind == "human_needed")
    assert f"**Community note:** {real_note}" in story.body_md


def test_official_source_with_a_known_url_is_kept():
    ctx = make_ctx()
    story, _, _, articles = write_story(ctx)
    official = next(n for n in story.context_notes if n.kind == "official_source")
    assert official.source_url == next(a.url for a in articles if a.official)


def test_metadata_risk_sources_and_disclosure():
    ctx = make_ctx()
    story, _, _, _ = write_story(ctx)

    assert story.needs  # at least one Kaleva user need
    assert story.risk_tier is RiskTier.HIGH  # permit story
    assert any(s.official for s in story.sources)
    assert "AI" in story.ai_disclosure
    assert ctx.store.get(Story, story.id) is not None


def test_feedback_is_recorded_and_versioned():
    ctx = make_ctx()
    comm = __import__("_b_helpers").community("oulu-international")
    articles = permit_articles()
    pitch = pitch_for("oulu-international", articles)
    claims = permit_claims(pitch.id, articles)

    story = ExplainerAgent(ctx).write(
        pitch, claims, comm, articles, feedback=["Add the official source."], version=2
    )

    assert story.version == 2
    assert any(t.step == "revise" and "official source" in t.reason for t in ctx.store.traces())


def test_provider_failure_fails_closed():
    ctx = make_ctx()

    def broken(payload):
        raise LLMError("quota exceeded")

    ctx.llm.register("explainer.write_story", broken)
    story, _, _, _ = write_story(ctx)

    assert story.body_md == ""
    assert story.status is StoryStatus.CHANGES_REQUESTED
    assert any(n.kind == "human_needed" for n in story.context_notes)


def test_verify_story_claims_rules():
    articles = permit_articles()
    pitch = pitch_for("oulu-international", articles)
    claims = permit_claims(pitch.id, articles)
    verified, attributed, disputed, unknown = claims

    assert verify_story_claims(f"{verified.text}.", claims) == []
    assert verify_story_claims(f"{attributed.text}.", claims)  # no attribution
    assert (
        verify_story_claims(
            f"Migri says {attributed.text[0].lower()}{attributed.text[1:]}.", claims
        )
        == []
    )
    assert verify_story_claims(f"{disputed.text}.", claims)  # one-sided
    assert verify_story_claims(f"{unknown.text}.", claims)  # unhedged
    assert (
        verify_story_claims(f"It is not yet confirmed that {unknown.text.lower()}.", claims) == []
    )
