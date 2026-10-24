from __future__ import annotations

from _b_helpers import make_ctx, review, write_story
from newsroom.agents.standards import StandardsEditorAgent
from newsroom.domain import ContextNote, ReviewRole, RiskTier, Translation
from newsroom.llm.base import LLMError


def _names(report, *, failed_only=True):
    return {c.name for c in report.checks if not (failed_only and c.passed)}


def _clean_story(ctx, community_id="oulu-international", **kwargs):
    story, claims, comm, arts = write_story(ctx, community_id, **kwargs)
    story.context_notes = [n for n in story.context_notes if n.kind != "human_needed"]
    return story, claims, comm, arts


def test_permit_story_is_high_risk_and_needs_experts():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)
    report = review(ctx, story, claims, comm)

    assert report.risk_tier is RiskTier.HIGH
    assert report.blocking_failures == []
    assert ReviewRole.EDITOR_IN_CHIEF in report.required_reviews
    assert ReviewRole.SUBJECT_EXPERT in report.required_reviews
    assert ReviewRole.COMMUNITY_REVIEWER in report.required_reviews  # Oulu lists trusted voices
    assert any(c.guideline == "7–17" for c in report.checks)
    assert any(c.guideline == "35–37" for c in report.checks)


def test_missing_official_source_blocks_a_high_risk_story():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)
    story.sources = [s.model_copy(update={"official": False}) for s in story.sources]
    report = review(ctx, story, claims, comm)

    assert "High-risk story links an official source" in _names(report)
    assert any(c.name.startswith("High-risk") for c in report.blocking_failures)
    assert any("official" in line.lower() for line in report.revision_feedback)


def test_low_risk_story_only_needs_the_editor_in_chief():
    ctx = make_ctx()
    from _b_helpers import article, link
    from newsroom.domain import Claim, ClaimStatus, new_id

    art = article(
        "a-bus", "City of Oulu", "https://example.test/oulu/bus", "Winter timetables",
        "Winter timetables start on 3 November.", official=True,
    )  # fmt: skip
    claim = Claim(
        id=new_id("claim"),
        pitch_id="p1",
        text="Winter timetables start on 3 November",
        status=ClaimStatus.VERIFIED,
        evidence=[link(art)],
    )
    story, claims, comm, _ = _clean_story(
        ctx, "sapmi-north", articles=[art], claims=[claim], title="Winter bus timetables"
    )
    story.context_notes = []
    report = review(ctx, story, claims, comm)

    assert report.risk_tier is RiskTier.LOW
    assert report.required_reviews == [ReviewRole.EDITOR_IN_CHIEF]
    assert report.blocking_failures == []


def test_human_needed_context_is_a_warning_for_the_editor_not_an_expert_requirement():
    ctx = make_ctx()
    story, claims, comm, _ = write_story(ctx)  # fake explainer adds a human_needed note
    report = review(ctx, story, claims, comm)

    assert any(n.kind == "human_needed" for n in story.context_notes)
    assert any("Ungrounded context" in c.name and not c.passed for c in report.checks)
    # It reaches the editor through the revision feedback, and the editor-in-chief approves.
    assert any("human" in line.lower() for line in report.revision_feedback)
    if report.risk_tier is not RiskTier.HIGH:
        assert report.required_reviews == [ReviewRole.EDITOR_IN_CHIEF]


def test_language_review_gates_its_own_language_not_the_story():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)

    def tr(language, **kw):
        return Translation(
            id=f"tr-{language}", story_id=story.id, story_version=story.version,
            language=language, body_md="x", **kw,
        )  # fmt: skip

    for translations in (
        [tr("uk")],  # lane human_review == required
        [tr("ar", status="needs_human_translation", machine_translated=False)],
        [tr("en", backtranslation_flags=["number 45 changed"])],
    ):
        report = review(ctx, story, claims, comm, translations)
        # The story is not held back for a language reviewer...
        assert ReviewRole.LANGUAGE_REVIEWER not in report.required_reviews
        # ...but the language is flagged so it goes out only after human review.
        assert any(
            "Language review" in c.name or "Human translation needed" in c.name
            for c in report.checks
        )


def test_blocking_rules():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)

    def failing(**changes):
        mutated = story.model_copy(update=changes)
        return _names(review(ctx, mutated, claims, comm))

    assert "No stigmatising wording" in failing(body_md=story.body_md + "\nThese people are lazy.")
    assert "No private details" in failing(body_md=story.body_md + "\nCall +358 40 123 4567.")
    assert "No hidden advertising" in failing(body_md=story.body_md + "\nUse code WINTER10.")
    assert "AI role is disclosed" in failing(ai_disclosure="")
    assert "Readers can report an error" in failing(ai_disclosure="Prepared with AI.")
    assert "Headline is not sensational" in failing(headline="SHOCKING!! Permit delays")
    assert "Sources acknowledged with links" in failing(sources=[])
    assert "Headline matches substance" in failing(headline="Quantum telescope discovers volcano")
    assert "Story has a body" in failing(body_md="")


def test_unknown_claim_stated_as_fact_blocks():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)
    flat = "Processing times will fall back to 30 days in spring."
    mutated = story.model_copy(update={"body_md": flat + "\n\n" + story.body_md})
    report = review(ctx, mutated, claims, comm)

    assert "Claims worded within the evidence" in _names(report)
    assert report.blocking_failures


def test_stated_claim_without_a_source_blocks():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)
    bare = [c.model_copy(update={"evidence": [], "source_article_id": None}) for c in claims]
    assert "Stated claims have sources" in _names(review(ctx, story, bare, comm))


def test_unavailable_second_opinion_fails_closed():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)

    def broken(payload):
        raise LLMError("quota")

    ctx.llm.register("standards.second_opinion", broken)
    report = review(ctx, story, claims, comm)

    assert "Second reading of headline and tone" in _names(report)
    assert report.blocking_failures


def test_right_of_reply_is_flagged_but_not_blocking():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)
    report = review(ctx, story, claims, comm)

    reply = next(c for c in report.checks if c.guideline == "18–22")
    assert not reply.passed and not reply.blocking  # employers allegedly overcharge
    assert any("reply" in line.lower() for line in report.revision_feedback)


def test_report_is_stored_and_agent_never_approves():
    ctx = make_ctx()
    story, claims, comm, _ = _clean_story(ctx)
    report = StandardsEditorAgent(ctx).review(story, claims, [], comm)

    from newsroom.domain import ReviewDecision, StandardsReport

    assert ctx.store.get(StandardsReport, report.id) is not None
    assert ctx.store.find(ReviewDecision) == []  # the copy chief never records an approval
    assert ContextNote  # imported for completeness
