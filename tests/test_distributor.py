from __future__ import annotations

import pytest

from _b_helpers import approvals_for, community, make_ctx, review, write_story
from newsroom.agents.distributor import (
    DistributorAgent,
    PublishRefused,
    spell_int,
)
from newsroom.agents.translator import TranslatorAgent
from newsroom.domain import (
    Channel,
    Decision,
    Publication,
    ReviewDecision,
    ReviewRole,
    Story,
    StoryStatus,
    Translation,
    new_id,
)


def _ready(ctx, community_id="oulu-international", **kwargs):
    story, claims, comm, arts = write_story(ctx, community_id, **kwargs)
    story.context_notes = [n for n in story.context_notes if n.kind != "human_needed"]
    report = review(ctx, story, claims, comm)
    assert report.blocking_failures == [], [c.name for c in report.blocking_failures]
    return story, comm, report


def _decision(story, role, reviewer="Anna Virtanen", *, version=None, decision=Decision.APPROVE):
    return ReviewDecision(
        id=new_id("rev"),
        story_id=story.id,
        story_version=version or story.version,
        role=role,
        reviewer=reviewer,
        decision=decision,
    )


def test_refuses_without_the_editor_in_chief():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    others = [d for d in approvals_for(story, report) if d.role is not ReviewRole.EDITOR_IN_CHIEF]

    with pytest.raises(PublishRefused, match="editor_in_chief"):
        DistributorAgent(ctx).render(story, [], comm, others)
    assert story.status is not StoryStatus.PUBLISHED


def test_refuses_an_approval_for_the_wrong_version():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    stale = [
        d.model_copy(update={"story_version": story.version + 1})
        for d in approvals_for(story, report)
    ]
    with pytest.raises(PublishRefused):
        DistributorAgent(ctx).render(story, [], comm, stale)


@pytest.mark.parametrize("name", ["", "   ", "explainer", "AI"])
def test_refuses_approvals_without_a_named_human(name):
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    approvals = [
        d.model_copy(update={"reviewer": name}) if d.role is ReviewRole.EDITOR_IN_CHIEF else d
        for d in approvals_for(story, report)
    ]
    with pytest.raises(PublishRefused):
        DistributorAgent(ctx).render(story, [], comm, approvals)


def test_refuses_when_a_required_expert_has_not_approved():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    assert ReviewRole.SUBJECT_EXPERT in report.required_reviews
    approvals = [d for d in approvals_for(story, report) if d.role is not ReviewRole.SUBJECT_EXPERT]
    with pytest.raises(PublishRefused, match="subject_expert"):
        DistributorAgent(ctx).render(story, [], comm, approvals)


def test_a_later_rejection_overrides_an_earlier_approval():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    approvals = approvals_for(story, report)
    approvals.append(
        _decision(story, ReviewRole.EDITOR_IN_CHIEF, "Anna Virtanen", decision=Decision.REJECT)
    )
    with pytest.raises(PublishRefused):
        DistributorAgent(ctx).render(story, [], comm, approvals)


def test_refuses_without_a_standards_review():
    ctx = make_ctx()
    story, claims, comm, _ = write_story(ctx)
    with pytest.raises(PublishRefused, match="standards review"):
        DistributorAgent(ctx).render(
            story, [], comm, [_decision(story, ReviewRole.EDITOR_IN_CHIEF)]
        )


def test_refuses_a_story_with_blocking_failures():
    ctx = make_ctx()
    story, claims, comm, _ = write_story(ctx)
    story.sources = []
    report = review(ctx, story, claims, comm)
    assert report.blocking_failures
    with pytest.raises(PublishRefused, match="blocking"):
        DistributorAgent(ctx).render(story, [], comm, approvals_for(story, report))


def test_publishes_with_full_approvals():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    approvals = approvals_for(story, report)

    publication = DistributorAgent(ctx).render(story, [], comm, approvals)

    assert ctx.store.get(Publication, publication.id) is not None
    assert story.status is StoryStatus.PUBLISHED
    assert ctx.store.get(Story, story.id).status is StoryStatus.PUBLISHED  # type: ignore[union-attr]
    assert set(publication.approval_ids) == {d.id for d in approvals}
    channels = {r.channel for r in publication.renditions}
    assert channels == {c.kind for c in comm.channels}
    assert all(r.language == "en" for r in publication.renditions)  # no unreviewed languages
    web = next(r for r in publication.renditions if r.channel == "web")
    assert "## Sources" in web.text and "AI" in web.text and "correction" in web.text


def test_lanes_without_human_approval_are_skipped_not_machine_published():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    translator = TranslatorAgent(ctx)
    uk = translator.translate(story, comm.lane("uk"), comm)  # type: ignore[arg-type]
    ar = translator.translate(story, comm.lane("ar"), comm)  # type: ignore[arg-type]
    report = review(ctx, story, [], comm, [uk, ar])

    publication = DistributorAgent(ctx).render(story, [uk, ar], comm, approvals_for(story, report))
    assert {r.language for r in publication.renditions} == {"en"}

    reviewed = uk.model_copy(update={"status": "human_reviewed", "backtranslation_flags": []})
    publication = DistributorAgent(ctx).render(
        story, [reviewed, ar], comm, approvals_for(story, report)
    )
    assert {r.language for r in publication.renditions} == {"en", "uk"}


def test_machine_draft_allowed_only_for_spot_check_low_risk_lanes():
    from newsroom.domain import RiskTier

    ctx = make_ctx()
    story, comm, report = _ready(ctx, "sapmi-north", title="Winter bus timetables", articles=None)
    story.risk_tier = RiskTier.LOW
    fi = Translation(
        id=new_id("tr"), story_id=story.id, story_version=story.version, language="fi",
        headline="Otsikko", standfirst="Ingressi", body_md="Teksti",
    )  # fmt: skip
    se = TranslatorAgent(ctx).translate(story, comm.lane("se"), comm)  # type: ignore[arg-type]
    agent = DistributorAgent(ctx)

    langs = agent._publishable_languages(story, [fi, se], comm, RiskTier.LOW)
    assert [code for code, _ in langs] == ["en", "fi"]  # Sámi never published without a human
    langs = agent._publishable_languages(story, [fi, se], comm, RiskTier.HIGH)
    assert [code for code, _ in langs] == ["en"]  # HIGH risk machine drafts are held back


def test_messaging_respects_max_chars():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    for limit in (700, 280, 160):
        comm_small = comm.model_copy(
            update={"channels": [Channel(kind="messaging", label="m", max_chars=limit)]}
        )
        publication = DistributorAgent(ctx).render(
            story.model_copy(update={"status": StoryStatus.APPROVED}),
            [],
            comm_small,
            approvals_for(story, report),
        )
        text = publication.renditions[0].text
        assert len(text) <= limit
        assert story.headline[: limit // 2].split(":")[0] in text or limit < 200


def test_audio_script_is_read_aloud_friendly():
    ctx = make_ctx()
    story, comm, report = _ready(ctx)
    comm_audio = comm.model_copy(update={"channels": [Channel(kind="audio", label="a")]})
    publication = DistributorAgent(ctx).render(story, [], comm_audio, approvals_for(story, report))
    text = publication.renditions[0].text

    assert "http" not in text and "#" not in text and "*" not in text
    assert "forty-five" in text  # "45 days" is spelled out for speech
    assert spell_int(2026) == "two thousand twenty-six"


def test_onboarding_pack_is_for_seasonal_communities_only():
    ctx = make_ctx()
    agent = DistributorAgent(ctx)
    story, comm, report = _ready(ctx, "lapland-seasonal", title="Winter housing")
    publication = agent.render(story, [], comm, approvals_for(story, report))

    pack = agent.onboarding_pack(comm, [publication], max_items=5)
    assert pack and all(r.channel == "messaging" for r in pack) and len(pack) <= 5
    assert agent.onboarding_pack(community("oulu-international"), [publication]) == []


def test_bridge_digest_in_finnish():
    ctx = make_ctx()
    agent = DistributorAgent(ctx)
    story, comm, report = _ready(ctx)
    publication = agent.render(story, [], comm, approvals_for(story, report))

    digest = agent.bridge_digest([comm], [publication])
    assert digest is not None and digest.channel == "digest" and digest.language == "fi"
    assert "yhteisöistä" in digest.text and story.headline in digest.text
    assert agent.bridge_digest([comm], []) is None
