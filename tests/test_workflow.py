"""End-to-end behaviour of the story workflow and the human gate."""

from __future__ import annotations

import pytest
from tests.conftest import approve_everything

from newsroom.domain import (
    CommunityMessage,
    InvestigationResult,
    Pitch,
    Publication,
    ReviewDecision,
    Story,
    Translation,
    new_id,
)
from newsroom.hitl import ReviewError

COMMUNITY = "lapland-seasonal"


def _cycle(newsroom, community=COMMUNITY):
    return newsroom.run_cycle([community], use_fixtures=True)


def test_cycle_stops_at_the_human_gate_and_publishes_nothing(newsroom):
    report = _cycle(newsroom)

    assert report.pitches >= 1
    assert any(run.status == "awaiting_review" for run in report.runs)
    assert newsroom.feed(COMMUNITY) == []
    assert newsroom.store.find(Publication) == []


def test_editor_in_chief_alone_is_not_enough_for_a_high_risk_story(newsroom):
    run = next(r for r in _cycle(newsroom).runs if r.pending)
    missing = run.pending["missing_roles"]
    assert "editor_in_chief" in missing
    if len(missing) < 2:
        pytest.skip("this story needs only the editor-in-chief")

    run = newsroom.submit_review(
        run.pitch_id,
        [{"role": "editor_in_chief", "reviewer": "Test editor", "decision": "approve"}],
    )

    assert run.status == "awaiting_review"
    assert "editor_in_chief" not in run.pending["missing_roles"]
    assert newsroom.feed(COMMUNITY) == []


def test_full_human_approval_publishes_with_named_approvers(newsroom):
    runs = _cycle(newsroom).runs
    finished = [approve_everything(newsroom, run) for run in runs if run.pending]

    published = [r for r in finished if r.status == "published"]
    assert published, [r.status for r in finished]
    for run in published:
        publication = newsroom.store.get(Publication, run.publication_id)
        assert publication is not None and publication.approval_ids
        approvals = [newsroom.store.get(ReviewDecision, i) for i in publication.approval_ids]
        assert all(a and a.reviewer.strip() for a in approvals)
        assert any(a.role.value == "editor_in_chief" for a in approvals)


def test_untrusted_reviewers_are_refused(newsroom):
    run = next(r for r in _cycle(newsroom).runs if r.pending)

    for bad in ("", "   ", "AI", "agent"):
        with pytest.raises(ReviewError):
            newsroom.submit_review(
                run.pitch_id,
                [{"role": "editor_in_chief", "reviewer": bad, "decision": "approve"}],
            )
    with pytest.raises(ReviewError):  # not one of the community's trusted voices
        newsroom.submit_review(
            run.pitch_id,
            [{"role": "subject_expert", "reviewer": "Some stranger", "decision": "approve"}],
        )
    assert newsroom.feed(COMMUNITY) == []


def test_rejection_ends_the_story_without_publishing(newsroom):
    run = next(r for r in _cycle(newsroom).runs if r.pending)

    result = newsroom.submit_review(
        run.pitch_id,
        [
            {
                "role": "editor_in_chief",
                "reviewer": "Test editor",
                "decision": "reject",
                "notes": "no",
            }
        ],
    )

    assert result.status == "rejected"
    assert newsroom.feed(COMMUNITY) == []


def test_changes_requested_creates_a_new_version_that_needs_fresh_approval(newsroom):
    run = next(r for r in _cycle(newsroom).runs if r.pending)
    first = newsroom.store.get(Story, run.story_id)

    run = newsroom.submit_review(
        run.pitch_id,
        [
            {
                "role": "editor_in_chief",
                "reviewer": "Test editor",
                "decision": "request_changes",
                "notes": "Explain the housing deposit rule more clearly.",
            }
        ],
    )

    assert run.status == "awaiting_review"
    second = newsroom.store.get(Story, run.story_id)
    assert second.version == first.version + 1
    assert "editor_in_chief" in run.pending["missing_roles"]  # v1 approval does not carry over
    assert newsroom.feed(COMMUNITY) == []


def test_prompt_injection_article_never_becomes_a_pitch(newsroom):
    report = _cycle(newsroom)

    assert report.flagged_articles, "the injection fixture should have been flagged"
    flagged = set(report.flagged_articles)
    for pitch in newsroom.store.find(Pitch):
        assert not flagged & set(pitch.article_ids)


def test_sami_lane_is_never_machine_published(newsroom):
    report = newsroom.run_cycle(["sapmi-north"], use_fixtures=True)
    runs = [approve_everything(newsroom, run) for run in report.runs if run.pending]

    for translation in newsroom.store.find(Translation):
        if translation.language == "se":
            assert translation.status == "needs_human_translation"
            assert translation.body_md == ""
    for publication in newsroom.store.find(Publication):
        assert all(r.language != "se" for r in publication.renditions)
    assert runs is not None


def test_investigation_loop_is_bounded(newsroom):
    calls: list[int] = []

    def never_sufficient(pitch, *, round_no=1, prior=None):
        calls.append(round_no)
        return InvestigationResult(
            claims=[],
            sufficient=False,
            open_questions=["unknown"],
            search_hints=["x"],
            round_no=round_no,
        )

    newsroom.workflow.investigator.investigate = never_sufficient  # type: ignore[method-assign]
    _cycle(newsroom)

    limit = newsroom.settings.max_investigation_rounds
    assert calls and max(calls) <= limit
    traces = newsroom.store.traces()
    assert any(t.step == "investigation" and t.decision == "stop" for t in traces)


def test_community_tip_becomes_a_lead_not_a_fact(newsroom):
    message = CommunityMessage(
        id=new_id("msg"),
        community_id=COMMUNITY,
        kind="tip",
        text="Staff housing near the Levi gondola has no heating this week.",
    )

    triage = newsroom.community_message(message)

    assert triage.action in {"open_pitch", "needs_human"}
    if triage.pitch is not None:
        assert triage.pitch.origin == "community_tip"
        assert triage.pitch.article_ids == []
        assert "verif" in triage.pitch.rationale.lower()


def test_audit_trail_records_decisions_with_reasons(newsroom):
    report = _cycle(newsroom)
    traces = newsroom.store.traces(report.run_id)

    assert traces
    assert all(t.agent and t.step and t.decision for t in traces)
    assert any(t.agent == "orchestrator" for t in traces)
