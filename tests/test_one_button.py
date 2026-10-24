"""The editor presses Publish. The agents do the rest. Humans still decide."""

from __future__ import annotations

import pytest

from newsroom.domain import Publication, Story, StoryStatus
from newsroom.hitl import ReviewError

EDITOR = {"role": "editor_in_chief", "reviewer": "Test editor-in-chief", "decision": "approve"}


def _publish_only_with_editor(newsroom, community):
    report = newsroom.run_cycle([community], use_fixtures=True)
    runs = [
        r for r in report.runs if r.pending and r.pending["missing_roles"] == ["editor_in_chief"]
    ]
    return report, runs


def test_an_ordinary_story_publishes_with_one_editor_click(newsroom):
    report, runs = _publish_only_with_editor(newsroom, "oulu-international")

    assert runs, "a low-risk story should need only the editor-in-chief"
    result = newsroom.submit_review(runs[0].pitch_id, [EDITOR])

    assert result.status == "published"
    publication = newsroom.store.get(Publication, result.publication_id)
    assert {r.language for r in publication.renditions} == {"en"}  # other languages wait for review
    assert newsroom.store.get(Story, result.story_id).status is StoryStatus.PUBLISHED


def test_the_agents_never_approve_for_the_editor(newsroom):
    _report, runs = _publish_only_with_editor(newsroom, "oulu-international")

    for bad in ("agent", "bot", "AI", ""):
        with pytest.raises(ReviewError):
            newsroom.submit_review(runs[0].pitch_id, [{**EDITOR, "reviewer": bad}])
    assert newsroom.feed("oulu-international") == []


def test_a_high_risk_story_still_needs_its_experts_by_default(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    run = next(r for r in report.runs if r.pending)

    assert run.pending["risk_tier"] == "high"
    assert set(run.pending["missing_roles"]) == {
        "editor_in_chief",
        "subject_expert",
        "community_reviewer",
    }
    assert "language_reviewer" not in run.pending["missing_roles"]  # never blocks the story


def test_the_high_risk_roles_can_be_relaxed_to_editor_only(newsroom):
    policy = newsroom.workflow.standards.policy.model_copy(deep=True)  # never mutate the shared one
    policy.review.high_risk_roles = []
    newsroom.workflow.standards.policy = policy

    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    run = next(r for r in report.runs if r.pending)

    assert run.pending["risk_tier"] == "high"
    assert run.pending["missing_roles"] == ["editor_in_chief"]
    result = newsroom.submit_review(run.pitch_id, [EDITOR])
    assert result.status == "published"  # one click, a human still decided


def test_a_published_story_cannot_be_changed_by_a_late_non_language_review(newsroom):
    _report, runs = _publish_only_with_editor(newsroom, "oulu-international")
    done = newsroom.submit_review(runs[0].pitch_id, [EDITOR])
    assert done.status == "published"

    with pytest.raises(ReviewError):
        newsroom.submit_review(
            runs[0].pitch_id,
            [{"role": "editor_in_chief", "reviewer": "Another editor", "decision": "reject"}],
        )
    assert len(newsroom.feed("oulu-international")) == 1


def test_the_scheduler_does_not_run_on_the_offline_model(newsroom):
    import threading

    from fastapi.testclient import TestClient

    from newsroom.api.app import create_app
    from newsroom.config import get_settings

    settings = get_settings()
    original = settings.cycle_interval_minutes
    settings.cycle_interval_minutes = 1
    try:
        with TestClient(create_app(newsroom)):
            names = {t.name for t in threading.enumerate()}
            assert "newsroom-scheduler" not in names  # offline stand-in never runs on live feeds
    finally:
        settings.cycle_interval_minutes = original
