"""The editor chooses where an approved story goes: the dashboard, the community page, or both."""

from __future__ import annotations

from newsroom.domain import Publication

EDITOR = {"role": "editor_in_chief", "reviewer": "Test editor-in-chief", "decision": "approve"}
COMMUNITY = "oulu-international"


def _publish(newsroom, destinations=None):
    report = newsroom.run_cycle([COMMUNITY], use_fixtures=True)
    run = next(r for r in report.runs if r.pending and r.pending["missing_roles"] == ["editor_in_chief"])
    decision = {**EDITOR, "destinations": destinations} if destinations else EDITOR
    return newsroom.submit_review(run.pitch_id, [decision])


def test_a_publication_defaults_to_both_places():
    pub = Publication(
        id="p", story_id="s", story_version=1, community_id="c", renditions=[], approval_ids=["a"]
    )

    assert pub.destinations == ["dashboard", "community"]


def test_without_a_choice_the_story_goes_to_both(newsroom):
    result = _publish(newsroom)

    assert newsroom.store.get(Publication, result.publication_id).destinations == [
        "dashboard",
        "community",
    ]
    assert len(newsroom.public_feed(COMMUNITY, "dashboard")) == 1
    assert len(newsroom.public_feed(COMMUNITY, "community")) == 1


def test_dashboard_only_stays_off_the_community_page(newsroom):
    _publish(newsroom, ["dashboard"])

    assert len(newsroom.public_feed(COMMUNITY, "dashboard")) == 1
    assert newsroom.public_feed(COMMUNITY, "community") == []


def test_community_only_stays_off_the_dashboard(newsroom):
    _publish(newsroom, ["community"])

    assert newsroom.public_feed(COMMUNITY, "dashboard") == []
    assert len(newsroom.public_feed(COMMUNITY, "community")) == 1
    assert len(newsroom.public_feed(COMMUNITY)) == 1  # no filter: everything published
