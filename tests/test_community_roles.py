"""Community admins and moderators are the community's trusted voices."""

from __future__ import annotations

import pytest

from newsroom.domain import ReviewRole, Translation
from newsroom.hitl import ReviewError

COMMUNITY = "lapland-seasonal"


def appoint(newsroom, ref, *, role="moderator", roles=(), languages=(), name="A Moderator"):
    return newsroom.set_community_role(
        ref,
        COMMUNITY,
        community_role=role,
        review_roles=[r.value if hasattr(r, "value") else r for r in roles],
        review_languages=list(languages),
        display_name=name,
    )


def waiting_run(newsroom):
    report = newsroom.run_cycle([COMMUNITY], use_fixtures=True)
    return next(r for r in report.runs if r.pending)


def test_a_plain_member_cannot_review(newsroom):
    run = waiting_run(newsroom)
    newsroom.join_community("member-plain", COMMUNITY)

    with pytest.raises(ReviewError):
        newsroom.submit_review(
            run.pitch_id,
            [
                {
                    "role": "community_reviewer",
                    "reviewer": "member:member-plain",
                    "decision": "approve",
                }
            ],
        )


def test_a_moderator_reviews_only_in_the_roles_they_were_given(newsroom):
    run = waiting_run(newsroom)
    appoint(newsroom, "mod-1", roles=[ReviewRole.COMMUNITY_REVIEWER])

    with pytest.raises(ReviewError):  # was not given subject_expert
        newsroom.submit_review(
            run.pitch_id,
            [{"role": "subject_expert", "reviewer": "member:mod-1", "decision": "approve"}],
        )
    result = newsroom.submit_review(
        run.pitch_id,
        [{"role": "community_reviewer", "reviewer": "member:mod-1", "decision": "approve"}],
    )

    assert "community_reviewer" not in result.pending["missing_roles"]


def test_a_moderator_of_another_community_cannot_review_here(newsroom):
    run = waiting_run(newsroom)
    newsroom.set_community_role(
        "mod-elsewhere",
        "oulu-international",
        community_role="moderator",
        review_roles=["community_reviewer"],
        review_languages=[],
        display_name="Elsewhere",
    )

    with pytest.raises(ReviewError):
        newsroom.submit_review(
            run.pitch_id,
            [
                {
                    "role": "community_reviewer",
                    "reviewer": "member:mod-elsewhere",
                    "decision": "approve",
                }
            ],
        )


def test_a_language_reviewer_needs_a_language_and_approves_only_that_language(newsroom):
    run = waiting_run(newsroom)
    appoint(newsroom, "mod-nolang", roles=[ReviewRole.LANGUAGE_REVIEWER])
    with pytest.raises(ReviewError):
        newsroom.submit_review(
            run.pitch_id,
            [{"role": "language_reviewer", "reviewer": "member:mod-nolang", "decision": "approve"}],
        )

    appoint(
        newsroom, "mod-uk", roles=[ReviewRole.LANGUAGE_REVIEWER], languages=["xx"]
    )  # not a lane
    member = newsroom.member("mod-uk")
    assert member.review_languages == []  # languages the community does not offer are dropped

    appoint(newsroom, "mod-ru", roles=[ReviewRole.LANGUAGE_REVIEWER], languages=["ru"])
    newsroom.submit_review(
        run.pitch_id,
        [{"role": "language_reviewer", "reviewer": "member:mod-ru", "decision": "approve"}],
    )
    translations = newsroom.store.find(Translation, story_id=run.story_id)
    assert translations and all(
        t.status == "human_reviewed" for t in translations if t.language == "ru"
    )


def test_a_moderator_cannot_be_editor_in_chief_and_staff_cannot_use_member_refs(newsroom):
    with pytest.raises(ValueError):
        appoint(newsroom, "mod-x", roles=[ReviewRole.EDITOR_IN_CHIEF])

    run = waiting_run(newsroom)
    appoint(newsroom, "mod-y", roles=[ReviewRole.COMMUNITY_REVIEWER])
    with pytest.raises(ReviewError):
        newsroom.submit_review(
            run.pitch_id,
            [{"role": "editor_in_chief", "reviewer": "member:mod-y", "decision": "approve"}],
        )


def test_demoting_a_moderator_removes_their_review_rights(newsroom):
    run = waiting_run(newsroom)
    appoint(newsroom, "mod-z", roles=[ReviewRole.COMMUNITY_REVIEWER])
    appoint(newsroom, "mod-z", role="member")

    with pytest.raises(ReviewError):
        newsroom.submit_review(
            run.pitch_id,
            [{"role": "community_reviewer", "reviewer": "member:mod-z", "decision": "approve"}],
        )


def test_story_publishes_with_moderators_as_trusted_voices_and_their_names_show(newsroom):
    run = waiting_run(newsroom)
    appoint(
        newsroom,
        "mod-all",
        roles=[ReviewRole.COMMUNITY_REVIEWER, ReviewRole.SUBJECT_EXPERT],
        name="Aino Moderator",
    )
    appoint(
        newsroom,
        "mod-ru",
        roles=[ReviewRole.LANGUAGE_REVIEWER],
        languages=["ru"],
        name="Ivan Reviewer",
    )

    for _ in range(4):
        if not run.pending:
            break
        decisions = []
        for role in run.pending["missing_roles"]:
            reviewer = "Test editor-in-chief" if role == "editor_in_chief" else "member:mod-all"
            decisions.append({"role": role, "reviewer": reviewer, "decision": "approve"})
        run = newsroom.submit_review(run.pitch_id, decisions)

    assert run.status == "published", run.status
    feed = newsroom.public_feed(COMMUNITY)
    assert {r["language"] for item in feed for r in item["renditions"]} == {"en"}  # Russian waits

    # The Russian-language reviewer, a community moderator, approves their language later.
    queue = [i for i in newsroom.review_queue() if i.get("type") == "language_review"]
    assert queue and "ru" in queue[0]["translations"]
    done = newsroom.submit_review(
        run.pitch_id,
        [{"role": "language_reviewer", "reviewer": "member:mod-ru", "decision": "approve"}],
    )
    assert done.status == "published"

    feed = newsroom.public_feed(COMMUNITY)
    assert len(feed) == 1  # the story is republished, not duplicated
    assert {r["language"] for item in feed for r in item["renditions"]} >= {"en", "ru"}
    names = {a["name"] for item in feed for a in item["approvers"]}
    assert {"Aino Moderator", "Ivan Reviewer"} <= names
    assert "community moderator" in {a["title"] for item in feed for a in item["approvers"]}
