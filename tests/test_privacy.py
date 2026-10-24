"""The privacy promise shown to community members, tested against what really happens."""

from __future__ import annotations

from newsroom import guards
from newsroom.domain import CommunityMessage, new_id

SECRET_EMAIL = "aino.virtanen@example.com"
SECRET_PHONE = "+358 40 123 4567"


def _message(newsroom, kind: str, text: str):
    return newsroom.community_message(
        CommunityMessage(
            id=new_id("msg"),
            community_id="lapland-seasonal",
            kind=kind,
            text=text,
            author_ref="member-1",
        )
    )


def test_a_message_with_contact_details_goes_to_a_human_and_never_to_the_model(newsroom):
    for kind in ("question", "tip", "local_knowledge", "correction"):
        triage = _message(
            newsroom,
            kind,
            f"My landlord in Levi keeps my deposit. Email {SECRET_EMAIL} or call {SECRET_PHONE}.",
        )
        assert triage.action == "needs_human"

    assert newsroom.ctx.llm.calls == []  # not a single model call carried the details


def test_an_ordinary_message_reaches_the_model_and_carries_no_contact_details(newsroom):
    triage = _message(newsroom, "tip", "Staff housing near the gondola has no heating this week.")

    assert triage.action in {"open_pitch", "needs_human"}
    assert newsroom.ctx.llm.calls, "an ordinary message should be read by the model"
    sent = str(newsroom.ctx.llm.calls)
    assert SECRET_EMAIL not in sent and "123 4567" not in sent


def test_masking_is_a_second_safeguard_for_anything_that_slips_through():
    masked = guards.minimise_personal_data(f"Write to {SECRET_EMAIL} or ring {SECRET_PHONE}.")

    assert SECRET_EMAIL not in masked
    assert "123 4567" not in masked


def test_the_member_reference_never_contains_a_name_or_email(newsroom):
    member = newsroom.join_community("hashed-ref-123", "lapland-seasonal")

    assert member.id == "hashed-ref-123"
    assert "@" not in member.model_dump_json()
