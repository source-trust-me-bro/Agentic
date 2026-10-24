"""Several communities picking the same event get one story, published to each of them."""

from __future__ import annotations

from tests.conftest import approve_everything

from newsroom.domain import Article, Pitch, Publication, Story, UserNeed

OWNER = "new-in-finland"  # priority 1: needs it most
OTHER = "oulu-international"  # priority 2


def _article(aid: str, source: str, official: bool, image: str | None) -> Article:
    return Article(
        id=aid,
        source_id=source.lower(),
        source_name=source,
        url=f"https://{source.lower()}.test/{aid}",
        title="Migri changes residence permit processing times",
        summary="The Finnish Immigration Service says permit decisions now take four weeks.",
        content="The Finnish Immigration Service said on Monday that residence permit "
        "decisions now take four weeks. Applicants can follow their case online.",
        language="en",
        official=official,
        event_id="evt-permits",
        image_url=image,
    )


def _pitch(community_id: str, article_ids: list[str]) -> Pitch:
    return Pitch(
        id=f"pitch-{community_id}",
        community_id=community_id,
        article_ids=article_ids,
        working_title="Residence permits now take four weeks",
        rationale="Affects everyone waiting for a permit.",
        needs=[UserNeed.USEFULNESS],
        score=0.8,
    )


def test_one_pitch_per_event_and_a_publication_for_every_community(newsroom):
    store = newsroom.store
    store.put(_article("a-migri", "Migri", True, None))
    store.put(_article("a-yle", "Yle", False, "https://yle.test/permit.jpg"))
    pitches = [_pitch(OTHER, ["a-yle"]), _pitch(OWNER, ["a-migri", "a-yle"])]
    for pitch in pitches:
        store.put(pitch)

    kept = newsroom._one_story_per_event(pitches)

    assert [p.community_id for p in kept] == [OWNER]
    assert kept[0].also_for == [OTHER]
    assert store.get(Pitch, f"pitch-{OTHER}") is None
    assert store.get(Pitch, f"pitch-{OWNER}").also_for == [OTHER]

    run = approve_everything(newsroom, newsroom._run_story(kept[0]))
    assert run.status == "published", run.pending

    story = store.get(Story, run.story_id)
    assert story.image_url == "https://yle.test/permit.jpg"
    assert story.event_ids == ["evt-permits"]
    assert story.body_md.startswith("## What happened")
    assert "## What it means for you" in story.body_md

    publications = store.find(Publication, story_id=story.id)
    assert sorted(p.community_id for p in publications) == sorted([OWNER, OTHER])
    assert newsroom.feed(OTHER) and newsroom.feed(OWNER)

    exported = [s for s in newsroom.dashboard_export()["stories"] if s["story_id"] == story.id]
    assert len(exported) == 1  # once, not once per community
    assert exported[0]["image_url"] == "https://yle.test/permit.jpg"
    assert exported[0]["event_ids"] == ["evt-permits"]
    assert set(exported[0]["community_ids"]) == {OWNER, OTHER}
