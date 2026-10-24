"""The newsroom attaches the basic official reference itself; editors review and add more."""

from __future__ import annotations

import pytest
from tests.conftest import approve_everything

from newsroom.agents.base import AgentContext
from newsroom.agents.librarian import SourceLibrarianAgent
from newsroom.domain import ContextNote, SourceRef, Story, StoryStatus, new_id
from newsroom.hitl import ReviewError
from newsroom.llm import FakeLLM
from newsroom.officials import load_official_sources, match_official_sources
from newsroom.registry import load_communities
from newsroom.store.memory import MemoryStore


def _story(headline: str, body: str, community_id: str = "oulu-international") -> Story:
    return Story(
        id=new_id("story"),
        pitch_id=new_id("pitch"),
        community_id=community_id,
        headline=headline,
        standfirst="",
        body_md=body,
        needs=[],
        context_notes=[ContextNote(text="x", kind="from_story")],
    )


def _librarian() -> tuple[SourceLibrarianAgent, MemoryStore]:
    store = MemoryStore()
    return SourceLibrarianAgent(AgentContext(llm=FakeLLM(), store=store)), store


def _community(community_id: str):
    return next(c for c in load_communities() if c.id == community_id)


def test_permit_story_gets_migri_as_an_official_reference():
    librarian, store = _librarian()
    story = _story(
        "Residence permit processing times grow", "Migri says a residence permit now takes 45 days."
    )

    librarian.attach(story, _community("oulu-international"))

    names = [s.name for s in story.sources]
    assert any("Migri" in n for n in names)
    for ref in story.sources:
        assert ref.official and ref.origin == "agent_reference"
        assert "does not verify" in ref.note  # honest: a reference, not proof of a claim


def test_attaching_is_idempotent_and_never_duplicates():
    librarian, _ = _librarian()
    story = _story("Residence permit news", "A residence permit change.")
    community = _community("oulu-international")

    librarian.attach(story, community)
    count = len(story.sources)
    librarian.attach(story, community)

    assert len(story.sources) == count


def test_a_story_with_an_existing_link_is_not_given_the_same_one_twice():
    librarian, _ = _librarian()
    story = _story("Residence permit news", "A residence permit change.")
    story.sources.append(SourceRef(name="Migri", url="https://migri.fi/en/", official=True))

    librarian.attach(story, _community("oulu-international"))

    assert sum(1 for s in story.sources if s.url.rstrip("/") == "https://migri.fi/en") == 1


def test_unrelated_story_gets_no_official_reference():
    librarian, _ = _librarian()
    story = _story("Local hockey team wins", "Kiekko-Kaleva won 3-2 in overtime.")

    librarian.attach(story, _community("oulu-international"))

    assert story.sources == []


def test_region_bound_bodies_only_match_their_region():
    registry = load_official_sources()

    oulu = {s.id for s in match_official_sources(["housing"], ["Oulu"], registry=registry)}
    lapland = {
        s.id for s in match_official_sources(["housing"], ["Levi", "Lapland"], registry=registry)
    }

    assert "city-oulu" in oulu
    assert "city-oulu" not in lapland


def test_every_official_source_is_a_plain_homepage_link():
    for source in load_official_sources():
        assert source.url.startswith("https://") and " " not in source.url
        assert source.topics


def test_workflow_attaches_official_references_without_any_human_help(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    run = next(r for r in report.runs if r.pending)

    story = newsroom.store.get(Story, run.story_id)

    assert any(s.origin == "agent_reference" and s.official for s in story.sources)
    # The basic official source is already there, so it is not what blocks the story.
    assert not any("official source" in c.lower() for c in run.pending["blocking_checks"])
    attach_traces = [t for t in newsroom.store.traces() if t.step == "attach_official_sources"]
    assert attach_traces


def test_editor_can_check_and_add_more_sources_and_origin_is_recorded(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    run = next(r for r in report.runs if r.pending)
    before = newsroom.store.get(Story, run.story_id)
    assert any(s.origin == "agent_reference" for s in before.sources)

    newsroom.add_story_source(
        run.story_id,
        name="Hospitality union collective agreement",
        url="https://example.test/agreement",
        official=False,
        reviewer="Test editor-in-chief",
    )

    after = newsroom.store.get(Story, run.story_id)
    added = [s for s in after.sources if s.origin == "editor"]
    assert [s.name for s in added] == ["Hospitality union collective agreement"]
    assert len(after.sources) == len(before.sources) + 1  # agent references are kept


def test_editor_source_input_is_validated(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    run = next(r for r in report.runs if r.pending)

    with pytest.raises(ReviewError):
        newsroom.add_story_source(
            run.story_id, name="x", url="javascript:alert(1)", official=False, reviewer="Editor"
        )
    with pytest.raises(ReviewError):
        newsroom.add_story_source(
            run.story_id, name="x", url="https://example.test", official=False, reviewer="agent"
        )


def test_story_publishes_after_human_approval_with_agent_attached_sources(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)
    finished = [approve_everything(newsroom, r) for r in report.runs if r.pending]

    assert any(r.status == "published" for r in finished), [r.status for r in finished]
    published = [s for s in newsroom.store.find(Story) if s.status is StoryStatus.PUBLISHED]
    assert published
    for story in published:
        assert any(s.official for s in story.sources)
