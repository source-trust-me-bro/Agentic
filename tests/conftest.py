from __future__ import annotations

"""Shared fixtures. Tests are hermetic: they never read a developer's local .env."""

import os

# Environment variables beat the .env file in pydantic-settings, so these pin the test
# configuration before any settings object is created.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["GEMINI_API_KEY"] = ""
os.environ["MONGODB_URI"] = ""
os.environ["API_KEY"] = ""
os.environ["HF_TOKEN"] = ""
os.environ["ARCHITECT_USES_MODEL"] = "true"  # tests exercise the model path explicitly
# The shipped defaults are relaxed (hackathon); the editorial rules are tested in strict mode.
os.environ["STRICT_STANDARDS"] = "true"
os.environ["POLICY_FILE"] = os.path.join(os.path.dirname(__file__), "fixtures", "policy_strict.yaml")
# A local .env tuned for speed (fewer rounds, pitches and calls) must not change what is tested.
os.environ["MAX_INVESTIGATION_ROUNDS"] = "3"
os.environ["MAX_REVISION_ROUNDS"] = "2"
os.environ["MAX_PITCHES_PER_CYCLE"] = "2"
os.environ["MAX_CONCURRENT_MODEL_CALLS"] = "8"
os.environ["CYCLE_MAX_MODEL_CALLS"] = "600"
# Test newsrooms have no members yet; every community language is translated.
os.environ["TRANSLATE_ONLY_READ_LANGUAGES"] = "false"


import pytest

from newsroom.config import Settings
from newsroom.domain import Community, ReviewRole
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.service import Newsroom, StoryRun
from newsroom.store.memory import MemoryStore


@pytest.fixture(autouse=True)
def _fresh_robots_cache():
    """robots.txt is cached per host; every test starts without what another test fetched."""

    from newsroom.sources import rss

    rss._ROBOTS.clear()
    yield
    rss._ROBOTS.clear()


@pytest.fixture
def newsroom() -> Newsroom:
    settings = Settings(llm_provider="fake", mongodb_uri="")
    return Newsroom(
        settings=settings,
        store=MemoryStore(),
        llm=FakeLLM(collect_fake_handlers()),
        persistent_checkpoints=False,
    )


def reviewer_for(community: Community, role: ReviewRole, languages: set[str]) -> str:
    """A named human who is allowed to act in this role for this community."""

    if role is ReviewRole.EDITOR_IN_CHIEF:
        return "Test editor-in-chief"
    for voice in community.trusted_voices:
        if role not in voice.can_review:
            continue
        if (
            role is ReviewRole.LANGUAGE_REVIEWER
            and languages
            and not languages & set(voice.languages)
        ):
            continue
        return voice.id
    return ""


OFFICIAL_SOURCE = (
    "Occupational Safety and Health Administration (official)",
    "https://example.test/official",
)


def approve_everything(newsroom: Newsroom, run: StoryRun, *, rounds: int = 5) -> StoryRun:
    community = newsroom.communities[run.community_id]
    for _ in range(rounds):
        if not run.pending:
            break
        if any("official source" in c.lower() for c in run.pending.get("blocking_checks", [])):
            # What a real editor does: attach the official source the agents could not find.
            newsroom.add_story_source(
                run.pending["story_id"],
                name=OFFICIAL_SOURCE[0],
                url=OFFICIAL_SOURCE[1],
                official=True,
                reviewer="Test editor-in-chief",
            )
        roles = [ReviewRole(r) for r in run.pending["missing_roles"]]
        languages = set(run.pending["translations"])
        decisions = [
            {
                "role": role.value,
                "reviewer": reviewer_for(community, role, languages),
                "decision": "approve",
                "notes": "ok",
            }
            for role in roles
        ]
        if not decisions or any(not d["reviewer"] for d in decisions):
            break
        run = newsroom.submit_review(run.pitch_id, decisions)
    return run
