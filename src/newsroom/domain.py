"""Domain model shared by every agent, the store, the graph and the API.

Design rules that the rest of the codebase relies on:

* Everything durable is a ``Doc`` with a stable string ``id`` and a ``COLLECTION``.
* Agents *propose*; only the human gate and the publisher turn a proposal into a
  published fact. ``Publication`` therefore always carries the approvals behind it.
* Scraped text is data. It is never trusted to carry instructions (see ``guards``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Doc(BaseModel):
    """Base for anything persisted. Subclasses set ``COLLECTION``."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    COLLECTION: ClassVar[str] = ""

    id: str


# --------------------------------------------------------------------------- enums


class UserNeed(StrEnum):
    """Kaleva Media's user needs model: every story must serve at least one."""

    INFORMATION = "information"  # what happened
    UNDERSTANDING = "understanding"  # why, and what it means
    USEFULNESS = "usefulness"  # how it helps me in everyday life
    EMOTION = "emotion"  # moves or inspires
    BREAK = "break"  # entertains and offers a breather
    COMMUNITY = "community"  # connects people, gives the community a voice


class RiskTier(StrEnum):
    """How much harm an error could do. Drives which human reviews are mandatory."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"  # residence permits, health, employment, legal rights, safety


class ClaimStatus(StrEnum):
    VERIFIED = "verified"  # corroborated by >= 2 independent sources
    ATTRIBUTED = "attributed"  # one source asserts it; attribution kept
    DISPUTED = "disputed"  # sources contradict each other; both kept
    UNKNOWN = "unknown"  # not enough evidence either way


class ReviewRole(StrEnum):
    EDITOR_IN_CHIEF = "editor_in_chief"  # always required before publication
    LANGUAGE_REVIEWER = "language_reviewer"  # required for small / high-risk languages
    COMMUNITY_REVIEWER = "community_reviewer"  # trusted voice from the community
    SUBJECT_EXPERT = "subject_expert"  # e.g. migration adviser for permit stories


class Decision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class StoryStatus(StrEnum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    CORRECTED = "corrected"


# ------------------------------------------------------------------ sources & articles


class SourceConfig(BaseModel):
    id: str
    name: str
    feed_url: str | None = None
    language: str = "fi"
    official: bool = False  # authority publishing its own decisions (city, Migri, Kela)
    region: str | None = None
    # Republishing translated text of a source needs the right to do so. The
    # pipeline records the answer instead of assuming it.
    reuse: Literal["link_and_summarize", "licensed", "official_public_info"] = "link_and_summarize"
    # "metadata_only": the publisher's full text is behind a paywall. We use only what the
    # feed publishes (headline, summary, link) and never try to get around the paywall.
    # Switch to "full" only when the publisher grants licensed full-text access.
    access: Literal["full", "metadata_only"] = "full"
    enabled: bool = True
    notes: str | None = None


class Article(Doc):
    COLLECTION: ClassVar[str] = "articles"

    source_id: str
    source_name: str
    url: str
    title: str
    summary: str | None = None
    content: str | None = None
    language: str = "fi"
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    official: bool = False
    # Syndicated copies share a key so they are not counted as independent evidence.
    syndication_key: str | None = None
    content_hash: str | None = None
    injection_flags: list[str] = Field(default_factory=list)
    # metadata_only articles contribute headline/summary only (paywalled publishers).
    access: Literal["full", "metadata_only"] = "full"
    # The canonical event this report belongs to (set by the knowledge architect).
    event_id: str | None = None


# ------------------------------------------------------------------------- communities


class LanguageLane(BaseModel):
    """One output language for a community, and how much human review it needs."""

    code: str  # en, uk, ru, ar, so, sv, se (North Sámi), smn (Inari), sms (Skolt)
    label: str
    # A = strong machine translation; B = usable but review high-risk items;
    # C = weak (e.g. Sámi languages): the machine only assists a human translator.
    tier: Literal["A", "B", "C"] = "A"
    human_review: Literal["spot_check", "required", "human_only"] = "spot_check"


class TrustedVoice(BaseModel):
    id: str
    name: str
    role: str  # e.g. "student union buddy", "resort HR officer", "Sámi language teacher"
    can_review: list[ReviewRole] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


class Channel(BaseModel):
    kind: Literal["web", "messaging", "audio", "digest"]
    label: str
    max_chars: int | None = None


class Lifecycle(BaseModel):
    seasonal: bool = False
    typical_stay_days: int | None = None
    onboarding_window_days: int = 7  # the first week is when information matters most


class Community(Doc):
    COLLECTION: ClassVar[str] = "communities"

    name: str
    description: str
    # Lower is more important. Orders community lists and breaks ties in recommendations.
    priority: int = 100
    regions: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    languages: list[LanguageLane]
    needs_weights: dict[UserNeed, float] = Field(default_factory=dict)
    trusted_voices: list[TrustedVoice] = Field(default_factory=list)
    channels: list[Channel] = Field(default_factory=list)
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)
    # Short cultural-context notes editors maintain, e.g. "tipping is not expected".
    cultural_context: list[str] = Field(default_factory=list)

    def lane(self, code: str) -> LanguageLane | None:
        return next((lane for lane in self.languages if lane.code == code), None)


# --------------------------------------------------------------- pitches and evidence


class Pitch(Doc):
    """A story idea proposed by the news editor for one community."""

    COLLECTION: ClassVar[str] = "pitches"

    community_id: str
    article_ids: list[str]
    working_title: str
    rationale: str
    needs: list[UserNeed]
    score: float = 0.0
    risk_tier: RiskTier = RiskTier.LOW
    origin: Literal["monitor", "community_tip", "editor"] = "monitor"
    # Where the editor chose to publish it. Empty means both places.
    publish_to: list[Literal["dashboard", "community"]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class EvidenceLink(BaseModel):
    article_id: str
    url: str
    source_name: str
    relation: Literal["supports", "contradicts"]
    independent: bool = True  # False for syndicated copies of the same report
    quote: str | None = None


class Claim(Doc):
    COLLECTION: ClassVar[str] = "claims"

    pitch_id: str
    text: str
    status: ClaimStatus = ClaimStatus.UNKNOWN
    attribution: str | None = None  # who says it ("the City of Oulu said...")
    source_article_id: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    # Deliberately not a probability. See the trust section of the README.
    uncertainty_note: str | None = None

    @property
    def independent_support(self) -> int:
        return len(
            {e.source_name for e in self.evidence if e.relation == "supports" and e.independent}
        )


class InvestigationResult(BaseModel):
    claims: list[Claim]
    sufficient: bool
    open_questions: list[str] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    round_no: int = 1


# ---------------------------------------------------------------- stories and review


class ContextNote(BaseModel):
    """Explanatory context that does not come from the original story."""

    text: str
    kind: Literal["from_story", "official_source", "community_knowledge", "human_needed"]
    source_url: str | None = None


class SourceRef(BaseModel):
    name: str
    url: str
    official: bool = False
    # Where the source came from: the reporting itself, a curated official reference the
    # newsroom attached automatically, or an editor. Editors can always see and change this.
    origin: Literal["report", "agent_reference", "editor"] = "report"
    note: str = ""


class Story(Doc):
    COLLECTION: ClassVar[str] = "stories"

    pitch_id: str
    community_id: str
    version: int = 1
    language: str = "en"  # language the draft was written in
    headline: str
    standfirst: str
    body_md: str
    needs: list[UserNeed]
    risk_tier: RiskTier = RiskTier.LOW
    claim_ids: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    context_notes: list[ContextNote] = Field(default_factory=list)
    ai_disclosure: str = ""
    status: StoryStatus = StoryStatus.DRAFT
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Translation(Doc):
    COLLECTION: ClassVar[str] = "translations"

    story_id: str
    story_version: int
    language: str
    headline: str = ""
    standfirst: str = ""
    body_md: str = ""
    status: Literal["machine_draft", "needs_human_translation", "human_reviewed", "rejected"] = (
        "machine_draft"
    )
    # Terms that must not be translated freely (permit names, authorities).
    protected_terms: list[str] = Field(default_factory=list)
    # Back-translation check for risky text: differences a human should look at.
    backtranslation_flags: list[str] = Field(default_factory=list)
    machine_translated: bool = True


class Check(BaseModel):
    guideline: str  # JSN guideline number, e.g. "7", "14", "35", or "AI" / "RISK"
    name: str
    passed: bool
    note: str = ""
    blocking: bool = False


class StandardsReport(Doc):
    COLLECTION: ClassVar[str] = "standards_reports"

    story_id: str
    story_version: int
    checks: list[Check]
    risk_tier: RiskTier
    required_reviews: list[ReviewRole]
    revision_feedback: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def blocking_failures(self) -> list[Check]:
        return [c for c in self.checks if c.blocking and not c.passed]


class ReviewDecision(Doc):
    COLLECTION: ClassVar[str] = "reviews"

    story_id: str
    story_version: int
    role: ReviewRole
    reviewer: str  # a named human, never an agent
    decision: Decision
    notes: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


class Rendition(BaseModel):
    """One channel-specific output of an approved story."""

    channel: Literal["web", "messaging", "audio", "digest"]
    language: str
    text: str
    link: str | None = None


class Publication(Doc):
    COLLECTION: ClassVar[str] = "publications"

    story_id: str
    story_version: int
    community_id: str
    renditions: list[Rendition]
    approval_ids: list[str]  # ReviewDecision ids; a publication without them is invalid
    # The editor chooses: the reader's dashboard, the community page, or both.
    destinations: list[Literal["dashboard", "community"]] = Field(
        default_factory=lambda: ["dashboard", "community"]
    )
    published_at: datetime = Field(default_factory=utcnow)


class Correction(Doc):
    COLLECTION: ClassVar[str] = "corrections"

    story_id: str
    description: str  # what was wrong and what is right (JSN 35)
    reported_by: str
    status: Literal["open", "accepted", "rejected", "published"] = "open"
    published_at: datetime | None = None


# ---------------------------------------------------------------- community desk


class CommunityMessage(Doc):
    COLLECTION: ClassVar[str] = "community_messages"

    community_id: str
    kind: Literal["question", "tip", "correction", "local_knowledge"]
    text: str
    language: str = "en"
    author_ref: str = "anonymous"  # pseudonymous reference, never raw contact data
    story_id: str | None = None
    status: Literal["new", "triaged", "answered", "rejected"] = "new"
    created_at: datetime = Field(default_factory=utcnow)


class CommunityTriage(BaseModel):
    action: Literal["open_pitch", "open_correction", "answer", "reject", "needs_human"]
    reason: str
    reply: str = ""
    pitch: Pitch | None = None
    correction: Correction | None = None


# ----------------------------------------------------------- topics, events, the tree


class Topic(Doc):
    """A persistent subject area: a node of the news tree."""

    COLLECTION: ClassVar[str] = "topics"

    name: str
    slug: str
    parent_id: str | None = None
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    status: Literal["active", "proposed"] = "active"
    # Ids of related topics (the semantic graph). Always symmetric: the Connector agent adds
    # each link to both topics.
    branches: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class TimelineEntry(BaseModel):
    at: datetime
    text: str
    article_id: str | None = None


class Event(Doc):
    """One real-world development, reported by one or many outlets in many languages.

    When Kaleva and Yle (or an English and a Finnish report) cover the same news they
    attach to the same Event, so the newsroom investigates the event once and can show
    how many independent outlets reported it.
    """

    COLLECTION: ClassVar[str] = "events"

    title: str
    summary: str = ""
    topic_ids: list[str] = Field(default_factory=list)
    article_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)  # people, places, bodies, numbers
    place: str | None = None
    timeline: list[TimelineEntry] = Field(default_factory=list)
    status: Literal["candidate", "confirmed", "merged"] = "candidate"
    merged_into: str | None = None
    first_seen: datetime = Field(default_factory=utcnow)
    last_updated: datetime = Field(default_factory=utcnow)


class TreeChange(BaseModel):
    kind: Literal[
        "create_topic", "create_event", "attach_article", "attach_event_to_topic", "merge_events"
    ]
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_article_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class TreeProposal(Doc):
    """Agents propose tree changes; code validates them; risky ones wait for a human."""

    COLLECTION: ClassVar[str] = "tree_proposals"

    changes: list[TreeChange]
    status: Literal["pending", "auto_committed", "needs_human", "approved", "rejected"] = "pending"
    validation_notes: list[str] = Field(default_factory=list)
    decided_by: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Member(Doc):
    """A community member as the newsroom knows them. ``id`` is a pseudonymous reference
    (a hash of the account id), never an e-mail address or a name."""

    COLLECTION: ClassVar[str] = "members"

    community_id: str | None = None
    language: str = "en"  # the language the member reads in
    # Further languages the member also reads. Stories in any other language are never shown.
    reading_languages: list[str] = Field(default_factory=list)
    # Set by an editor when this person is one of the community's trusted voices.
    trusted_voice_id: str | None = None
    # Community admins and moderators are the community's trusted voices. They may act as
    # reviewers only in the roles (and languages) they have been given.
    community_role: Literal["member", "moderator", "admin"] = "member"
    review_roles: list[ReviewRole] = Field(default_factory=list)
    review_languages: list[str] = Field(default_factory=list)
    # Shown to readers next to an approval, so it is set only with the person's consent.
    display_name: str = ""
    created_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------- following, reactions and sharing

ShareTarget = Literal["facebook", "x", "whatsapp", "telegram", "linkedin", "email", "copy_link"]


class OfficialAccount(Doc):
    """An authority or service a member can follow next to the communities (Migri, the city)."""

    COLLECTION: ClassVar[str] = "official_accounts"

    name: str
    description: str = ""
    url: str = ""
    languages: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    # Feeds (``SourceConfig.id``) whose stories show up when a member follows this account.
    source_ids: list[str] = Field(default_factory=list)
    priority: int = 100


class Follow(Doc):
    """A member follows a community or an official account. The id is derived from the
    three fields, so following twice is one record."""

    COLLECTION: ClassVar[str] = "follows"

    member_id: str
    target_kind: Literal["community", "official"]
    target_id: str
    created_at: datetime = Field(default_factory=utcnow)

    @staticmethod
    def key(member_id: str, target_kind: str, target_id: str) -> str:
        return f"follow:{member_id}:{target_kind}:{target_id}"


class Reaction(Doc):
    """A like. One per member and story (the id is derived), removed again by unliking."""

    COLLECTION: ClassVar[str] = "reactions"

    member_id: str
    story_id: str
    community_id: str | None = None  # where the member saw the story
    kind: Literal["like"] = "like"
    created_at: datetime = Field(default_factory=utcnow)

    @staticmethod
    def key(member_id: str, story_id: str) -> str:
        return f"like:{member_id}:{story_id}"


class Comment(Doc):
    COLLECTION: ClassVar[str] = "comments"

    story_id: str
    member_id: str  # pseudonymous reference
    community_id: str | None = None
    text: str
    language: str = "en"
    status: Literal["visible", "hidden"] = "visible"
    hidden_by: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Share(Doc):
    """A member chose to share a story to a social network (or copy its link)."""

    COLLECTION: ClassVar[str] = "shares"

    story_id: str
    member_id: str
    community_id: str | None = None
    platform: ShareTarget
    created_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------------------ audit trail


class TraceEvent(Doc):
    """Every agent decision, with the reason, so the newsroom can be audited."""

    COLLECTION: ClassVar[str] = "agent_runs"

    run_id: str
    agent: str
    step: str
    decision: str
    reason: str = ""
    refs: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
