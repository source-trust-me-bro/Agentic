"""Human-in-the-loop rules.

A story is published only when named humans have approved the exact version that will
go out, in every role the standards editor required, and no blocking standards failure
remains. These rules are plain code, independent of any model, and they are enforced
twice: here (graph) and again inside the distributor.
"""

from __future__ import annotations

from newsroom.domain import (
    Community,
    Decision,
    Member,
    ReviewDecision,
    ReviewRole,
    SourceRef,
    StandardsReport,
    Story,
    StoryStatus,
    Translation,
    new_id,
    utcnow,
)
from newsroom.store.base import Store


class ReviewError(ValueError):
    """A review decision that is not allowed (unknown reviewer, wrong role...)."""


def latest_report(store: Store, story_id: str, version: int) -> StandardsReport | None:
    reports = [
        r for r in store.find(StandardsReport, story_id=story_id) if r.story_version == version
    ]
    return max(reports, key=lambda r: r.created_at) if reports else None


def approvals_for(store: Store, story_id: str, version: int) -> list[ReviewDecision]:
    return [d for d in store.find(ReviewDecision, story_id=story_id) if d.story_version == version]


MEMBER_PREFIX = "member:"


def _member_for(store: Store | None, reviewer: str) -> Member | None:
    if store is None or not reviewer.startswith(MEMBER_PREFIX):
        return None
    return store.get(Member, reviewer.removeprefix(MEMBER_PREFIX))


def reviewer_languages(store: Store | None, community: Community, reviewer: str) -> set[str]:
    """Languages this reviewer is trusted to approve."""

    member = _member_for(store, reviewer)
    if member is not None:
        return set(member.review_languages)
    return {
        lang
        for voice in community.trusted_voices
        if reviewer.strip() in (voice.id, voice.name)
        for lang in voice.languages
    }


def validate_reviewer(
    role: ReviewRole, reviewer: str, community: Community, store: Store | None = None
) -> None:
    """Reviewers are named humans. Community roles must come from the community's own
    trusted voices: its admins and moderators (members the community has appointed) and
    the voices listed in its configuration. The community takes part in quality assurance."""

    if not reviewer or not reviewer.strip():
        raise ReviewError("A review needs the name of a human reviewer.")
    if reviewer.strip().lower() in {"agent", "ai", "bot", "system", "model"}:
        raise ReviewError("Agents cannot approve stories.")
    if role is ReviewRole.EDITOR_IN_CHIEF:
        if reviewer.startswith(MEMBER_PREFIX):
            raise ReviewError("The editor-in-chief role is for newsroom staff.")
        return  # newsroom staff; the API authenticates who may use this role
    if reviewer.startswith(MEMBER_PREFIX):
        member = _member_for(store, reviewer)
        if (
            member is None
            or member.community_id != community.id
            or member.community_role not in {"moderator", "admin"}
        ):
            raise ReviewError(f"This account is not a moderator or admin of '{community.name}'.")
        if role not in member.review_roles:
            raise ReviewError(f"This moderator has not been given the '{role.value}' role.")
        if role is ReviewRole.LANGUAGE_REVIEWER and not member.review_languages:
            raise ReviewError("A language reviewer needs at least one language.")
        return
    voices = [
        v
        for v in community.trusted_voices
        if reviewer.strip() in (v.id, v.name) and role in v.can_review
    ]
    if not voices:
        raise ReviewError(
            f"'{reviewer}' is not a trusted voice of '{community.name}' for role '{role.value}'."
        )


def validate_decisions(community: Community, raw: list[dict], store: Store | None = None) -> None:
    """Check every decision BEFORE the paused workflow is resumed, so a bad submission can
    never leave a story stuck on an invalid answer."""

    if not raw:
        raise ReviewError("Submit at least one decision.")
    for item in raw:
        try:
            role = ReviewRole(item["role"])
            Decision(item["decision"])
        except (KeyError, ValueError) as exc:
            raise ReviewError(f"Invalid decision: {item!r}") from exc
        validate_reviewer(role, str(item.get("reviewer", "")), community, store)


def missing_roles(store: Store, story: Story) -> list[ReviewRole]:
    report = latest_report(store, story.id, story.version)
    if report is None:
        return [ReviewRole.EDITOR_IN_CHIEF]
    needed = {ReviewRole.EDITOR_IN_CHIEF, *report.required_reviews}
    approved = {
        d.role
        for d in approvals_for(store, story.id, story.version)
        if d.decision is Decision.APPROVE
    }
    return sorted(needed - approved, key=lambda role: role.value)


def can_publish(store: Store, story: Story) -> tuple[bool, list[str]]:
    """Return (allowed, reasons it is not allowed)."""

    reasons: list[str] = []
    report = latest_report(store, story.id, story.version)
    if report is None:
        reasons.append("No standards review exists for this version.")
    elif report.blocking_failures:
        reasons.extend(f"Blocking check failed: {c.name}" for c in report.blocking_failures)
    decisions = approvals_for(store, story.id, story.version)
    if any(d.decision is Decision.REJECT for d in decisions):
        reasons.append("A reviewer rejected this version.")
    reasons.extend(f"Missing approval: {role.value}" for role in missing_roles(store, story))
    return (not reasons, reasons)


def record_decisions(
    store: Store, story: Story, community: Community, raw: list[dict]
) -> list[ReviewDecision]:
    """Validate and store human decisions for the story's current version."""

    saved: list[ReviewDecision] = []
    for item in raw:
        role = ReviewRole(item["role"])
        reviewer = str(item.get("reviewer", ""))
        validate_reviewer(role, reviewer, community, store)
        decision = ReviewDecision(
            id=new_id("rev"),
            story_id=story.id,
            story_version=story.version,
            role=role,
            reviewer=reviewer.strip(),
            decision=Decision(item["decision"]),
            notes=str(item.get("notes", "")),
        )
        store.put(decision)
        saved.append(decision)
        if role is ReviewRole.LANGUAGE_REVIEWER and decision.decision is Decision.APPROVE:
            _mark_translations_reviewed(
                store, story, reviewer_languages(store, community, decision.reviewer)
            )
    return saved


def _mark_translations_reviewed(store: Store, story: Story, languages: set[str]) -> None:
    """A reviewer approves only the languages they are trusted for."""

    for translation in store.find(Translation, story_id=story.id):
        if (
            translation.story_version == story.version
            and translation.status == "machine_draft"
            and translation.language in languages
        ):
            translation.status = "human_reviewed"
            store.put(translation)


def submit_human_translation(
    store: Store,
    story: Story,
    community: Community,
    *,
    language: str,
    headline: str,
    standfirst: str,
    body_md: str,
    reviewer: str,
) -> Translation:
    """A language reviewer supplies the text themselves (the only route for tier-C
    languages such as North Sámi)."""

    validate_reviewer(ReviewRole.LANGUAGE_REVIEWER, reviewer, community)
    translation = Translation(
        id=new_id("tr"),
        story_id=story.id,
        story_version=story.version,
        language=language,
        headline=headline,
        standfirst=standfirst,
        body_md=body_md,
        status="human_reviewed",
        machine_translated=False,
    )
    # Replace any placeholder for this language and version.
    for existing in store.find(Translation, story_id=story.id):
        if existing.story_version == story.version and existing.language == language:
            store.delete(Translation, existing.id)
    store.put(translation)
    return translation


def attach_source(
    store: Store,
    story: Story,
    community: Community,
    *,
    name: str,
    url: str,
    official: bool,
    reviewer: str,
) -> Story:
    """A named editor attaches a source the agents could not find, most often the official
    one a high-risk story must link. The standards review re-runs when review resumes."""

    validate_reviewer(ReviewRole.EDITOR_IN_CHIEF, reviewer, community)
    if not url.lower().startswith(("http://", "https://")):
        raise ReviewError("A source needs an http(s) link.")
    if not name.strip():
        raise ReviewError("A source needs a name.")
    if all(ref.url != url for ref in story.sources):
        story.sources.append(
            SourceRef(name=name.strip(), url=url, official=official, origin="editor")
        )
        story.updated_at = utcnow()
        store.put(story)
    return story


def review_queue(store: Store) -> list[Story]:
    return [s for s in store.find(Story, status=StoryStatus.IN_REVIEW.value)]
