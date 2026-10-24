"""The newsroom: wires the agents, the graph and the human review queue together."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from newsroom.agents.architect import ArchitectResult, KnowledgeArchitectAgent
from newsroom.agents.base import AgentContext
from newsroom.agents.community_desk import CommunityDeskAgent
from newsroom.agents.connector import ConnectorResult, TopicConnectorAgent
from newsroom.agents.distributor import DistributorAgent
from newsroom.agents.editor import NewsEditorAgent
from newsroom.agents.monitor import MonitorAgent
from newsroom.config import Settings, get_settings
from newsroom.domain import (
    Article,
    Community,
    CommunityMessage,
    CommunityTriage,
    Correction,
    Decision,
    Member,
    Pitch,
    Publication,
    ReviewDecision,
    ReviewRole,
    SourceConfig,
    Story,
    StoryStatus,
    TraceEvent,
    Translation,
    new_id,
)
from newsroom.graph import StoryWorkflow
from newsroom.hitl import (
    ReviewError,
    approvals_for,
    attach_source,
    record_decisions,
    review_queue,
    submit_human_translation,
    validate_decisions,
)
from newsroom.limits import CycleLimits
from newsroom.llm import build_llm
from newsroom.llm.budget import BudgetedLLM
from newsroom.registry import load_communities, load_fixture_articles, load_sources
from newsroom.store import build_store
from newsroom.store.base import Store
from newsroom.topics import seed_topics


@dataclass
class StoryRun:
    pitch_id: str
    community_id: str
    working_title: str
    status: str  # awaiting_review | published | rejected | blocked
    story_id: str | None = None
    publication_id: str | None = None
    pending: dict[str, Any] | None = None


@dataclass
class CycleReport:
    run_id: str
    new_articles: int = 0
    flagged_articles: list[str] = field(default_factory=list)
    events: int = 0
    event_ids: list[str] = field(default_factory=list)
    pitches: int = 0
    runs: list[StoryRun] = field(default_factory=list)
    articles_processed: int = 0
    articles_deferred: int = 0  # saved, and picked up by the next cycle
    stopped_reason: str | None = None
    connector: dict[str, Any] | None = None  # topic-relationship work, when configured
    model_usage: dict[str, dict[str, float]] = field(default_factory=dict)


def make_checkpointer(settings: Settings, *, persistent: bool = True) -> Any:
    """SQLite checkpoints let a human review a story days later, even after a restart."""

    if not persistent:
        return InMemorySaver()
    path = Path(settings.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


class Newsroom:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: Store | None = None,
        llm: Any | None = None,
        communities: list[Community] | None = None,
        sources: list[SourceConfig] | None = None,
        checkpointer: Any | None = None,
        persistent_checkpoints: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or build_store(self.settings)
        self.limits = CycleLimits(
            max_model_calls=self.settings.cycle_max_model_calls,
            max_seconds=self.settings.cycle_max_seconds,
        )
        self.ctx = AgentContext(
            llm=BudgetedLLM(
                llm or build_llm(self.settings),
                self.limits,
                max_input_tokens=self.settings.llm_max_input_tokens,
            ),
            store=self.store,
            settings=self.settings,
            limits=self.limits,
        )
        self.communities = {c.id: c for c in (communities or load_communities())}
        for community in self.communities.values():
            self.store.put(community)
        self.sources = sources if sources is not None else load_sources()
        self.monitor = MonitorAgent(self.ctx)
        self.architect = KnowledgeArchitectAgent(self.ctx)
        self.editor = NewsEditorAgent(self.ctx)
        self.desk = CommunityDeskAgent(self.ctx)
        self.connector = TopicConnectorAgent(self.ctx)
        self.distributor = DistributorAgent(self.ctx)
        self.workflow = StoryWorkflow(
            self.ctx,
            self.communities,
            checkpointer or make_checkpointer(self.settings, persistent=persistent_checkpoints),
        )
        seed_topics(self.store)

    # ------------------------------------------------------------------- cycle

    def run_cycle(
        self,
        community_ids: list[str] | None = None,
        *,
        use_fixtures: bool = False,
        fixtures: list[Article] | None = None,
    ) -> CycleReport:
        """Collect news, cluster it into events, pick stories per community, and run each
        story until it needs a human.

        A cycle is bounded: a few items per feed, a cap on articles, a time budget, a model-call
        budget and a Stop button. Whatever is left over is saved and handled next time."""

        self.limits.start()
        try:
            return self._run_cycle(community_ids, use_fixtures=use_fixtures, fixtures=fixtures)
        finally:
            self.limits.finish()

    def _backlog(self, exclude: set[str], room: int) -> list[Article]:
        """Saved articles that were never clustered (a cycle ended early), newest first."""

        if room <= 0:
            return []
        waiting = [a for a in self.store.find(Article, event_id=None) if a.id not in exclude]
        waiting.sort(
            key=lambda a: a.published_at.timestamp() if a.published_at else 0, reverse=True
        )
        return waiting[:room]

    def _run_cycle(
        self,
        community_ids: list[str] | None,
        *,
        use_fixtures: bool,
        fixtures: list[Article] | None,
    ) -> CycleReport:
        report = CycleReport(run_id=self.ctx.run_id)
        cap = self.settings.max_articles_per_cycle
        new = self.monitor.scan(
            self.sources,
            limit_per_source=self.settings.max_items_per_source,
            max_new=cap,
            fixtures=(fixtures if fixtures is not None else load_fixture_articles())
            if use_fixtures or fixtures is not None
            else None,
        )
        report.new_articles = len(new)
        report.flagged_articles = [a.id for a in new if a.injection_flags]

        backlog = self._backlog({a.id for a in new}, cap - len(new))
        articles = [*new, *backlog][:cap]
        report.articles_processed = len(articles)
        report.articles_deferred = max(0, len(new) + len(backlog) - len(articles))

        architect_result: ArchitectResult = self.architect.process(articles)
        report.events = len(architect_result.events)
        report.event_ids = [e.id for e in architect_result.events]
        # The architect sets event_id on the stored articles; reload so selection sees it.
        articles = [self.store.get(Article, a.id) or a for a in articles]

        wanted = community_ids or list(self.communities)
        for community_id in wanted:
            if self.limits.stop_reason():
                break
            community = self.communities[community_id]
            pitches = self.editor.select(
                community, articles, max_pitches=self.settings.max_pitches_per_cycle
            )
            report.pitches += len(pitches)
            for pitch in pitches:
                if self.limits.stop_reason():
                    break
                report.runs.append(self._run_story(pitch))
        report.stopped_reason = self.limits.triggered
        connector_report = self._run_connector()
        if connector_report is not None:
            report.connector = connector_report.model_dump(mode="json")
        report.stopped_reason = self.limits.triggered
        report.model_usage = {task: dict(stats) for task, stats in self.limits.usage.items()}
        if report.model_usage:
            self._trace_model_usage(report)
        return report

    def _trace_model_usage(self, report: CycleReport) -> None:
        """Write prompt size and latency per task to the audit trail."""

        total_calls = sum(int(s["calls"]) for s in report.model_usage.values())
        total_tokens = sum(int(s["input_tokens"]) for s in report.model_usage.values())
        total_seconds = sum(s["seconds"] for s in report.model_usage.values())
        self.store.trace(
            TraceEvent(
                id=new_id("trace"),
                run_id=report.run_id,
                agent="model_layer",
                step="usage",
                decision="recorded",
                reason=(
                    f"{total_calls} model call(s), ~{total_tokens} input token(s), "
                    f"{total_seconds:.1f}s in calls"
                ),
                refs={
                    "usage": report.model_usage,
                    "max_input_tokens": self.settings.llm_max_input_tokens,
                },
            )
        )

    def _run_connector(self) -> ConnectorResult | None:
        """Link related topics through the backend API when the integration is on.

        Runs inside the normal cycle (same run id, same model budget, persisted
        traces). A failed backend call must never break the cycle: the result
        records it and the next run converges on the same graph.
        """

        if not self.settings.backend_api_url:
            return None
        reason = self.limits.stop_reason()
        if reason:
            self.connector.trace(
                "run",
                "deferred",
                f"cycle budget spent ({reason}); connector waits for the next run",
            )
            return None
        return self.connector.run()

    def run_connector(self) -> ConnectorResult | None:
        """Refresh topic relationships on demand (no-op when the backend is not configured)."""

        return self._run_connector()

    def _run_story(self, pitch: Pitch) -> StoryRun:
        result = self.workflow.start(pitch)
        return self._summarise(pitch, result)

    def _summarise(self, pitch: Pitch, result: dict[str, Any]) -> StoryRun:
        pending = self.workflow.pending(pitch.id)
        values = self.workflow.values(pitch.id)
        outcome = values.get("outcome") or result.get("outcome") or ""
        status = "awaiting_review" if pending else (outcome or "finished")
        return StoryRun(
            pitch_id=pitch.id,
            community_id=pitch.community_id,
            working_title=pitch.working_title,
            status=status,
            story_id=values.get("story_id"),
            publication_id=values.get("publication_id"),
            pending=pending,
        )

    # ------------------------------------------------------------------ review

    def review_queue(self) -> list[dict[str, Any]]:
        """Stories waiting for a human, with exactly what is still missing.

        Includes stories already published in some languages that are waiting for a language
        reviewer to approve the rest."""

        items = []
        for story in review_queue(self.store):
            pending = self.workflow.pending(story.pitch_id)
            if pending:
                items.append(pending)
        items.extend(self._language_review_items())
        return items

    def _pending_languages(self, story: Story) -> dict[str, str]:
        """Languages of a published story that still need a language reviewer."""

        community = self.communities[story.community_id]
        translations = [
            t
            for t in self.store.find(Translation, story_id=story.id)
            if t.story_version == story.version
        ]
        publishable = {
            code
            for code, _ in self.distributor._publishable_languages(
                story, translations, community, story.risk_tier
            )
        }
        return {
            t.language: t.status
            for t in translations
            if t.language not in publishable and t.status == "machine_draft"
        }

    def _language_review_items(self) -> list[dict[str, Any]]:
        items = []
        for story in self.store.find(Story, status=StoryStatus.PUBLISHED.value):
            pending = self._pending_languages(story)
            if not pending:
                continue
            items.append(
                {
                    "type": "language_review",
                    "pitch_id": story.pitch_id,
                    "community_id": story.community_id,
                    "story_id": story.id,
                    "story_version": story.version,
                    "community": self.communities[story.community_id].name,
                    "headline": story.headline,
                    "standfirst": story.standfirst,
                    "risk_tier": story.risk_tier.value,
                    "missing_roles": [ReviewRole.LANGUAGE_REVIEWER.value],
                    "blocking_checks": [],
                    "context_needing_a_human": [],
                    "translations": pending,
                    "sources": [],
                }
            )
        return items

    def _review_published_languages(
        self, pitch: Pitch, decisions: list[dict[str, Any]]
    ) -> StoryRun:
        """A language reviewer approves a language of an already published story. The story
        is published again with that language added."""

        stories = [s for s in self.store.find(Story, pitch_id=pitch.id)]
        story = max(stories, key=lambda s: s.version) if stories else None
        if story is None or story.status not in (StoryStatus.PUBLISHED, StoryStatus.CORRECTED):
            raise ReviewError("This story is not waiting for a review.")
        if any(item.get("role") != ReviewRole.LANGUAGE_REVIEWER.value for item in decisions):
            raise ReviewError("Only language reviews can be added to a published story.")
        community = self.communities[story.community_id]
        record_decisions(self.store, story, community, decisions)
        translations = [
            t
            for t in self.store.find(Translation, story_id=story.id)
            if t.story_version == story.version
        ]
        approvals = [
            d
            for d in approvals_for(self.store, story.id, story.version)
            if d.decision is Decision.APPROVE
        ]
        previous = [p for p in self.store.find(Publication, story_id=story.id)]
        publication = self.distributor.render(story, translations, community, approvals)
        # Credit the language reviewers too: readers should see who approved each language.
        credited = [d.id for d in approvals if d.role is ReviewRole.LANGUAGE_REVIEWER]
        publication.approval_ids = list(dict.fromkeys([*publication.approval_ids, *credited]))
        self.store.put(publication)
        for old in previous:
            if old.id != publication.id:
                self.store.delete(Publication, old.id)
        return StoryRun(
            pitch_id=pitch.id,
            community_id=pitch.community_id,
            working_title=pitch.working_title,
            status="published",
            story_id=story.id,
            publication_id=publication.id,
        )

    def submit_review(self, pitch_id: str, decisions: list[dict[str, Any]]) -> StoryRun:
        """Resume a paused story with decisions from named humans."""

        pitch = self.store.get(Pitch, pitch_id)
        if pitch is None:
            raise KeyError(f"Unknown pitch {pitch_id}.")
        validate_decisions(self.communities[pitch.community_id], decisions, self.store)
        if self.workflow.pending(pitch_id) is None:
            return self._review_published_languages(pitch, decisions)
        result = self.workflow.resume(pitch_id, decisions)
        return self._summarise(pitch, result)

    def submit_human_translation(
        self,
        story_id: str,
        *,
        language: str,
        headline: str,
        standfirst: str,
        body_md: str,
        reviewer: str,
    ):
        story = self.store.get(Story, story_id)
        if story is None:
            raise KeyError(f"Unknown story {story_id}.")
        return submit_human_translation(
            self.store,
            story,
            self.communities[story.community_id],
            language=language,
            headline=headline,
            standfirst=standfirst,
            body_md=body_md,
            reviewer=reviewer,
        )

    def add_story_source(
        self, story_id: str, *, name: str, url: str, official: bool, reviewer: str
    ) -> Story:
        story = self.store.get(Story, story_id)
        if story is None:
            raise KeyError(f"Unknown story {story_id}.")
        return attach_source(
            self.store,
            story,
            self.communities[story.community_id],
            name=name,
            url=url,
            official=official,
            reviewer=reviewer,
        )

    # --------------------------------------------------------------- community

    def community_message(self, message: CommunityMessage) -> CommunityTriage:
        """A question, tip, correction or piece of local knowledge from a community member."""

        self.store.put(message)
        community = self.communities[message.community_id]
        triage = self.desk.triage(message, community)
        if triage.pitch is not None:
            self.store.put(triage.pitch)
        if triage.correction is not None:
            self.store.put(triage.correction)
        return triage

    def publish_correction(self, correction_id: str, *, reviewer: str, text: str) -> Correction:
        """Corrections are published by a named editor and say what was wrong (JSN 35–37)."""

        if not reviewer.strip():
            raise ValueError("A correction needs the name of the editor who approved it.")
        correction = self.store.get(Correction, correction_id)
        if correction is None:
            raise KeyError(f"Unknown correction {correction_id}.")
        from newsroom.domain import utcnow

        correction.description = text
        correction.status = "published"
        correction.published_at = utcnow()
        self.store.put(correction)
        story = self.store.get(Story, correction.story_id)
        if story is not None:
            story.status = StoryStatus.CORRECTED
            self.store.put(story)
        self.store.trace(
            TraceEvent(
                id=new_id("trace"),
                run_id=self.ctx.run_id,
                agent="orchestrator",
                step="correction",
                decision="published",
                reason=f"Correction approved by {reviewer}.",
                refs={"correction_id": correction.id, "story_id": correction.story_id},
            )
        )
        return correction

    # ------------------------------------------------------------------ reading

    def feed(self, community_id: str) -> list[Publication]:
        pubs = [p for p in self.store.find(Publication) if p.community_id == community_id]
        return sorted(pubs, key=lambda p: p.published_at, reverse=True)

    def public_feed(self, community_id: str) -> list[dict[str, Any]]:
        """What a community member sees: the story, who approved it, its sources, the AI
        disclosure and any published corrections. Internal drafts never appear here."""

        community = self.communities[community_id]
        voices = {v.id: v for v in community.trusted_voices}
        items: list[dict[str, Any]] = []
        for publication in self.feed(community_id):
            story = self.store.get(Story, publication.story_id)
            if story is None:
                continue
            approvers = []
            for approval_id in publication.approval_ids:
                approval = self.store.get(ReviewDecision, approval_id)
                if approval is None:
                    continue
                voice = voices.get(approval.reviewer)
                member = (
                    self.store.get(Member, approval.reviewer.removeprefix("member:"))
                    if (approval.reviewer.startswith("member:"))
                    else None
                )
                if member is not None:
                    name = member.display_name or "Community moderator"
                    title = f"community {member.community_role}"
                else:
                    name = voice.name if voice else approval.reviewer
                    title = voice.role if voice else "newsroom editor"
                approvers.append({"role": approval.role.value, "name": name, "title": title})
            corrections = [
                {"text": c.description, "published_at": c.published_at}
                for c in self.store.find(Correction, story_id=story.id)
                if c.status == "published"
            ]
            items.append(
                {
                    "story_id": story.id,
                    "published_at": publication.published_at,
                    "headline": story.headline,
                    "standfirst": story.standfirst,
                    "needs": [n.value for n in story.needs],
                    "risk_tier": story.risk_tier.value,
                    "sources": [s.model_dump() for s in story.sources],
                    "context_notes": [
                        n.model_dump() for n in story.context_notes if n.kind != "human_needed"
                    ],
                    "ai_disclosure": story.ai_disclosure,
                    "approvers": approvers,
                    "corrections": corrections,
                    "renditions": [r.model_dump() for r in publication.renditions],
                }
            )
        return items

    # ----------------------------------------------------------------- members

    def member(self, ref: str) -> Member:
        existing = self.store.get(Member, ref)
        if existing is not None:
            return existing
        member = Member(id=ref)
        self.store.put(member)
        return member

    def join_community(self, ref: str, community_id: str, language: str | None = None) -> Member:
        community = self.communities.get(community_id)
        if community is None:
            raise KeyError(f"Unknown community {community_id}.")
        member = self.member(ref)
        member.community_id = community_id
        if language and community.lane(language) is not None:
            member.language = language
        self.store.put(member)
        return member

    def set_community_role(
        self,
        ref: str,
        community_id: str,
        *,
        community_role: str,
        review_roles: list[str],
        review_languages: list[str],
        display_name: str,
    ) -> Member:
        """Appoint (or remove) a community admin or moderator. They are the community's
        trusted voices and may review only in the roles and languages given here."""

        community = self.communities.get(community_id)
        if community is None:
            raise KeyError(f"Unknown community {community_id}.")
        if community_role not in {"member", "moderator", "admin"}:
            raise ValueError("community_role must be member, moderator or admin.")
        roles = [ReviewRole(r) for r in review_roles]
        if ReviewRole.EDITOR_IN_CHIEF in roles:
            raise ValueError("Community moderators cannot hold the editor-in-chief role.")
        allowed_languages = {lane.code for lane in community.languages}
        languages = [code for code in review_languages if code in allowed_languages]
        if community_role == "member":
            roles, languages = [], []
        member = self.member(ref)
        member.community_id = community_id
        member.community_role = community_role  # type: ignore[assignment]
        member.review_roles = roles
        member.review_languages = languages
        member.display_name = display_name.strip()[:120]
        self.store.put(member)
        return member

    def link_trusted_voice(self, ref: str, community_id: str, voice_id: str) -> Member:
        """An editor confirms that this account belongs to one of the trusted voices."""

        community = self.communities.get(community_id)
        if community is None or voice_id not in {v.id for v in community.trusted_voices}:
            raise KeyError("Unknown community or trusted voice.")
        member = self.member(ref)
        member.community_id = community_id
        member.trusted_voice_id = voice_id
        self.store.put(member)
        return member

    def my_messages(self, ref: str, community_id: str | None = None) -> list[CommunityMessage]:
        messages = [m for m in self.store.find(CommunityMessage) if m.author_ref == ref]
        if community_id:
            messages = [m for m in messages if m.community_id == community_id]
        return sorted(messages, key=lambda m: m.created_at, reverse=True)
