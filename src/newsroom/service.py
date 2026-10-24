"""The newsroom: wires the agents, the graph and the human review queue together."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from newsroom.agents.architect import ArchitectResult, KnowledgeArchitectAgent
from newsroom.agents.base import AgentContext
from newsroom.agents.community_desk import CommunityDeskAgent
from newsroom.agents.connector import ConnectorAgent
from newsroom.agents.distributor import DistributorAgent
from newsroom.agents.editor import NewsEditorAgent
from newsroom.agents.explainer import limit_words, word_count
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
    Topic,
    TraceEvent,
    Translation,
    new_id,
)
from newsroom.engagement import EngagementService
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
from newsroom.parallel import run_parallel, start_in_background
from newsroom.registry import load_communities, load_fixture_articles, load_sources
from newsroom.store import build_store
from newsroom.store.base import Store
from newsroom.topics import expected_topic_ids, seed_topics


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
    topic_links: int = 0  # new links the Connector added to the topic graph
    stopped_reason: str | None = None


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
                self.settings.max_concurrent_model_calls,
            ),
            store=self.store,
            settings=self.settings,
            limits=self.limits,
        )
        self.communities = {c.id: c for c in (communities or load_communities())}
        for community in self.communities.values():
            self.store.put(community)
        # Sources given by the caller stay as given; otherwise sources.yaml is read again before
        # every cycle, so adding or removing a feed needs no restart.
        self._sources_from_file = sources is None
        self.sources = sources if sources is not None else load_sources(self.settings.sources_file)
        self.monitor = MonitorAgent(self.ctx)
        self.architect = KnowledgeArchitectAgent(self.ctx)
        self.connector = ConnectorAgent(self.ctx)
        self.editor = NewsEditorAgent(self.ctx)
        self.desk = CommunityDeskAgent(self.ctx)
        self.distributor = DistributorAgent(self.ctx)
        self.workflow = StoryWorkflow(
            self.ctx,
            self.communities,
            checkpointer or make_checkpointer(self.settings, persistent=persistent_checkpoints),
        )
        self.ensure_seeded()
        self.engagement = EngagementService(self)
        self.ctx.progress.usage_source = self.ctx.llm.stats

    def reload_sources(self) -> None:
        """Read config/sources.yaml again. A file that cannot be read keeps the current list."""

        if not self._sources_from_file:
            return
        try:
            self.sources = load_sources(self.settings.sources_file)
        except Exception:  # noqa: BLE001 - a typo in the file must not stop a cycle
            self.ctx.progress.log("newsroom", "sources.yaml could not be read; using the last list", "error")

    def ensure_seeded(self) -> None:
        """Put the configured topics and communities in the database if they are missing.

        Done at start-up, before every cycle and when the topic tree is read, so emptying the
        database while the service runs does not leave an empty tree until the next restart.
        Idempotent: what already exists is kept as it is."""

        # Two queries when nothing is missing, which is almost always.
        have = self.store.get_many(Community, self.communities.keys())
        for community in self.communities.values():
            if community.id not in have:
                self.store.put(community)
        if not expected_topic_ids() <= {t.id for t in self.store.find(Topic)}:
            seed_topics(self.store)

    # ------------------------------------------------------------------- cycle

    def run_cycle(
        self,
        community_ids: list[str] | None = None,
        *,
        use_fixtures: bool = False,
        fixtures: list[Article] | None = None,
        news_count: int | None = None,
    ) -> CycleReport:
        """Collect news, cluster it into events, pick stories per community, and run each
        story until it needs a human.

        ``news_count`` is the editor's choice of how many articles to collect this time (10, 50
        or 100). A cycle is still bounded by a time budget, a model-call budget and a Stop
        button. Whatever is left over is saved and handled next time."""

        self.reload_sources()
        self.limits.start()
        self.ctx.llm.reset_stats()
        self.ctx.progress.start()
        try:
            self.ensure_seeded()
            report = self._run_cycle(
                community_ids, use_fixtures=use_fixtures, fixtures=fixtures, news_count=news_count
            )
            self.ctx.progress.finish(
                "stopped" if report.stopped_reason else "done", report.stopped_reason or ""
            )
            return report
        except Exception as exc:
            self.ctx.progress.finish("failed", f"{type(exc).__name__}: {str(exc)[:200]}")
            raise
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
        news_count: int | None = None,
    ) -> CycleReport:
        report = CycleReport(run_id=self.ctx.run_id)
        cap = news_count or self.settings.max_articles_per_cycle
        feeds = max(1, sum(1 for s in self.sources if s.enabled and s.feed_url))
        # Read twice a feed's share, so items stored by an earlier cycle do not hide new ones.
        per_source = max(self.settings.max_items_per_source, 2 * -(-cap // feeds))
        # A bigger intake means a few more stories per community, not one story per article.
        max_pitches = max(self.settings.max_pitches_per_cycle, cap // 25)
        new = self.monitor.scan(
            self.sources,
            limit_per_source=per_source,
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

        progress = self.ctx.progress
        progress.finish_stage("collect", detail=f"{len(new)} new articles")
        architect_result: ArchitectResult = self.architect.process(articles)
        report.events = len(architect_result.events)
        report.event_ids = [e.id for e in architect_result.events]
        # The architect sets event_id on the stored articles; reload so selection sees it.
        articles = [self.store.get(Article, a.id) or a for a in articles]
        # Linking related topics needs nothing from the stories, so it runs while they are written.
        def link() -> int:
            progress.begin("link", detail="Finding related topics")
            added = self.connector.connect()
            progress.finish_stage("link", detail=f"{added} new links")
            return added

        if self.limits.stop_reason():
            progress.finish_stage("link", "skipped", "stopped before linking")
            finish_linking = None
        else:
            finish_linking = start_in_background(link, "connector")

        wanted = community_ids or list(self.communities)
        progress.begin("select", total=len(wanted), detail="The news editor agent weighs each community's needs")

        def select(community_id: str) -> list[Pitch]:
            try:
                if self.limits.stop_reason():
                    return []
                community = self.communities[community_id]
                return self.editor.select(community, articles, max_pitches=max_pitches)
            finally:
                progress.advance("select")

        pitches = [p for found in self._parallel(select, wanted) for p in found]
        pitches = self._one_story_per_event(pitches)
        report.pitches = len(pitches)
        progress.finish_stage("select", detail=f"{len(pitches)} stories chosen")
        progress.begin("write", total=len(pitches), detail="Stories are written side by side")

        def run(pitch: Pitch) -> StoryRun | None:
            return None if self.limits.stop_reason() else self._run_story(pitch)

        try:
            report.runs.extend(r for r in self._parallel(run, pitches) if r is not None)
            progress.finish_stage(
                "write", detail=f"{len(report.runs)} stories ready for a person to review"
            )
        finally:
            if finish_linking is not None:
                report.topic_links = finish_linking()
        report.stopped_reason = self.limits.triggered
        return report

    def _one_story_per_event(self, pitches: list[Pitch]) -> list[Pitch]:
        """Several communities often pick the same news. Write it once, for the community that
        needs it most, and publish it to the others too (``Pitch.also_for``)."""

        articles = self.store.get_many(Article, (a for p in pitches for a in p.article_ids))

        def event_of(p: Pitch) -> str | None:
            return next(
                (
                    articles[a].event_id
                    for a in p.article_ids
                    if a in articles and articles[a].event_id
                ),
                None,
            )

        def priority(p: Pitch) -> int:
            community = self.communities.get(p.community_id)
            return community.priority if community else 100

        ranked = sorted(pitches, key=lambda p: (-p.score, priority(p)))
        owner: dict[str, Pitch] = {}
        keep: list[Pitch] = []
        for pitch in ranked:
            event = event_of(pitch)
            first = owner.get(event) if event else None
            if first is None:
                if event:
                    owner[event] = pitch
                keep.append(pitch)
                continue
            if pitch.community_id not in (first.community_id, *first.also_for):
                first.also_for.append(pitch.community_id)
            self.store.put(first)
            self.store.delete(Pitch, pitch.id)
        return keep

    def _targets(self, story: Story, pitch: Pitch | None = None) -> list[Community]:
        """The communities a story is published to: its own, then those in ``also_for``."""

        pitch = pitch or self.store.get(Pitch, story.pitch_id)
        also_for = pitch.also_for if pitch else []
        others = [self.communities[c] for c in also_for if c in self.communities]
        return [self.communities[story.community_id], *others]

    def _parallel(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        """Run ``fn`` over ``items`` a few at a time, results in the original order. Model calls
        are slow and the items are independent, so this is where a cycle gets its speed."""

        return run_parallel(fn, items, self.settings.story_workers)

    def _run_story(self, pitch: Pitch) -> StoryRun:
        community = self.communities.get(pitch.community_id)
        self.ctx.progress.story_started(
            pitch.id, pitch.working_title, community.name if community else pitch.community_id
        )
        try:
            result = self.workflow.start(pitch)
        except Exception:
            self.ctx.progress.story_finished(pitch.id, "failed")
            raise
        run = self._summarise(pitch, result)
        self.ctx.progress.story_finished(pitch.id, run.status)
        self.ctx.progress.advance("write")
        return run

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

    def _pending_languages(self, story: Story, pitch: Pitch | None = None) -> dict[str, str]:
        """Languages of a published story that still need a language reviewer, in any of
        the communities it was published to."""

        translations = [
            t
            for t in self.store.find(Translation, story_id=story.id)
            if t.story_version == story.version
        ]
        publishable = {
            code
            for community in self._targets(story, pitch)
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
        stories = self.store.find(Story, status=StoryStatus.PUBLISHED.value)
        pitches = self.store.get_many(Pitch, (s.pitch_id for s in stories))
        for story in stories:
            pending = self._pending_languages(story, pitches.get(story.pitch_id))
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
        # Published again in every community that has the story (its own first).
        credited = [d.id for d in approvals if d.role is ReviewRole.LANGUAGE_REVIEWER]
        fresh: list[Publication] = []
        for target in self._targets(story, pitch):
            renewed = self.distributor.render(story, translations, target, approvals)
            # Credit the language reviewers too: readers should see who approved each language.
            renewed.approval_ids = list(dict.fromkeys([*renewed.approval_ids, *credited]))
            self.store.put(renewed)
            fresh.append(renewed)
        publication = fresh[0]
        for old in previous:
            if old.id not in {p.id for p in fresh}:
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
        # The editor-in-chief says where the story goes: dashboard, community page, or both.
        chosen = next(
            (
                item["destinations"]
                for item in decisions
                if item.get("role") == ReviewRole.EDITOR_IN_CHIEF.value and item.get("destinations")
            ),
            None,
        )
        if chosen:
            pitch.publish_to = [d for d in ("dashboard", "community") if d in chosen]
            self.store.put(pitch)
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

    def feed(self, community_id: str, destination: str | None = None) -> list[Publication]:
        # The database filters by community; only the destination is checked here.
        pubs = [
            p
            for p in self.store.find(Publication, community_id=community_id)
            if not destination or destination in p.destinations
        ]
        return sorted(pubs, key=lambda p: p.published_at, reverse=True)

    def public_feed(
        self, community_id: str, destination: str | None = None
    ) -> list[dict[str, Any]]:
        """What a community member sees: the story, who approved it, its sources, the AI
        disclosure and any published corrections. Internal drafts never appear here.

        Everything the page needs is read in a handful of batch queries (stories, approvals,
        reviewers, corrections), not one query per story, so it stays fast on a remote database."""

        community = self.communities[community_id]
        # A story shared with this community (one story per event) may carry approvals from
        # the trusted voices of the community it was written for; this community's come first.
        voices = {v.id: v for c in self.communities.values() for v in c.trusted_voices}
        voices.update({v.id: v for v in community.trusted_voices})
        publications = self.feed(community_id, destination)

        stories = self.store.get_many(Story, (p.story_id for p in publications))
        approvals = self.store.get_many(
            ReviewDecision, (a for p in publications for a in p.approval_ids)
        )
        members = self.store.get_many(
            Member,
            (
                d.reviewer.removeprefix("member:")
                for d in approvals.values()
                if d.reviewer.startswith("member:")
            ),
        )
        published_corrections: dict[str, list[Correction]] = {}
        for c in self.store.find_in(Correction, "story_id", stories.keys(), status="published"):
            published_corrections.setdefault(c.story_id, []).append(c)

        items: list[dict[str, Any]] = []
        for publication in publications:
            story = stories.get(publication.story_id)
            if story is None:
                continue
            approvers = []
            for approval_id in publication.approval_ids:
                approval = approvals.get(approval_id)
                if approval is None:
                    continue
                voice = voices.get(approval.reviewer)
                member = (
                    members.get(approval.reviewer.removeprefix("member:"))
                    if approval.reviewer.startswith("member:")
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
                for c in published_corrections.get(story.id, [])
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

    def topics_export(self) -> list[dict[str, Any]]:
        """The active topics, with their parents and related topics, for the reader app."""

        self.ensure_seeded()
        return [
            {
                "id": t.id,
                "name": t.name,
                "parent_id": t.parent_id,
                "keywords": t.keywords,
                "description": t.description,
                "branches": t.branches,
            }
            for t in self.store.find(Topic)
            if t.status == "active"
        ]

    def dashboard_export(self) -> dict[str, Any]:
        """Everything the reader app needs to show published stories on its dashboard: the
        topic tree (with branches) and every story sent to the dashboard, once per language."""

        stats = self.engagement.all_stats()
        topics = self.topics_export()
        stories: list[dict[str, Any]] = []
        # Everything is read in batches (stories, topic links, translations), not per story.
        publications = [
            p for p in self.store.find(Publication) if "dashboard" in p.destinations
        ]
        found = self.store.get_many(Story, (p.story_id for p in publications))
        live = {
            sid: s
            for sid, s in found.items()
            if s.status in (StoryStatus.PUBLISHED, StoryStatus.CORRECTED)
        }
        links = self.engagement._links(live.values())
        translations_of: dict[str, dict[str, Translation]] = {}
        for t in self.store.find_in(Translation, "story_id", live.keys()):
            story_of = live[t.story_id]
            if t.story_version == story_of.version and t.body_md.strip():
                translations_of.setdefault(t.story_id, {})[t.language] = t
        # A story published to several communities (one story per event) is exported once,
        # in every language any of its publications carries.
        published_in: dict[str, list[Publication]] = {}
        for publication in publications:
            if publication.story_id in live:
                published_in.setdefault(publication.story_id, []).append(publication)
        for story_id, story_publications in published_in.items():
            story = live[story_id]
            publication = story_publications[0]
            topic_ids = links[story.id][1]
            translations = translations_of.get(story.id, {})
            versions = []
            for language in dict.fromkeys(
                r.language
                for p in story_publications
                for r in p.renditions
                if r.channel == "web"
            ):
                if language == story.language:
                    text = (story.headline, story.standfirst, story.body_md)
                elif language in translations:
                    t = translations[language]
                    text = (t.headline or story.headline, t.standfirst, t.body_md)
                else:
                    continue
                # Byte-sized on the dashboard in every language, older stories included.
                limit = self.settings.max_story_words
                standfirst = limit_words(text[1] or "", min(35, limit // 4))
                body = limit_words(text[2], max(limit - word_count(standfirst), limit // 2))
                versions.append(
                    {
                        "language": language,
                        "headline": text[0],
                        "standfirst": standfirst,
                        "body_md": body,
                    }
                )
            stories.append(
                {
                    "story_id": story.id,
                    "community_id": story.community_id,
                    "community_ids": list(
                        dict.fromkeys(p.community_id for p in story_publications)
                    ),
                    "published_at": publication.published_at,
                    "topic_ids": topic_ids,
                    "image_url": story.image_url,
                    "event_ids": story.event_ids,
                    "sources": [s.model_dump() for s in story.sources],
                    "engagement": stats.get(story.id) or self.engagement._stat(None),
                    "versions": versions,
                }
            )
        return {"topics": topics, "stories": stories}

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
