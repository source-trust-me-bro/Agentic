"""The story workstream as a LangGraph graph.

    investigate ──(insufficient, rounds left)──► investigate
         │ (sufficient or bounds reached)
         ▼
      explain ─► translate ─► standards ──(blocking, rounds left)──► revise ─► explain
                                  │ (clean, or rounds exhausted)
                                  ▼
                             human_gate  ◄── INTERRUPT: named humans review this version
                                  │
          ┌────────── reject ─────┼──── request changes ──► revise ─► explain
          ▼                       ▼ (all required roles approved, nothing blocking)
        rejected               publish ─► done

Design points:

* LangGraph owns the workflow and the checkpointed run state. The store owns the durable
  facts. State holds ids and small values; agents read and write the store.
* Loops are bounded (``max_investigation_rounds``, ``max_revision_rounds``) and every
  routing decision is traced with its reason.
* ``human_gate`` calls ``interrupt``. Nothing before the interrupt has side effects, so a
  resume after a restart is safe. Publication is possible only through ``can_publish``.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from newsroom.agents.base import AgentContext
from newsroom.agents.distributor import DistributorAgent
from newsroom.agents.explainer import ExplainerAgent
from newsroom.agents.investigator import InvestigatorAgent
from newsroom.agents.librarian import SourceLibrarianAgent
from newsroom.agents.standards import StandardsEditorAgent
from newsroom.agents.translator import TranslatorAgent
from newsroom.domain import (
    Article,
    Claim,
    Community,
    Decision,
    InvestigationResult,
    Member,
    Pitch,
    ReviewDecision,
    Story,
    StoryStatus,
    TraceEvent,
    Translation,
    new_id,
)
from newsroom.hitl import (
    approvals_for,
    can_publish,
    latest_report,
    missing_roles,
    record_decisions,
)
from newsroom.parallel import run_parallel


class StoryState(TypedDict, total=False):
    run_id: str
    pitch_id: str
    community_id: str
    # investigation loop
    investigation_round: int
    sufficient: bool
    search_hints: list[str]
    open_questions: list[str]
    claim_ids: list[str]
    # writing / revision loop
    revision_round: int
    human_revisions: int
    feedback: list[str]
    story_id: str
    story_version: int
    translation_ids: list[str]
    report_id: str
    blocking: bool
    # human gate
    gate_result: str  # publish | revise | rejected | wait | blocked
    missing_roles: list[str]
    # result
    outcome: str  # "" | published | rejected | blocked
    publication_id: str


class StoryWorkflow:
    """Builds and runs the story graph."""

    def __init__(self, ctx: AgentContext, communities: dict[str, Community], checkpointer: Any):
        self.ctx = ctx
        self.store = ctx.store
        self.communities = communities
        self.investigator = InvestigatorAgent(ctx)
        self.explainer = ExplainerAgent(ctx)
        self.librarian = SourceLibrarianAgent(ctx)
        self.translator = TranslatorAgent(ctx)
        self.standards = StandardsEditorAgent(ctx)
        self.distributor = DistributorAgent(ctx)
        self.graph = self._build().compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------ helpers

    def _trace(self, state: StoryState, step: str, decision: str, reason: str = "", **refs: Any):
        self.store.trace(
            TraceEvent(
                id=new_id("trace"),
                run_id=state.get("run_id", self.ctx.run_id),
                agent="orchestrator",
                step=step,
                decision=decision,
                reason=reason,
                refs=refs,
            )
        )

    def _community(self, state: StoryState) -> Community:
        return self.communities[state["community_id"]]

    def _story(self, state: StoryState) -> Story:
        story = self.store.get(Story, state["story_id"])
        if story is None:
            raise RuntimeError(f"Story {state['story_id']} is missing from the store.")
        return story

    # -------------------------------------------------------------------- nodes

    def investigate(self, state: StoryState) -> StoryState:
        pitch = self.store.get(Pitch, state["pitch_id"])
        assert pitch is not None
        round_no = state.get("investigation_round", 0) + 1
        prior = None
        if round_no > 1:
            prior = InvestigationResult(
                claims=[],
                sufficient=False,
                search_hints=state.get("search_hints", []),
                open_questions=state.get("open_questions", []),
                round_no=round_no - 1,
            )
        result = self.investigator.investigate(pitch, round_no=round_no, prior=prior)
        return {
            "investigation_round": round_no,
            "sufficient": result.sufficient,
            "search_hints": result.search_hints,
            "open_questions": result.open_questions,
            "claim_ids": [c.id for c in result.claims],
        }

    def after_investigate(self, state: StoryState) -> str:
        limit = self.ctx.settings.max_investigation_rounds
        if state.get("sufficient"):
            self._trace(state, "investigation", "stop", "Evidence is sufficient.")
            return "explain"
        if state.get("investigation_round", 1) >= limit:
            self._trace(
                state,
                "investigation",
                "stop",
                f"Round limit {limit} reached; open questions stay visible to editors.",
                open_questions=state.get("open_questions", []),
            )
            return "explain"
        self._trace(
            state,
            "investigation",
            "continue",
            "Evidence insufficient; searching again with new hints.",
            hints=state.get("search_hints", []),
        )
        return "investigate"

    def explain(self, state: StoryState) -> StoryState:
        pitch = self.store.get(Pitch, state["pitch_id"])
        assert pitch is not None
        community = self._community(state)
        claims = [c for cid in state.get("claim_ids", []) if (c := self.store.get(Claim, cid))]
        article_ids = list(pitch.article_ids)
        for claim in claims:
            for link in claim.evidence:
                if link.article_id not in article_ids:
                    article_ids.append(link.article_id)
        articles = [a for aid in article_ids if (a := self.store.get(Article, aid))]
        version = state.get("story_version", 0) + 1
        story = self.explainer.write(
            pitch,
            claims,
            community,
            articles,
            feedback=state.get("feedback") or None,
            version=version,
        )
        story.status = StoryStatus.DRAFT
        self.store.put(story)
        return {"story_id": story.id, "story_version": story.version, "feedback": []}

    def attach_sources(self, state: StoryState) -> StoryState:
        """The basic official reference is attached by the newsroom itself; editors then
        review it and add more through the review desk."""

        story = self._story(state)
        self.librarian.attach(story, self._community(state))
        return {}

    def _targets(self, state: StoryState) -> list[Community]:
        """The community the story is written for, then every other community that picked
        the same event (``Pitch.also_for``). One story is published to all of them."""

        pitch = self.store.get(Pitch, state["pitch_id"])
        also_for = pitch.also_for if pitch else []
        others = [self.communities[c] for c in also_for if c in self.communities]
        return [self._community(state), *others]

    def translate(self, state: StoryState) -> StoryState:
        community = self._community(state)
        story = self._story(state)
        # Every language any target community reads, each once.
        codes: dict[str, Any] = {}
        for target in self._targets(state):
            for lane in target.languages:
                codes.setdefault(lane.code, lane)
        lanes = [lane for code, lane in codes.items() if code != story.language]
        if self.ctx.settings.translate_only_read_languages:
            readers = {
                code
                for m in self.store.find(Member)
                for code in (m.language, *m.reading_languages)
            }
            readers |= set(self.ctx.settings.always_translate)
            lanes = [lane for lane in lanes if lane.code in readers]
        progress = self.ctx.progress
        pitch_id = state.get("pitch_id", "")
        progress.story_languages(pitch_id, total=len(lanes))

        def translate_one(lane: Any) -> Translation:
            translation = self.translator.translate(story, lane, community)
            progress.story_languages(pitch_id, add_done=1, language=lane.label)
            return translation

        # Each language is its own model call, so they run side by side (order is kept).
        translations = run_parallel(
            translate_one, lanes, self.ctx.settings.translation_workers, "translate"
        )
        return {"translation_ids": [t.id for t in translations]}

    def standards_review(self, state: StoryState) -> StoryState:
        community = self._community(state)
        story = self._story(state)
        claims = [c for cid in state.get("claim_ids", []) if (c := self.store.get(Claim, cid))]
        translations = [
            t for tid in state.get("translation_ids", []) if (t := self.store.get(Translation, tid))
        ]
        report = self.standards.review(story, claims, translations, community)
        story.status = StoryStatus.IN_REVIEW
        self.store.put(story)
        blocking = bool(report.blocking_failures)
        return {
            "report_id": report.id,
            "blocking": blocking,
            "feedback": list(report.revision_feedback),
        }

    def after_standards(self, state: StoryState) -> str:
        limit = self.ctx.settings.max_revision_rounds
        if state.get("blocking") and state.get("revision_round", 0) < limit:
            self._trace(
                state,
                "standards",
                "revise",
                "Blocking standards failures; sending feedback back to the explainer.",
                feedback=state.get("feedback", []),
            )
            return "revise"
        self._trace(
            state,
            "standards",
            "to_human",
            "Blocking failures remain; editors must resolve them."
            if state.get("blocking")
            else "Standards review passed; asking humans to review.",
        )
        return "human_gate"

    def revise(self, state: StoryState) -> StoryState:
        story = self._story(state)
        story.status = StoryStatus.CHANGES_REQUESTED
        self.store.put(story)
        return {"revision_round": state.get("revision_round", 0) + 1}

    def human_gate(self, state: StoryState) -> StoryState:
        story = self._story(state)
        community = self._community(state)
        report = latest_report(self.store, story.id, story.version)
        translations = [
            t for tid in state.get("translation_ids", []) if (t := self.store.get(Translation, tid))
        ]
        # Everything above this line must be free of side effects: a resume re-runs it.
        answer = interrupt(
            {
                "type": "human_review",
                "pitch_id": state["pitch_id"],
                "community_id": state["community_id"],
                "story_id": story.id,
                "standfirst": story.standfirst,
                "sources": [
                    {
                        "name": ref.name,
                        "url": ref.url,
                        "official": ref.official,
                        "origin": ref.origin,
                    }
                    for ref in story.sources
                ],
                "context_needing_a_human": [
                    n.text for n in story.context_notes if n.kind == "human_needed"
                ],
                "story_version": story.version,
                "community": community.name,
                "headline": story.headline,
                "risk_tier": story.risk_tier.value,
                "missing_roles": [r.value for r in missing_roles(self.store, story)],
                "blocking_checks": [c.name for c in (report.blocking_failures if report else [])],
                "translations": {t.language: t.status for t in translations},
                "instructions": (
                    "Submit decisions: [{role, reviewer, decision, notes}]. A named human "
                    "must approve in every missing role before this can publish."
                ),
            }
        )
        story = self._story(state)  # may have changed while paused (e.g. a source was attached)
        self._refresh_review(state, story, community)
        decisions = (answer or {}).get("decisions", []) if isinstance(answer, dict) else []
        recorded: list[ReviewDecision] = record_decisions(self.store, story, community, decisions)
        for decision in recorded:
            self._trace(
                state,
                "human_review",
                decision.decision.value,
                decision.notes or "No notes.",
                role=decision.role.value,
                reviewer=decision.reviewer,
                story_id=story.id,
                version=story.version,
            )

        all_decisions = approvals_for(self.store, story.id, story.version)
        if any(d.decision is Decision.REJECT for d in all_decisions):
            return {"gate_result": "rejected", "outcome": "rejected"}
        change_notes = [
            d.notes or f"{d.role.value} requested changes"
            for d in all_decisions
            if d.decision is Decision.REQUEST_CHANGES
        ]
        if change_notes:
            human_limit = self.ctx.settings.max_human_revision_rounds
            used = state.get("human_revisions", 0)
            if used < human_limit:
                return {
                    "gate_result": "revise",
                    "feedback": change_notes,
                    "human_revisions": used + 1,
                }
            return {"gate_result": "blocked", "outcome": "blocked", "feedback": change_notes}
        allowed, _reasons = can_publish(self.store, story)
        if allowed:
            return {"gate_result": "publish"}
        # Blocking checks or missing approvals: keep the story waiting. An editor can attach
        # the missing source, add approvals, request changes, or reject it.
        return {
            "gate_result": "wait",
            "missing_roles": [r.value for r in missing_roles(self.store, story)],
        }

    def _refresh_review(self, state: StoryState, story: Story, community: Community) -> None:
        """Re-run the standards review if the story changed after its last review."""

        report = latest_report(self.store, story.id, story.version)
        if report is not None and story.updated_at <= report.created_at:
            return
        claims = [c for cid in state.get("claim_ids", []) if (c := self.store.get(Claim, cid))]
        translations = [
            t for tid in state.get("translation_ids", []) if (t := self.store.get(Translation, tid))
        ]
        self.standards.review(story, claims, translations, community)
        self._trace(state, "standards", "re_reviewed", "Story changed while in review.")

    def after_gate(self, state: StoryState) -> str:
        result = state.get("gate_result", "wait")
        self._trace(state, "human_gate", result, f"Gate outcome: {result}.")
        return {
            "publish": "publish",
            "revise": "revise",
            "rejected": "finish",
            "blocked": "finish",
            "wait": "human_gate",
        }[result]

    def publish(self, state: StoryState) -> StoryState:
        story = self._story(state)
        community = self._community(state)
        allowed, reasons = can_publish(self.store, story)
        if not allowed:  # defence in depth: the graph never publishes on its own authority
            self._trace(state, "publish", "refused", "; ".join(reasons))
            return {"outcome": "blocked"}
        translations = [
            t for tid in state.get("translation_ids", []) if (t := self.store.get(Translation, tid))
        ]
        approvals = [
            d
            for d in approvals_for(self.store, story.id, story.version)
            if d.decision is Decision.APPROVE
        ]
        publication = self.distributor.render(story, translations, community, approvals)
        # The same story goes to every other community that picked this event.
        for other in self._targets(state)[1:]:
            self.distributor.render(story, translations, other, approvals)
        self._trace(
            state,
            "publish",
            "published",
            "Approved by named humans in every required role.",
            publication_id=publication.id,
            approvals=[d.reviewer for d in approvals],
        )
        return {"outcome": "published", "publication_id": publication.id}

    def _reports(self, step: str, node: Any) -> Any:
        """Tell the editor which step this story has reached, then run the node."""

        def run(state: StoryState) -> StoryState:
            self.ctx.progress.story_step(state.get("pitch_id", ""), step)
            return node(state)

        return run

    # -------------------------------------------------------------------- graph

    def _build(self) -> StateGraph:
        graph = StateGraph(StoryState)
        graph.add_node("investigate", self._reports("investigate", self.investigate))
        graph.add_node("explain", self._reports("explain", self.explain))
        graph.add_node("attach_sources", self.attach_sources)
        graph.add_node("translate", self._reports("translate", self.translate))
        graph.add_node("standards", self._reports("standards", self.standards_review))
        graph.add_node("revise", self.revise)
        graph.add_node("human_gate", self._reports("review", self.human_gate))
        graph.add_node("publish", self.publish)

        graph.add_edge(START, "investigate")
        graph.add_conditional_edges(
            "investigate",
            self.after_investigate,
            {"investigate": "investigate", "explain": "explain"},
        )
        graph.add_edge("explain", "attach_sources")
        graph.add_edge("attach_sources", "translate")
        graph.add_edge("translate", "standards")
        graph.add_conditional_edges(
            "standards", self.after_standards, {"revise": "revise", "human_gate": "human_gate"}
        )
        graph.add_edge("revise", "explain")
        graph.add_conditional_edges(
            "human_gate",
            self.after_gate,
            {"publish": "publish", "revise": "revise", "finish": END, "human_gate": "human_gate"},
        )
        graph.add_edge("publish", END)
        return graph

    # ---------------------------------------------------------------------- api

    @staticmethod
    def config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(self, pitch: Pitch) -> dict[str, Any]:
        """Run the story until it finishes or pauses at the human gate."""

        state: StoryState = {
            "run_id": self.ctx.run_id,
            "pitch_id": pitch.id,
            "community_id": pitch.community_id,
        }
        return self.graph.invoke(state, self.config(pitch.id))

    def resume(self, thread_id: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        from langgraph.types import Command

        return self.graph.invoke(Command(resume={"decisions": decisions}), self.config(thread_id))

    def pending(self, thread_id: str) -> dict[str, Any] | None:
        """The human-review request a paused run is waiting on, if any."""

        snapshot = self.graph.get_state(self.config(thread_id))
        for task in snapshot.tasks:
            for item in task.interrupts:
                return dict(item.value)
        return None

    def values(self, thread_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self.config(thread_id)).values)
