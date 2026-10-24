"""Community desk: turns messages from community members into newsroom work.

Moderation (JSN annex L1-L4 on public content) and every decision are plain code. The
model only drafts titles and answers. Principles:

* a tip is a LEAD, never evidence: it becomes a pitch that must be verified;
* a correction opens a correction case for a human editor (JSN 35-37);
* a question is answered only from published stories or editor-maintained community
  notes. If the answer is not there, a human gets the question; nothing is invented;
* local knowledge goes to an editor as a potential context note, attributed to the
  community;
* personal data never reaches a hosted model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent
from newsroom.domain import (
    Community,
    CommunityMessage,
    CommunityTriage,
    Correction,
    Pitch,
    Story,
    StoryStatus,
    UserNeed,
    new_id,
)
from newsroom.lexicon import words
from newsroom.llm.base import LLMError

# --------------------------------------------------------------------------- schemas


class TriageProposal(BaseModel):
    title: str = ""
    rationale: str = ""
    needs: list[UserNeed] = Field(default_factory=list)
    answer: str | None = None
    answer_source_ids: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------- moderation rules

_HATE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bkill (all|them|every)\b",
        r"\bdeath to\b",
        r"\bsubhuman\b",
        r"\bvermin\b",
        r"\bgas them\b",
        r"go back to (your|their) (country|home)",
        r"\btapetaan\b",
        r"\bpainukaa kotiin\b",
    )
]
_PRIVATE = [
    ("an e-mail address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("a phone number", re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d")),
    ("a Finnish personal identity code", re.compile(r"\b\d{6}[-+A]\d{3}[0-9A-Z]\b", re.I)),
]


def moderation_flags(text: str) -> tuple[str | None, str | None, list[str]]:
    """Return (hate reason, injection reason, personal-data kinds)."""

    hate = next((p.pattern for p in _HATE if p.search(text)), None)
    injection = guards.scan_injection(text)
    private = [name for name, pattern in _PRIVATE if pattern.search(text)]
    return (
        "hate or incitement against a group" if hate else None,
        f"text addressed to an AI: {injection[:3]}" if injection else None,
        private,
    )


# ----------------------------------------------------------------------- the agent


class CommunityDeskAgent(Agent):
    name = "community_desk"

    def triage(self, message: CommunityMessage, community: Community) -> CommunityTriage:
        result = self._decide(message, community)
        message.status = {
            "answer": "answered",
            "reject": "rejected",
        }.get(result.action, "triaged")
        self.store.put(message)
        if result.pitch:
            self.store.put(result.pitch)
        if result.correction:
            self.store.put(result.correction)
        self.trace(
            "triage",
            result.action,
            result.reason,
            message_id=message.id,
            kind=message.kind,
            community_id=community.id,
        )
        return result

    # ----------------------------------------------------------------- decisions

    def _decide(self, message: CommunityMessage, community: Community) -> CommunityTriage:
        hate, injection, private = moderation_flags(message.text)
        if hate:
            return CommunityTriage(
                action="reject",
                reason=f"Moderation (public content rules): {hate}.",
                reply="We cannot publish or act on this message. Please keep to respectful "
                "language. Nothing was shared.",
            )
        if injection:
            return CommunityTriage(
                action="needs_human",
                reason=f"Moderation: {injection}. A human must read this; it was not processed.",
            )
        if private:
            return CommunityTriage(
                action="needs_human",
                reason=f"Privacy check: the message contains {', '.join(private)}. A human "
                "must decide what may be used (JSN 23-29).",
            )

        text = guards.minimise_personal_data(message.text)
        handlers = {
            "tip": self._tip,
            "correction": self._correction,
            "question": self._question,
            "local_knowledge": self._local_knowledge,
        }
        return handlers[message.kind](message, community, text)

    def _tip(self, message: CommunityMessage, community: Community, text: str) -> CommunityTriage:
        proposal = self._propose(
            "tip", {"text": text, "community": community.name}, text, community
        )
        needs = [n for n in (proposal.needs if proposal else []) if isinstance(n, UserNeed)]
        title = ((proposal.title if proposal else "") or text.split(".")[0])[:120].strip()
        risk, _ = guards.classify_risk(text, self.policy)
        pitch = Pitch(
            id=new_id("pitch"),
            community_id=community.id,
            article_ids=[],
            working_title=title or "Community tip",
            rationale=(
                "Community tip: a LEAD, not a fact. It is not evidence and must be verified "
                "against independent sources before anything is reported. "
                + ((proposal.rationale if proposal else "")[:300])
            ).strip(),
            needs=list(dict.fromkeys(needs)) or [UserNeed.COMMUNITY],
            score=0.0,
            risk_tier=risk,
            origin="community_tip",
        )
        return CommunityTriage(
            action="open_pitch",
            reason="Tip converted into a lead that requires verification.",
            reply="Thank you. An editor will look into this and check it with independent "
            "sources before anything is reported.",
            pitch=pitch,
        )

    def _correction(
        self, message: CommunityMessage, community: Community, text: str
    ) -> CommunityTriage:
        if not message.story_id:
            return CommunityTriage(
                action="needs_human",
                reason="Correction request without a story reference; an editor must find "
                "which story it concerns.",
            )
        story = self.store.get(Story, message.story_id)
        if story is None:
            return CommunityTriage(
                action="needs_human",
                reason=f"Correction refers to unknown story {message.story_id}.",
            )
        correction = Correction(
            id=new_id("corr"),
            story_id=story.id,
            description=text[:2000],
            reported_by=message.author_ref,
            status="open",
        )
        return CommunityTriage(
            action="open_correction",
            reason="Correction case opened for an editor to check (JSN 35-37).",
            reply="Thank you. An editor will check this. If it is a factual error we will "
            "correct it and say what was wrong.",
            correction=correction,
        )

    def _question(
        self, message: CommunityMessage, community: Community, text: str
    ) -> CommunityTriage:
        sources = self._answer_sources(community)
        if not sources:
            return CommunityTriage(
                action="needs_human",
                reason="No published story or community note to answer from.",
            )
        proposal = self._propose(
            "question",
            {"text": text, "sources": sources, "community": community.name},
            text,
            community,
            extra=sources,
        )
        valid = {s["id"]: s for s in sources}
        cited = [i for i in (proposal.answer_source_ids if proposal else []) if i in valid]
        answer = (proposal.answer or "").strip() if proposal else ""
        if not answer or not cited:
            return CommunityTriage(
                action="needs_human",
                reason="The answer is not covered by published stories or community notes; "
                "a human answers it. Nothing was invented.",
            )
        names = ", ".join(valid[i]["label"] for i in cited)
        return CommunityTriage(
            action="answer",
            reason=f"Answered only from: {names}.",
            reply=f"{answer}\n\nSource: {names}. {guards.ai_disclosure('en', self.policy)}",
        )

    def _local_knowledge(
        self, message: CommunityMessage, community: Community, text: str
    ) -> CommunityTriage:
        return CommunityTriage(
            action="needs_human",
            reason="Local knowledge: route to an editor as a potential context note, attributed "
            "to the community and checked before use.",
            reply="Thank you for sharing this. An editor will review it and, if it is used, "
            "credit it to the community.",
        )

    # ------------------------------------------------------------------ helpers

    def _answer_sources(self, community: Community) -> list[dict[str, str]]:
        sources = []
        for story in self.store.find(Story, community_id=community.id):
            if story.status in (StoryStatus.PUBLISHED, StoryStatus.CORRECTED):
                sources.append(
                    {
                        "id": story.id,
                        "label": story.headline,
                        "text": f"{story.standfirst} {story.body_md}"[:1500],
                    }
                )
        for index, note in enumerate(community.cultural_context):
            sources.append(
                {"id": f"note:{index}", "label": "community notes kept by editors", "text": note}
            )
        return sources

    def _propose(
        self,
        mode: str,
        payload: dict[str, Any],
        text: str,
        community: Community,
        extra: list[dict[str, str]] | None = None,
    ) -> TriageProposal | None:
        blocks = "\n\n".join(
            f"[{s['id']}] {s['label']}\n{guards.wrap_untrusted(s['text'], s['id'])}"
            for s in (extra or [])
        )
        system = (
            f"You help a community newsroom triage a community {mode}. For a tip, propose a "
            "neutral working title and why an editor should check it. For a question, answer "
            "ONLY using the listed sources and list the ids you used; if they do not contain "
            "the answer, leave the answer empty. Never invent facts. " + guards.UNTRUSTED_PREAMBLE
        )
        prompt = (
            f"Community: {community.name}\nMessage:\n{guards.wrap_untrusted(text, 'message')}"
            + (f"\n\nSources:\n{blocks}" if blocks else "")
        )
        try:
            return self.llm.generate(
                task="triage_message",
                system=system,
                prompt=prompt,
                schema=TriageProposal,
                payload={**payload, "mode": mode},
            )
        except LLMError as exc:
            self.trace("triage", "skipped", f"model unavailable: {str(exc)[:160]}")
            return None


# ------------------------------------------------------------------ offline handler


def _fake_triage(payload: dict[str, Any]) -> TriageProposal:
    text = payload.get("text", "")
    if payload.get("mode") == "question":
        wanted = {w for w in words(text) if len(w) >= 4}
        best, best_overlap = None, 0
        for source in payload.get("sources", []):
            overlap = len(wanted & {w for w in words(source["text"]) if len(w) >= 4})
            if overlap > best_overlap:
                best, best_overlap = source, overlap
        if best and best_overlap >= 2:
            return TriageProposal(answer=best["text"].strip()[:400], answer_source_ids=[best["id"]])
        return TriageProposal()
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return TriageProposal(
        title=first[:100],
        rationale="Worth checking because it may affect people in the community. [offline stand-in]",
        needs=[UserNeed.COMMUNITY, UserNeed.USEFULNESS],
    )


FAKE_HANDLERS: dict[str, Callable[[dict], BaseModel]] = {"triage_message": _fake_triage}
