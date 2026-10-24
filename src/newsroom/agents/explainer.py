"""Explainer (reporter/writer): turns verified claims into an explanatory story.

The model writes the narrative. Plain Python then enforces the editorial rules the model
cannot be trusted with:

* only VERIFIED and ATTRIBUTED claims are stated, ATTRIBUTED ones with their attribution;
* DISPUTED claims are shown two-sided, UNKNOWN claims are never stated as fact;
* context that does not come from the articles must be grounded (official source with a
  URL, or an editor-maintained community note) or it becomes ``human_needed``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent, AgentContext
from newsroom.domain import (
    Article,
    Claim,
    ClaimStatus,
    Community,
    ContextNote,
    Pitch,
    RiskTier,
    SourceRef,
    Story,
    UserNeed,
    new_id,
    utcnow,
)
from newsroom.langdetect import LANGUAGE_NAMES, detect_language
from newsroom.llm.base import LLMError
from newsroom.registry import load_sources

_RISK_ORDER = {RiskTier.LOW: 0, RiskTier.ELEVATED: 1, RiskTier.HIGH: 2}


def word_count(text: str) -> int:
    """Words a reader sees: markdown symbols do not count."""

    return len(re.findall(r"[^\s#*_`>|-]+", text or ""))


def limit_words(text: str, limit: int) -> str:
    """Keep whole paragraphs, then whole sentences, until ``limit`` words. Never cuts a
    sentence in half, so the story still reads cleanly."""

    if word_count(text) <= limit:
        return text
    kept: list[str] = []
    used = 0
    for block in re.split(r"\n\s*\n", text.strip()):
        size = word_count(block)
        if used + size <= limit:
            kept.append(block)
            used += size
            continue
        sentences = re.split(r"(?<=[.!?])\s+", block)
        partial: list[str] = []
        for sentence in sentences:
            if used + word_count(sentence) > limit:
                break
            partial.append(sentence)
            used += word_count(sentence)
        # Drop a heading or list fragment left without its content.
        if partial and not re.match(r"^\s*(#|[-*] )", " ".join(partial)):
            kept.append(" ".join(partial))
        break
    return "\n\n".join(kept).strip()

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has",
    "have", "had", "its", "their", "will", "can", "not", "but", "into", "than", "over",
    "also", "been", "about", "which", "while", "when", "what", "who", "how", "they",
    "your", "you", "our", "may", "more", "most", "some", "any", "per",
}  # fmt: skip

ATTRIBUTION_MARKERS = (
    "says", "said", "according to", "reports", "reported", "told", "announced",
    "states", "stated", "claims", "claimed", "notes", "published",
)  # fmt: skip
HEDGE_MARKERS = (
    "not yet known", "unknown", "unclear", "no confirmation", "unconfirmed", "we don't know",
    "we do not know", "not confirmed", "not yet confirmed", "remains to be", "is not known",
    "has not been confirmed", "cannot confirm", "can't confirm",
)  # fmt: skip
DISPUTE_MARKERS = (
    "disput", "differ", "disagree", "contradict", "conflicting", "other sources",
    "however", "otherwise", "say otherwise", "but ", "while ",
)  # fmt: skip
UNKNOWN_HEADING = "what we don't know yet"


# ------------------------------------------------------------------------------ schemas


class ContextNoteDraft(BaseModel):
    text: str
    kind: Literal["from_story", "official_source", "community_knowledge", "human_needed"] = (
        "human_needed"
    )
    source_url: str | None = None


class StoryDraft(BaseModel):
    headline: str
    standfirst: str
    body_md: str
    needs: list[UserNeed] = Field(default_factory=list)
    context_notes: list[ContextNoteDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------- helpers


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.casefold()) if len(t) >= 3 and t not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _mentions(sentence: str, claim: Claim) -> bool:
    wanted = _tokens(claim.text)
    if len(wanted) < 3:
        return claim.text.casefold().strip(" .") in sentence.casefold()
    return len(wanted & _tokens(sentence)) / len(wanted) >= 0.7


def _has(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in markers)


def _attribution_name(claim: Claim) -> str:
    name = (claim.attribution or "").strip()
    name = re.sub(r"\b(says?|said|reports?|announced|told)\b\s*$", "", name, flags=re.I).strip()
    return name.rstrip(",:") or "the source"


def _lower_first(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].lower() + text[1:] if text else text


def _names(claim: Claim, relation: str) -> str:
    seen: list[str] = []
    for link in claim.evidence:
        if link.relation == relation and link.source_name not in seen:
            seen.append(link.source_name)
    return " and ".join(seen)


def render_claim_statement(claim: Claim) -> str | None:
    """The only wording the code itself uses to state a claim, per status."""

    text = claim.text.strip().rstrip(".")
    if claim.status is ClaimStatus.VERIFIED:
        return f"{text}."
    if claim.status is ClaimStatus.ATTRIBUTED:
        return f"According to {_attribution_name(claim)}, {_lower_first(text)}."
    if claim.status is ClaimStatus.DISPUTED:
        supporters = _names(claim, "supports") or _attribution_name(claim)
        opponents = _names(claim, "contradicts") or "other sources"
        return (
            f"Sources disagree: {supporters} report that {_lower_first(text)}, "
            f"but {opponents} report otherwise."
        )
    return None


def verify_story_claims(story_text: str, claims: list[Claim]) -> list[str]:
    """Return problems where the story states a claim more strongly than the evidence allows."""

    problems: list[str] = []
    hedged_section = False
    sentences: list[tuple[str, bool]] = []  # (sentence, inside the unknowns section)
    for line in story_text.splitlines():
        stripped = line.strip()
        if UNKNOWN_HEADING in line.casefold():
            hedged_section = True
            continue
        if hedged_section and stripped and not stripped.startswith(("-", "*", "•")):
            hedged_section = False  # prose after the list is no longer part of the section
        for sentence in _sentences(line):
            sentences.append((sentence, hedged_section))
    paragraphs = [p for p in re.split(r"\n\s*\n", story_text) if p.strip()]

    for claim in claims:
        if claim.status is ClaimStatus.UNKNOWN:
            for sentence, in_unknowns in sentences:
                if (
                    _mentions(sentence, claim)
                    and not in_unknowns
                    and not _has(sentence, HEDGE_MARKERS)
                ):
                    problems.append(f"UNKNOWN claim stated as fact: {claim.text!r}")
                    break
        elif claim.status is ClaimStatus.ATTRIBUTED:
            name_tokens = _tokens(claim.attribution or "")
            for index, (sentence, _) in enumerate(sentences):
                if not _mentions(sentence, claim):
                    continue
                window = " ".join(s for s, _ in sentences[max(0, index - 1) : index + 1])
                named = bool(name_tokens) and len(name_tokens & _tokens(window)) >= max(
                    1, len(name_tokens) // 2
                )
                if not (_has(window, ATTRIBUTION_MARKERS) or named):
                    problems.append(f"ATTRIBUTED claim without attribution: {claim.text!r}")
                    break
        elif claim.status is ClaimStatus.DISPUTED:
            opponents = [
                link.source_name for link in claim.evidence if link.relation == "contradicts"
            ]
            for paragraph in paragraphs:
                if not any(_mentions(s, claim) for s in _sentences(paragraph)):
                    continue
                two_sided = _has(paragraph, DISPUTE_MARKERS) and (
                    not opponents or any(o.casefold() in paragraph.casefold() for o in opponents)
                )
                if not two_sided:
                    problems.append(f"DISPUTED claim presented one-sidedly: {claim.text!r}")
                    break
    return problems


def _strip_sentences(body: str, claims: list[Claim]) -> str:
    """Drop sentences that state the given claims, keeping markdown lines intact."""

    kept_lines = []
    for line in body.splitlines():
        if line.lstrip().startswith("#") or not line.strip():
            kept_lines.append(line)
            continue
        kept = [s for s in _sentences(line) if not any(_mentions(s, c) for c in claims)]
        if kept:
            prefix = re.match(r"^\s*(?:[-*>]\s+)?", line)
            kept_lines.append((prefix.group(0) if prefix else "") + " ".join(kept))
    return "\n".join(kept_lines)


def _max_risk(*tiers: RiskTier) -> RiskTier:
    return max(tiers, key=lambda tier: _RISK_ORDER[tier])


# ------------------------------------------------------------------------------ agent


class ExplainerAgent(Agent):
    name = "explainer"

    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(ctx)

    # -- public ------------------------------------------------------------------

    def write(
        self,
        pitch: Pitch,
        claims: list[Claim],
        community: Community,
        articles: list[Article],
        *,
        feedback: list[str] | None = None,
        version: int = 1,
    ) -> Story:
        feedback = list(feedback or [])
        if feedback:
            self.trace("revise", "revision requested", "; ".join(feedback)[:500], pitch_id=pitch.id)

        draft: StoryDraft | None = None
        problems: list[str] = []
        language_fix: list[str] = []
        try:
            for attempt in range(2):
                draft = self._draft(
                    pitch, claims, community, articles, feedback + problems + language_fix, attempt
                )
                problems = verify_story_claims(self._text(draft), claims)
                # The model sometimes answers in the language of its sources instead of English.
                written = detect_language(self._text(draft))
                language_fix = (
                    [
                        "Write the whole story in English. The last draft was in "
                        f"{LANGUAGE_NAMES.get(written, written)}."
                    ]
                    if written not in (None, "en")
                    else []
                )
                if not problems and not language_fix:
                    break
                self.trace(
                    "verify", "model draft needs another pass",
                    "; ".join(problems + language_fix)[:500], attempt=attempt + 1,
                )  # fmt: skip
        except LLMError as exc:
            self.trace("draft", "model unavailable", str(exc)[:300], pitch_id=pitch.id)
            return self._failed_story(pitch, claims, community, articles, version, str(exc))

        assert draft is not None
        body = draft.body_md
        if problems:
            body = self._repair(body, claims)
            self.trace("repair", "code repaired claim wording", "; ".join(problems)[:500])

        notes = self._ground_notes(draft.context_notes, articles, community)
        body = self._finish_body(body, claims, notes)
        headline = re.sub(r"[#*_`]+", "", draft.headline).strip() or pitch.working_title
        # Byte-sized: the standfirst is the gist, and the whole story stays under the limit.
        limit = self.settings.max_story_words
        standfirst = limit_words(draft.standfirst.strip(), min(35, limit // 4))
        body_before = word_count(body)
        body = limit_words(body, max(limit - word_count(standfirst), limit // 2))
        if word_count(body) < body_before:
            self.trace(
                "length", "trimmed", f"body cut from {body_before} to {word_count(body)} words"
            )

        remaining = verify_story_claims(f"{headline}\n\n{standfirst}\n\n{body}", claims)
        if remaining:
            # Still not clean: do not publish a draft that overstates evidence.
            notes.append(
                ContextNote(
                    text="Automatic checks found claims worded more strongly than the evidence: "
                    + "; ".join(remaining),
                    kind="human_needed",
                )
            )
            self.trace("verify", "claim problems remain", "; ".join(remaining)[:500])

        full_text = f"{headline}\n{standfirst}\n{body}"
        # Never file text under the wrong language: if it is still not English, say what it is.
        written_in = detect_language(full_text) or "en"
        if written_in != "en":
            notes.append(
                ContextNote(
                    text=f"The draft came out in {LANGUAGE_NAMES.get(written_in, written_in)}, "
                    "not English. It is filed under its real language and needs a person to "
                    "check it or rewrite it in English.",
                    kind="human_needed",
                )
            )
            self.trace("language", "draft not in English", written_in, pitch_id=pitch.id)
        story = Story(
            id=new_id("story"),
            pitch_id=pitch.id,
            community_id=community.id,
            version=version,
            language=written_in,
            headline=headline,
            standfirst=standfirst,
            body_md=body,
            needs=self._needs(draft, pitch),
            risk_tier=_max_risk(pitch.risk_tier, guards.classify_risk(full_text, self.policy)[0]),
            claim_ids=[c.id for c in claims],
            sources=self._sources(articles, claims),
            context_notes=notes,
            ai_disclosure=guards.ai_disclosure(written_in, self.policy),
            image_url=_pick_image(articles),
            event_ids=list(dict.fromkeys(a.event_id for a in articles if a.event_id)),
            updated_at=utcnow(),
        )
        self.store.put(story)
        human_needed = sum(1 for n in notes if n.kind == "human_needed")
        self.trace(
            "write", "story drafted", f"{len(claims)} claims, {human_needed} items for a human",
            story_id=story.id, version=version, risk=story.risk_tier.value,
        )  # fmt: skip
        return story

    # -- model call ---------------------------------------------------------------

    def _draft(
        self,
        pitch: Pitch,
        claims: list[Claim],
        community: Community,
        articles: list[Article],
        feedback: list[str],
        attempt: int,
    ) -> StoryDraft:
        usable = [c for c in claims if c.status is not ClaimStatus.UNKNOWN or c.unknowns]
        payload: dict[str, Any] = {
            "pitch": {"title": pitch.working_title, "needs": [n.value for n in pitch.needs]},
            "community": {
                "name": community.name,
                "description": community.description,
                "cultural_context": community.cultural_context,
            },
            "claims": [
                {
                    "text": c.text,
                    "status": c.status.value,
                    "attribution": c.attribution,
                    "supporting": [e.source_name for e in c.evidence if e.relation == "supports"],
                    "contradicting": [
                        e.source_name for e in c.evidence if e.relation == "contradicts"
                    ],
                    "unknowns": c.unknowns,
                }
                for c in usable
            ],
            "articles": [
                {"title": a.title, "url": a.url, "official": a.official, "source": a.source_name}
                for a in articles
            ],
            "feedback": feedback,
            "attempt": attempt,
        }
        claim_lines = "\n".join(
            f"- [{c['status'].upper()}] {c['text']}"
            + (f" (attribution: {c['attribution']})" if c["attribution"] else "")
            for c in payload["claims"]
        )
        source_blocks = "\n".join(
            guards.wrap_untrusted(
                f"{a.title}\n{a.summary or ''}\n{(a.content or '')[:1500]}", a.source_name
            )
            for a in articles
        )
        system = (
            "You are the explanatory reporter of a community newsroom. Write for the community "
            "described below: explain what the news means for them, what to do next, and the "
            "cultural context they may lack. Do not merely translate. Rules: state only VERIFIED "
            "and ATTRIBUTED claims, attributing the latter by name; present DISPUTED claims "
            "two-sided; never state UNKNOWN claims as fact. Mark anything you cannot ground in "
            "the sources as kind=human_needed instead of inventing it. Serve at least one user "
            "need: information, understanding, usefulness, emotion, break, community.\n"
            f"Byte-sized: the standfirst is the gist in one sentence, and standfirst plus body "
            f"together must be UNDER {self.settings.max_story_words} words. Lead with what "
            "happened and what it means for the reader; use short paragraphs and no filler.\n"
            "Structure body_md as exactly two sections. First '## What happened': 2 to 3 short "
            "sentences of plain facts from the sources (who, what, when, where), no "
            "interpretation. Then '## What it means for you': the explanation for this "
            "community: why it matters, what to do next, and context they may lack.\n"
            "Always write in English, whatever the language of the sources or the community: "
            "the translators make every other language from your English text.\n"
            + guards.UNTRUSTED_PREAMBLE
        )
        prompt = (
            f"Community: {community.name}. {community.description}\n"
            f"Editor-maintained cultural context: {community.cultural_context}\n"
            f"Working title: {pitch.working_title}\nClaims and their status:\n{claim_lines}\n"
            f"Revision requests: {feedback or 'none'}\nSources:\n{source_blocks}"
        )
        return self.llm.generate(
            task="explainer.write_story",
            system=system,
            prompt=prompt,
            schema=StoryDraft,
            payload=payload,
        )

    # -- rule enforcement -----------------------------------------------------------

    @staticmethod
    def _text(draft: StoryDraft) -> str:
        return f"{draft.headline}\n\n{draft.standfirst}\n\n{draft.body_md}"

    def _repair(self, body: str, claims: list[Claim]) -> str:
        unknown = [c for c in claims if c.status is ClaimStatus.UNKNOWN]
        flagged = [
            c
            for c in claims
            if c.status in (ClaimStatus.ATTRIBUTED, ClaimStatus.DISPUTED)
            and verify_story_claims(body, [c])
        ]
        body = _strip_sentences(body, unknown + flagged)
        lines = [f"- {s}" for c in flagged if (s := render_claim_statement(c))]
        if lines:
            body = body.rstrip() + "\n\n## What the sources say\n" + "\n".join(lines)
        return body

    def _finish_body(self, body: str, claims: list[Claim], notes: list[ContextNote]) -> str:
        body = body.rstrip()
        community_notes = [n.text for n in notes if n.kind == "community_knowledge"]
        fresh = [t for t in community_notes if t.casefold() not in body.casefold()]
        if fresh:
            body += "\n\n" + "\n".join(f"> **Community note:** {t}" for t in fresh)

        gaps: list[str] = []
        for claim in claims:
            if claim.status is ClaimStatus.UNKNOWN:
                gaps.append(f"Not yet confirmed: {claim.text.strip().rstrip('.')}.")
            gaps.extend(u.strip() for u in claim.unknowns if u.strip())
        if gaps and UNKNOWN_HEADING not in body.casefold():
            body += "\n\n## What we don't know yet\n" + "\n".join(
                f"- {g}" for g in dict.fromkeys(gaps)
            )
        return body

    def _ground_notes(
        self, drafts: list[Any], articles: list[Article], community: Community
    ) -> list[ContextNote]:
        official_urls = {a.url for a in articles if a.official}
        article_tokens: set[str] = set()
        for article in articles:
            article_tokens |= _tokens(
                f"{article.title} {article.summary or ''} {article.content or ''}"
            )
        editor_notes = {n: _tokens(n) for n in community.cultural_context}

        notes: list[ContextNote] = []
        for draft in drafts:
            text = draft.text.strip()
            if not text:
                continue
            kind = draft.kind
            url: str | None = None
            if kind == "from_story":
                wanted = _tokens(text)
                if not wanted or len(wanted & article_tokens) / len(wanted) < 0.6:
                    kind = "human_needed"
            elif kind == "official_source":
                if draft.source_url and draft.source_url in official_urls:
                    url = draft.source_url
                else:
                    kind = "human_needed"
            elif kind == "community_knowledge":
                wanted = _tokens(text)
                match = next(
                    (
                        original
                        for original, tokens in editor_notes.items()
                        if wanted and len(wanted & tokens) / max(len(wanted), len(tokens)) >= 0.8
                    ),
                    None,
                )
                if match is None:
                    kind = "human_needed"
                else:
                    text = match  # use the editors' own wording
            if kind == "human_needed":
                self.trace("ground", "context not grounded", text[:200])
            note = ContextNote(text=text, kind=kind, source_url=url)
            if note not in notes:
                notes.append(note)
        return notes

    @staticmethod
    def _needs(draft: StoryDraft, pitch: Pitch) -> list[UserNeed]:
        needs = list(dict.fromkeys(draft.needs)) or list(dict.fromkeys(pitch.needs))
        return needs or [UserNeed.INFORMATION]

    def _sources(self, articles: list[Article], claims: list[Claim]) -> list[SourceRef]:
        """Every report the story rests on, each explained for the reader: who the source is,
        what it published, and what we could use of it. Official sources come first."""

        about = {s.id: s.about for s in load_sources() if s.about}
        by_id = {a.id: a for a in articles}
        refs: dict[str, SourceRef] = {
            a.url: SourceRef(
                name=a.source_name,
                url=a.url,
                official=a.official,
                note=explain_source(a, about.get(a.source_id)),
            )
            for a in articles
        }
        for claim in claims:
            for link in claim.evidence:
                if link.url in refs:
                    continue
                known = by_id.get(link.article_id) or self.store.get(Article, link.article_id)
                refs[link.url] = SourceRef(
                    name=link.source_name,
                    url=link.url,
                    official=bool(known and known.official),
                    note=(
                        explain_source(known, about.get(known.source_id))
                        if known
                        else "Used to check a fact in this story."
                    ),
                )
        return sorted(refs.values(), key=lambda r: not r.official)

    def _failed_story(
        self,
        pitch: Pitch,
        claims: list[Claim],
        community: Community,
        articles: list[Article],
        version: int,
        reason: str,
    ) -> Story:
        """Fail closed: a human must write this story. The empty body blocks approval."""

        story = Story(
            id=new_id("story"),
            pitch_id=pitch.id,
            community_id=community.id,
            version=version,
            headline=pitch.working_title,
            standfirst="",
            body_md="",
            needs=list(pitch.needs) or [UserNeed.INFORMATION],
            risk_tier=pitch.risk_tier,
            claim_ids=[c.id for c in claims],
            sources=self._sources(articles, claims),
            context_notes=[
                ContextNote(
                    text="Automatic drafting failed, so a human must write this story. "
                    f"Reason: {reason[:200]}",
                    kind="human_needed",
                )
            ],
            ai_disclosure=guards.ai_disclosure("en", self.policy),
            status="changes_requested",  # type: ignore[arg-type]
        )
        self.store.put(story)
        return story


def _pick_image(articles: list[Article]) -> str | None:
    """A linked photo for the story: official sources first, then full-access reports."""

    ranked = sorted(articles, key=lambda a: (not a.official, a.access != "full"))
    return next((a.image_url for a in ranked if a.image_url), None)


# ------------------------------------------------------------------- offline stand-in


def explain_source(article: Article, about: str | None = None) -> str:
    """One or two plain sentences telling the reader what this source is and what it said."""

    who = about or (
        f"{article.source_name} is an official body." if article.official
        else f"{article.source_name} is a news outlet."
    )  # fmt: skip
    title = article.title.strip().rstrip(".")
    if article.official:
        said = f"This is its own announcement: “{title}”."
    elif article.access == "metadata_only":
        said = (
            f"It reported “{title}”. The full story is paywalled, so the newsroom "
            "used only the public headline and summary."
        )
    else:
        said = f"It reported “{title}”."
    return f"{who} {said}"


def _fake_write_story(payload: dict[str, Any]) -> StoryDraft:
    title = payload["pitch"]["title"].strip().rstrip(".")
    community = payload["community"]
    claims = payload["claims"]
    articles = payload["articles"]
    feedback = payload.get("feedback") or []

    lines: list[str] = []
    for claim in claims:
        status = claim["status"]
        text = claim["text"].strip().rstrip(".")
        if status == "verified":
            lines.append(f"{text}.")
        elif status == "attributed":
            who = re.sub(
                r"\b(says?|said|reports?|announced|told)\b\s*$", "", claim["attribution"] or "",
                flags=re.I,
            ).strip() or "the source"  # fmt: skip
            lines.append(f"According to {who}, {_lower_first(text)}.")
        elif status == "disputed":
            sup = " and ".join(dict.fromkeys(claim["supporting"])) or "one source"
            con = " and ".join(dict.fromkeys(claim["contradicting"])) or "other sources"
            lines.append(
                f"Sources disagree: {sup} report that {_lower_first(text)}, but {con} report otherwise."
            )
        # unknown claims are never stated; the code adds an honest "not yet confirmed" line

    # Same two sections as a model-written story: the facts, then what they mean.
    body = ["## What happened"]
    body += lines or ["The facts behind this story are still being checked."]
    body += [
        "",
        "## What it means for you",
        f"For {community['name']}, this matters in practice: {title}.",
        "",
        "What to do next:",
        "- Read the original reports linked below before you act.",
        "- If this affects you directly, check the official source before you act.",
    ]
    if feedback:
        body += ["", f"(Revised after editor feedback: {len(feedback)} point(s) addressed.)"]

    notes: list[ContextNoteDraft] = []
    if community["cultural_context"]:
        notes.append(
            ContextNoteDraft(text=community["cultural_context"][0], kind="community_knowledge")
        )
    official = next((a for a in articles if a["official"]), None)
    if official:
        notes.append(
            ContextNoteDraft(
                text=f"{official['source']} is the official source for this topic.",
                kind="official_source",
                source_url=official["url"],
            )
        )
    notes.append(
        ContextNoteDraft(
            text="Check whether local rules or deadlines have changed since the source was published.",
            kind="human_needed",
        )
    )

    needs = list(dict.fromkeys([*payload["pitch"]["needs"], "understanding", "usefulness"]))
    return StoryDraft(
        headline=f"{title}: what it means for you",
        standfirst=f"What this means for {community['name']}, and what to do next.",
        body_md="\n".join(body),
        needs=needs,  # type: ignore[arg-type]
        context_notes=notes,
    )


def _lower_first_text(text: str) -> str:
    return _lower_first(text)


FAKE_HANDLERS: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
    "explainer.write_story": _fake_write_story,
}
