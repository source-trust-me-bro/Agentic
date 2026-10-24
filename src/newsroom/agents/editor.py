"""News editor: decides which reports become story pitches for one community.

Scoring, thresholds, diversity and risk are plain code. The model only proposes the
working title, the rationale and the user needs a story would serve. Reports that carry
instructions aimed at an AI are never pitched: they go to humans.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent
from newsroom.domain import Article, Community, Pitch, RiskTier, UserNeed, new_id
from newsroom.lexicon import (
    INTEREST_CONCEPTS,
    NATIONWIDE_INTERESTS,
    concepts,
    has_prefix_term,
    nouns_overlap,
    numbers,
    proper_nouns,
    words,
)
from newsroom.llm.base import LLMError
from newsroom.outlets import outlet_of


def is_relevant(hits: list[str], in_region: bool, official: bool) -> bool:
    """Gate before scoring.

    Local news must be about the community's own region (one interest hit is enough
    there). Nationwide official information (permits, benefits) needs two hits. Anything
    else needs three, so a story that merely mentions "workers" is not pitched.
    """

    return (bool(hits) and in_region) or (len(hits) >= 2 and official) or len(hits) >= 3


# --------------------------------------------------------------------------- schemas


class PitchProposal(BaseModel):
    candidate_id: str
    working_title: str
    rationale: str
    needs: list[UserNeed] = Field(default_factory=list)


class PitchProposals(BaseModel):
    proposals: list[PitchProposal] = Field(default_factory=list)


# ------------------------------------------------------------------ pure helpers


def _stem(word: str) -> str:
    return word[: max(4, len(word) - 2)]


def interest_hits(text: str, community: Community) -> list[str]:
    """Community interests that the text touches (Finnish or English wording)."""

    found = concepts(text)
    hits = []
    for interest in community.interests:
        mapped = INTEREST_CONCEPTS.get(interest.casefold(), [])
        if any(name in found for name in mapped) or has_prefix_term(text, interest):
            hits.append(interest)
    return hits


def region_hit(text: str, community: Community) -> bool:
    for region in community.regions:
        for word in words(region):
            if len(word) >= 4 and has_prefix_term(text, _stem(word)):
                return True
    return False


def infer_needs(text: str) -> list[UserNeed]:
    """Rough, deterministic reading of which reader needs a report could serve."""

    found = concepts(text)
    needs = [UserNeed.INFORMATION]
    if found & {"permit", "housing", "transport", "health", "tax", "work", "winter", "safety"}:
        needs.append(UserNeed.USEFULNESS)
    if found & {"permit", "processing", "union", "funding", "council"} or any(
        has_prefix_term(text, w) for w in ("because", "why", "means", "koska", "tarkoittaa")
    ):
        needs.append(UserNeed.UNDERSTANDING)
    if found & {"language", "culture"} or any(
        has_prefix_term(text, w) for w in ("community", "yhteis", "neighbour")
    ):
        needs.append(UserNeed.COMMUNITY)
    return needs


@dataclass
class Candidate:
    cid: str
    articles: list[Article]
    event_id: str | None = None
    text: str = ""
    hits: list[str] = field(default_factory=list)
    needs: list[UserNeed] = field(default_factory=list)
    score: float = 0.0
    outlets: int = 1

    @property
    def primary(self) -> Article:
        """Best article to name the pitch after: official, then full text, then the rest."""

        return sorted(
            self.articles,
            key=lambda a: (not a.official, a.access != "full", a.published_at is None),
        )[0]


def _title_jaccard(a: str, b: str) -> float:
    ta = {w for w in words(a) if len(w) >= 3}
    tb = {w for w in words(b) if len(w) >= 3}
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def group_candidates(articles: list[Article]) -> list[list[Article]]:
    """Group reports of the same story.

    event_id wins (the knowledge architect already decided these are one event, across
    outlets and languages). Only reports without an event fall back to syndication key
    or near-identical titles.
    """

    groups: list[list[Article]] = []
    by_event: dict[str, list[Article]] = {}
    loose: list[Article] = []
    for article in articles:
        if article.event_id:
            by_event.setdefault(article.event_id, []).append(article)
        else:
            loose.append(article)
    groups.extend(by_event.values())

    loose_groups: list[list[Article]] = []
    for article in loose:
        for group in loose_groups:
            if any(
                (article.syndication_key and article.syndication_key == g.syndication_key)
                or _title_jaccard(article.title, g.title) >= 0.6
                for g in group
            ):
                group.append(article)
                break
        else:
            loose_groups.append([article])
    groups.extend(loose_groups)
    return groups


def _fingerprint(text: str) -> tuple[set[str], set[str]]:
    nums = {n for n in numbers(text) if len(n) >= 2}
    return nums, proper_nouns(text)


def same_story(a: Candidate, b: Candidate) -> bool:
    """Cross-language duplicate check for candidates that have no shared event."""

    na, pa = _fingerprint(a.text)
    nb, pb = _fingerprint(b.text)
    shared = len(na & nb) + len(nouns_overlap(pa, pb))
    return shared >= 2 and bool(concepts(a.text) & concepts(b.text))


# ----------------------------------------------------------------------- the agent


class NewsEditorAgent(Agent):
    name = "news_editor"

    def select(
        self, community: Community, articles: list[Article], *, max_pitches: int = 3
    ) -> list[Pitch]:
        usable = self._drop_flagged(articles)
        candidates = self._build_candidates(community, usable)
        ranked = self._rank(community, candidates)
        chosen = self._choose(ranked, max_pitches)
        proposals = self._ask_model(community, chosen)
        pitches = [self._make_pitch(community, cand, proposals.get(cand.cid)) for cand in chosen]
        for pitch, cand in zip(pitches, chosen, strict=True):
            self.store.put(pitch)
            self.trace(
                "select",
                "selected",
                f"score {pitch.score:.2f}; needs {[n.value for n in pitch.needs]}; "
                f"matches interests {cand.hits}; {cand.outlets} outlet(s)",
                pitch_id=pitch.id,
                community_id=community.id,
                article_ids=pitch.article_ids,
                risk=pitch.risk_tier.value,
            )
        return pitches

    # ------------------------------------------------------------------ stages

    def _drop_flagged(self, articles: list[Article]) -> list[Article]:
        usable = []
        for article in articles:
            if article.injection_flags:
                self.trace(
                    "select",
                    "excluded_injection",
                    "text addressed to an AI: sent to humans, never pitched",
                    article_id=article.id,
                    flags=article.injection_flags,
                )
                continue
            usable.append(article)
        return usable

    def _build_candidates(self, community: Community, articles: list[Article]) -> list[Candidate]:
        candidates = []
        for index, group in enumerate(group_candidates(articles), start=1):
            text = " ".join(f"{a.title} {a.summary or ''} {a.content or ''}" for a in group)
            cand = Candidate(
                cid=f"cand{index}",
                articles=group,
                event_id=group[0].event_id,
                text=text,
                hits=interest_hits(text, community),
                needs=infer_needs(text),
                outlets=len({outlet_of(a.source_name).casefold() for a in group}),
            )
            candidates.append(cand)
        return candidates

    def _rank(self, community: Community, candidates: list[Candidate]) -> list[Candidate]:
        published = [a.published_at for c in candidates for a in c.articles if a.published_at]
        reference = max(published) if published else datetime.now(UTC)
        weights = community.needs_weights
        top_weight = max(weights.values()) if weights else 1.0

        ranked = []
        for cand in candidates:
            if not cand.hits:
                self.trace(
                    "select",
                    "rejected",
                    f"no overlap with the interests of {community.name}",
                    article_ids=[a.id for a in cand.articles],
                    title=cand.primary.title[:120],
                )
                continue
            in_region = region_hit(cand.text, community) or any(
                hit.casefold() in NATIONWIDE_INTERESTS for hit in cand.hits
            )
            official_group = any(a.official for a in cand.articles)
            if not is_relevant(cand.hits, in_region, official_group):
                self.trace(
                    "select",
                    "rejected",
                    f"weak match for {community.name}: matches {cand.hits}, "
                    f"{'in' if in_region else 'not in'} its region, "
                    f"{'official' if official_group else 'not official'} source",
                    article_ids=[a.id for a in cand.articles],
                    title=cand.primary.title[:120],
                )
                continue
            relevance = 0.75 * min(1.0, len(cand.hits) / 2) + 0.25 * (1.0 if in_region else 0.0)
            fit = (
                max((weights.get(n, 0.0) for n in cand.needs), default=0.0) / top_weight
                if weights
                else 0.5
            )
            ages = [
                (reference - a.published_at).total_seconds() / 86400
                for a in cand.articles
                if a.published_at
            ]
            recency = max(0.0, 1 - min(ages) / 14) if ages else 0.3
            official = 1.0 if official_group else 0.0
            corroboration = min(0.1, 0.05 * (cand.outlets - 1))
            cand.score = min(
                1.0,
                0.5 * relevance + 0.3 * fit + 0.1 * recency + 0.1 * official + corroboration,
            )
            ranked.append(cand)
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked

    def _choose(self, ranked: list[Candidate], max_pitches: int) -> list[Candidate]:
        cap = math.ceil(max_pitches / 2)
        per_source: Counter[str] = Counter()
        chosen: list[Candidate] = []
        deferred: list[Candidate] = []
        for cand in ranked:
            if len(chosen) >= max_pitches:
                break
            twin = next((c for c in chosen if same_story(c, cand)), None)
            if twin and not (cand.event_id and cand.event_id == twin.event_id):
                self.trace(
                    "select",
                    "skipped_same_story",
                    "another pitch already covers this story (possibly in another language)",
                    article_ids=[a.id for a in cand.articles],
                )
                continue
            source = cand.primary.source_id
            if per_source[source] >= cap:
                deferred.append(cand)
                self.trace(
                    "select",
                    "deferred_diversity",
                    f"already {per_source[source]} pitch(es) from source {source}",
                    article_ids=[a.id for a in cand.articles],
                )
                continue
            per_source[source] += 1
            chosen.append(cand)
        for cand in deferred:  # only fill leftover slots: "unless fewer candidates"
            if len(chosen) >= max_pitches:
                break
            chosen.append(cand)
        return chosen

    def _ask_model(self, community: Community, chosen: list[Candidate]) -> dict[str, PitchProposal]:
        if not chosen:
            return {}
        payload: dict[str, Any] = {
            "community": {
                "id": community.id,
                "name": community.name,
                "interests": community.interests,
                "needs_weights": {k.value: v for k, v in community.needs_weights.items()},
            },
            "candidates": [
                {
                    "candidate_id": c.cid,
                    "interest_hits": c.hits,
                    "inferred_needs": [n.value for n in c.needs],
                    "articles": [
                        {
                            "title": a.title,
                            "summary": a.summary,
                            "language": a.language,
                            "source_name": a.source_name,
                            "official": a.official,
                            "access": a.access,
                        }
                        for a in c.articles
                    ],
                }
                for c in chosen
            ],
        }
        blocks = "\n\n".join(
            f"Candidate {c.cid}:\n"
            + guards.wrap_untrusted(
                "\n".join(f"{a.title}\n{a.summary or ''}" for a in c.articles), c.cid
            )
            for c in chosen
        )
        system = (
            "You are the news editor of a community newsroom. For each candidate propose a "
            "clear working title (in the community's main language), one sentence on why it "
            "matters to this community, and which reader needs it serves "
            "(information, understanding, usefulness, emotion, break, community). "
            "Do not add facts that are not in the candidate. " + guards.UNTRUSTED_PREAMBLE
        )
        prompt = f"Community: {community.name}. {community.description}\n\n{blocks}"
        try:
            result = self.llm.generate(
                task="select_stories",
                system=system,
                prompt=prompt,
                schema=PitchProposals,
                payload=payload,
            )
        except LLMError as exc:
            self.trace("select", "model_unavailable", str(exc)[:200])
            return {}
        return {p.candidate_id: p for p in result.proposals}

    def _make_pitch(
        self, community: Community, cand: Candidate, proposal: PitchProposal | None
    ) -> Pitch:
        title = (proposal.working_title if proposal else "").strip()[:200] or cand.primary.title
        rationale = (proposal.rationale if proposal else "").strip()[:600] or (
            f"Matches {', '.join(cand.hits)} for {community.name}."
        )
        needs = [n for n in (proposal.needs if proposal else []) if isinstance(n, UserNeed)]
        needs = list(dict.fromkeys(needs)) or cand.needs
        if not needs:
            weights = community.needs_weights
            needs = [max(weights, key=weights.get)] if weights else [UserNeed.INFORMATION]
        risk, _topics = guards.classify_risk(cand.text, self.policy)
        return Pitch(
            id=new_id("pitch"),
            community_id=community.id,
            article_ids=[a.id for a in cand.articles],
            working_title=title,
            rationale=rationale,
            needs=needs,
            score=round(cand.score, 3),
            risk_tier=risk if isinstance(risk, RiskTier) else RiskTier.LOW,
            origin="monitor",
        )


# ------------------------------------------------------------------ offline handler


def _fake_select(payload: dict[str, Any]) -> PitchProposals:
    community = payload.get("community", {})
    proposals = []
    for cand in payload.get("candidates", []):
        articles = cand.get("articles", [])
        english = next((a for a in articles if a.get("language") == "en"), None)
        chosen = english or (articles[0] if articles else {})
        hits = ", ".join(cand.get("interest_hits", [])) or "everyday life"
        proposals.append(
            PitchProposal(
                candidate_id=cand["candidate_id"],
                working_title=chosen.get("title", "Untitled"),
                rationale=(
                    f"Relevant to {community.get('name', 'this community')}: touches {hits}. "
                    "[offline stand-in]"
                ),
                needs=[UserNeed(n) for n in cand.get("inferred_needs", ["information"])],
            )
        )
    return PitchProposals(proposals=proposals)


FAKE_HANDLERS: dict[str, Callable[[dict], BaseModel]] = {"select_stories": _fake_select}
