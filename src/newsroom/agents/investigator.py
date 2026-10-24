"""Reporter: extracts claims and looks for independent evidence.

The model proposes claims and reads candidate reports. Everything that decides what a
reader will be told is plain code:

* independence is judged per OUTLET (five Yle feeds are one voice, a syndicated copy is
  repetition, not corroboration);
* a claim is VERIFIED only with two independent supporting outlets, DISPUTED when an
  independent source contradicts it, ATTRIBUTED with a single source, otherwise UNKNOWN;
* official and full-text sources outrank paywalled (headline-and-summary-only) ones;
* there is no numeric confidence anywhere, only explicit unknowns.

Text inside reports is data. Reports flagged for instruction-like text are never read.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent, article_excerpt, prompt_articles
from newsroom.domain import (
    Article,
    Claim,
    ClaimStatus,
    EvidenceLink,
    InvestigationResult,
    Pitch,
    RiskTier,
    new_id,
)
from newsroom.lexicon import concepts, nouns_overlap, numbers, proper_nouns, query_terms
from newsroom.llm.base import LLMError
from newsroom.outlets import outlet_of

MAX_CLAIMS_PER_PITCH = 8
PAYWALL_PREFIX = "Paywalled:"

# --------------------------------------------------------------------------- schemas


class ClaimDraft(BaseModel):
    text: str
    attribution: str | None = None
    source_article_id: str | None = None


class ClaimDrafts(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    article_id: str
    relation: Literal["supports", "contradicts", "unrelated"]
    quote: str | None = None


class EvidenceAssessments(BaseModel):
    assessments: list[EvidenceAssessment] = Field(default_factory=list)


# ------------------------------------------------------------------ pure helpers


def signals(claim_text: str, other_text: str) -> tuple[int, int, int]:
    """Shared (numbers, proper nouns, concepts) between a claim and another text."""

    shared_numbers = len(numbers(claim_text) & numbers(other_text))
    shared_nouns = len(nouns_overlap(proper_nouns(claim_text), proper_nouns(other_text)))
    shared_concepts = len(concepts(claim_text) & concepts(other_text))
    return shared_numbers, shared_nouns, shared_concepts


def relation_from_text(
    claim_text: str, other_text: str
) -> Literal["supports", "contradicts"] | None:
    """Conservative, checkable relation between a claim and a text.

    supports      all the claim's numbers appear and there are >= 3 shared signals
    contradicts   same subject (>= 3 shared nouns/concepts) but the numbers differ
    """

    claim_numbers, other_numbers = numbers(claim_text), numbers(other_text)
    n, p, c = signals(claim_text, other_text)
    if claim_numbers and other_numbers and not (claim_numbers & other_numbers) and p + c >= 3:
        return "contradicts"
    if (not claim_numbers or claim_numbers <= other_numbers) and n + p + c >= 3:
        return "supports"
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _snippet_in(quote: str | None, text: str) -> str | None:
    """Keep a model-provided quote only if it really occurs in the text."""

    if quote and _norm(quote) and _norm(quote) in _norm(text):
        return quote.strip()[:300]
    return None


def _keywords(text: str, limit: int = 4) -> list[str]:
    nums = sorted(numbers(text))
    nouns = sorted(proper_nouns(text))
    return (nums + nouns)[:limit]


# ----------------------------------------------------------------------- the agent


class InvestigatorAgent(Agent):
    name = "reporter"

    def investigate(
        self,
        pitch: Pitch,
        *,
        round_no: int = 1,
        prior: InvestigationResult | None = None,
    ) -> InvestigationResult:
        articles = {
            a.id: a for aid in pitch.article_ids if (a := self.store.get(Article, aid)) is not None
        }
        readable = {}
        for article in articles.values():
            if article.injection_flags:
                self.trace(
                    "read",
                    "skipped_injection",
                    "report contains text addressed to an AI; not read",
                    article_id=article.id,
                )
            else:
                readable[article.id] = article

        claims = self.store.find(Claim, pitch_id=pitch.id)
        if not claims:
            claims = self._extract(pitch, list(readable.values()))
        hints = prior.search_hints if prior else []

        lookup = dict(articles)
        assessment_cap = self.settings.max_evidence_assessments_per_story
        assessments = 0
        for claim in claims:
            if claim.status in (ClaimStatus.VERIFIED, ClaimStatus.DISPUTED):
                continue
            # Re-checking a claim with the same hints and no new evidence used to cost one
            # model call per claim per round. Nothing can have changed, so do not ask again.
            if round_no > 1 and claim.assessed_hints == list(hints):
                self.trace(
                    "assess",
                    "skipped_unchanged",
                    "same search hints as the previous round; no new evidence to ask about",
                    claim_id=claim.id,
                    pitch_id=pitch.id,
                    round=round_no,
                )
                continue
            if assessments >= assessment_cap:
                self.trace(
                    "assess",
                    "capped",
                    f"assessment cap of {assessment_cap} call(s) per story reached",
                    claim_id=claim.id,
                    pitch_id=pitch.id,
                    round=round_no,
                )
                break
            if self._corroborate(claim, lookup, hints):
                assessments += 1
                claim.assessed_hints = list(hints)
                claim.assessed_rounds += 1
            self._decide(claim, lookup)
            self.store.put(claim)
            self.trace(
                "assess",
                claim.status.value,
                f"{claim.independent_support} independent supporting outlet(s); "
                f"{sum(e.relation == 'contradicts' for e in claim.evidence)} contradicting",
                claim_id=claim.id,
                pitch_id=pitch.id,
                round=round_no,
            )

        open_questions, search_hints = self._gaps(pitch, claims, lookup)
        return InvestigationResult(
            claims=claims,
            sufficient=not open_questions,
            open_questions=open_questions,
            search_hints=search_hints,
            round_no=round_no,
        )

    # ------------------------------------------------------------- extraction

    def _extract(self, pitch: Pitch, articles: list[Article]) -> list[Claim]:
        if not articles:
            return []
        all_articles = articles
        articles = prompt_articles(articles, self.settings.max_articles_per_prompt)
        if len(articles) < len(all_articles):
            self.trace(
                "extract",
                "trimmed",
                f"{len(all_articles) - len(articles)} older report(s) left out of the prompt; "
                f"context budget keeps {self.settings.max_articles_per_prompt}",
                pitch_id=pitch.id,
            )
        excerpt_chars = self.settings.extract_article_chars
        payload = {
            "articles": [
                {
                    "id": a.id,
                    "title": a.title,
                    "summary": a.summary,
                    # Paywalled outlets: only what the feed published.
                    "content": (a.content or "")[:excerpt_chars] if a.access == "full" else None,
                    "source_name": a.source_name,
                    "official": a.official,
                    "access": a.access,
                    "language": a.language,
                }
                for a in articles
            ]
        }
        blocks = "\n\n".join(
            f"Report {a.id} ({a.source_name}):\n"
            + guards.wrap_untrusted(article_excerpt(a, excerpt_chars), a.id)
            for a in articles
        )
        system = (
            "You are a reporter. List the checkable factual claims each report makes, one "
            "sentence each, with who said it. Do not add facts. Do not judge truth. "
            + guards.UNTRUSTED_PREAMBLE
        )
        try:
            drafts = self.llm.generate(
                task="extract_claims",
                system=system,
                prompt=blocks,
                schema=ClaimDrafts,
                payload=payload,
            )
        except LLMError as exc:
            self.trace(
                "extract", "skipped", f"model unavailable: {str(exc)[:160]}", pitch_id=pitch.id
            )
            return []

        by_id = {a.id: a for a in articles}
        seen: set[str] = set()
        claims: list[Claim] = []
        for draft in drafts.claims:
            text = draft.text.strip()
            if not 15 <= len(text) <= 400 or _norm(text) in seen:
                continue
            seen.add(_norm(text))
            source = by_id.get(draft.source_article_id or "") or articles[0]
            claim = Claim(
                id=new_id("claim"),
                pitch_id=pitch.id,
                text=text,
                status=ClaimStatus.UNKNOWN,
                attribution=(draft.attribution or outlet_of(source.source_name)).strip(),
                source_article_id=source.id,
                evidence=[
                    EvidenceLink(
                        article_id=source.id,
                        url=source.url,
                        source_name=outlet_of(source.source_name),
                        relation="supports",
                        independent=True,
                    )
                ],
            )
            claims.append(claim)
            if len(claims) >= MAX_CLAIMS_PER_PITCH:
                break
        self.trace("extract", "claims_extracted", f"{len(claims)} claim(s)", pitch_id=pitch.id)
        return claims

    # ----------------------------------------------------------- corroboration

    def _corroborate(self, claim: Claim, lookup: dict[str, Article], hints: list[str]) -> bool:
        """Ask the model about other reports. False when no model call was made."""

        source = lookup.get(claim.source_article_id or "") or (
            self.store.get(Article, claim.source_article_id) if claim.source_article_id else None
        )
        known = {e.article_id for e in claim.evidence}
        exclude = set(known)
        if source:
            exclude.add(source.id)
        query = " ".join(query_terms(claim.text, extra=hints))
        found = self.store.search_articles(query, exclude_ids=exclude, limit=5)
        candidates = []
        for article in found:
            if article.injection_flags:
                self.trace(
                    "read",
                    "skipped_injection",
                    "candidate evidence contains text addressed to an AI; not read",
                    article_id=article.id,
                )
                continue
            candidates.append(article)
        if not candidates:
            return False
        candidates = prompt_articles(candidates, self.settings.max_articles_per_prompt)
        excerpt_chars = self.settings.extract_article_chars

        payload: dict[str, Any] = {
            "claim": {"id": claim.id, "text": claim.text},
            "candidates": [
                {
                    "article_id": a.id,
                    "title": a.title,
                    "summary": a.summary,
                    "content": (a.content or "")[:excerpt_chars] if a.access == "full" else None,
                    "source_name": a.source_name,
                    "official": a.official,
                    "access": a.access,
                    "language": a.language,
                }
                for a in candidates
            ],
        }
        blocks = "\n\n".join(
            f"Candidate {a.id} ({a.source_name}):\n"
            + guards.wrap_untrusted(article_excerpt(a, excerpt_chars), a.id)
            for a in candidates
        )
        system = (
            "You are a reporter checking one claim against other reports, possibly in another "
            "language. For each candidate say whether it supports the claim, contradicts it, or "
            "is unrelated, with a short exact quote. Be strict: only say supports when the "
            "candidate states the same fact. " + guards.UNTRUSTED_PREAMBLE
        )
        try:
            result = self.llm.generate(
                task="assess_evidence",
                system=system,
                prompt=f"Claim: {claim.text}\n\n{blocks}",
                schema=EvidenceAssessments,
                payload=payload,
            )
        except LLMError as exc:
            self.trace(
                "assess", "skipped", f"model unavailable: {str(exc)[:160]}", claim_id=claim.id
            )
            return False

        by_id = {a.id: a for a in candidates}
        for item in result.assessments:
            cand = by_id.get(item.article_id)
            if cand is None or item.relation == "unrelated":
                continue
            visible_text = f"{cand.title} {cand.summary or ''}"
            if cand.access == "metadata_only":
                # Only a claim the public headline/summary itself states can count.
                if relation_from_text(claim.text, visible_text) != item.relation:
                    note = (
                        f"{PAYWALL_PREFIX} {outlet_of(cand.source_name)} reported the event, but "
                        "its text is behind a paywall, so details cannot be verified."
                    )
                    if note not in claim.unknowns:
                        claim.unknowns.append(note)
                    self.trace(
                        "assess",
                        "metadata_only_not_counted",
                        "headline/summary does not itself state the claim",
                        claim_id=claim.id,
                        article_id=cand.id,
                    )
                    continue
                quote = _snippet_in(item.quote, visible_text) or cand.title
            else:
                quote = _snippet_in(item.quote, f"{visible_text} {cand.content or ''}")
            claim.evidence.append(
                EvidenceLink(
                    article_id=cand.id,
                    url=cand.url,
                    source_name=outlet_of(cand.source_name),
                    relation=item.relation,
                    independent=self._independent(source, cand),
                    quote=quote,
                )
            )
        return True

    @staticmethod
    def _independent(source: Article | None, candidate: Article) -> bool:
        """Corroboration is not repetition: same outlet or same syndicated copy is not independent."""

        if source is None:
            return True
        if source.id == candidate.id:
            return False
        if outlet_of(source.source_name).casefold() == outlet_of(candidate.source_name).casefold():
            return False
        return not (
            source.syndication_key
            and candidate.syndication_key
            and source.syndication_key == candidate.syndication_key
        )

    # ------------------------------------------------------------- the verdict

    def _decide(self, claim: Claim, lookup: dict[str, Article]) -> None:
        def article(link: EvidenceLink) -> Article | None:
            return lookup.get(link.article_id) or self.store.get(Article, link.article_id)

        def strong(link: EvidenceLink) -> bool:
            art = article(link)
            return bool(art and (art.official or art.access == "full"))

        original = claim.source_article_id
        supports = [e for e in claim.evidence if e.relation == "supports"]
        contradictions = [
            e for e in claim.evidence if e.relation == "contradicts" and e.independent
        ]
        strong_extra_support = any(
            e.independent and e.article_id != original and strong(e) for e in supports
        )
        strong_contradiction = any(strong(e) for e in contradictions)
        weak_contradiction = bool(contradictions) and not strong_contradiction

        if strong_contradiction or (weak_contradiction and not strong_extra_support):
            claim.status = ClaimStatus.DISPUTED
        elif claim.independent_support >= 2:
            claim.status = ClaimStatus.VERIFIED
        elif supports:
            claim.status = ClaimStatus.ATTRIBUTED
        else:
            claim.status = ClaimStatus.UNKNOWN

        paywalled_support = [
            e for e in supports if (a := article(e)) and a.access == "metadata_only"
        ]
        kept = [u for u in claim.unknowns if u.startswith(PAYWALL_PREFIX)]
        unknowns = list(kept)
        if claim.status == ClaimStatus.UNKNOWN:
            unknowns.append("No supporting source could be established.")
            claim.uncertainty_note = "Not established. Do not present as fact."
        elif claim.status == ClaimStatus.ATTRIBUTED:
            unknowns.append(
                f"Only one source so far ({claim.attribution or 'unnamed'}); not independently confirmed."
            )
            claim.uncertainty_note = (
                "Reported by one source. Attribute it; do not state it as fact."
            )
        elif claim.status == ClaimStatus.DISPUTED:
            claim.uncertainty_note = "Sources disagree. Show both accounts with their sources."
            if weak_contradiction and not strong_contradiction:
                unknowns.append(
                    "The disagreement rests on a paywalled outlet's headline or summary only."
                )
        else:
            claim.uncertainty_note = (
                f"Confirmed by {claim.independent_support} independent outlets (not a probability)."
            )
        if paywalled_support and not any(strong(e) and e.independent for e in supports):
            unknowns.append(
                f"{PAYWALL_PREFIX} this claim rests on headline and summary of a paywalled "
                "report; the full text could not be read."
            )
        claim.unknowns = list(dict.fromkeys(unknowns))

    # ------------------------------------------------------------------- gaps

    def _gaps(
        self, pitch: Pitch, claims: list[Claim], lookup: dict[str, Article]
    ) -> tuple[list[str], list[str]]:
        high = pitch.risk_tier == RiskTier.HIGH
        questions: list[str] = []
        hints: list[str] = []
        if not claims:
            return ["No claims could be extracted from this pitch."], []
        for claim in claims:
            official = any(
                e.relation == "supports"
                and (art := lookup.get(e.article_id) or self.store.get(Article, e.article_id))
                and art.official
                for e in claim.evidence
            )
            gap = None
            if claim.status == ClaimStatus.UNKNOWN:
                gap = f"Nothing supports: {claim.text}"
            elif high and not official:
                gap = f"High-risk claim needs the official source: {claim.text}"
            elif claim.status == ClaimStatus.ATTRIBUTED and not official:
                gap = f"Single source only; look for an independent report: {claim.text}"
            if gap:
                questions.append(gap)
                hints.append(" ".join(_keywords(claim.text)))
        return questions, [h for h in dict.fromkeys(hints) if h]


# ------------------------------------------------------------------ offline handlers

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_REPORT_VERBS = (
    "says", "said", "kertoo", "kertovat", "recommend", "according", "mukaan", "announce",
    "ilmoitti", "approved", "myöntää", "nousee", "tulevat voimaan", "struggle", "voitti",
)  # fmt: skip
_ATTRIBUTION = [
    re.compile(r"([A-ZÅÄÖ][\w-]*(?: [A-ZÅÄÖ][\w-]*){0,4}) (?:says|said|kertoo|recommends)"),
    re.compile(r"[Aa]ccording to ([A-ZÅÄÖ][\w-]*(?: [A-ZÅÄÖ][\w-]*){0,3})"),
    re.compile(r"([A-ZÅÄÖ][\w-]*(?: [A-ZÅÄÖ][\w-]*){0,3}) mukaan"),
]


def _attribution(sentence: str) -> str | None:
    for pattern in _ATTRIBUTION:
        match = pattern.search(sentence)
        if match:
            return match.group(1).strip()
    return None


def _fake_extract(payload: dict[str, Any]) -> ClaimDrafts:
    drafts: list[ClaimDraft] = []
    for art in payload.get("articles", []):
        text = art.get("content") or art.get("summary") or art.get("title") or ""
        sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
        picked = [
            s
            for s in sentences
            if len(s) >= 25
            and not s.endswith("?")
            and (any(ch.isdigit() for ch in s) or any(v in s.casefold() for v in _REPORT_VERBS))
        ][:3]
        if not picked and sentences:
            picked = [sentences[0]]
        for sentence in picked:
            drafts.append(
                ClaimDraft(
                    text=sentence,
                    attribution=_attribution(sentence),
                    source_article_id=art.get("id"),
                )
            )
    return ClaimDrafts(claims=drafts)


def _best_sentence(claim_text: str, text: str) -> str | None:
    best, best_score = None, 0
    for sentence in (s.strip() for s in _SENTENCE.split(text) if s.strip()):
        n, p, c = signals(claim_text, sentence)
        if n + p + c > best_score:
            best, best_score = sentence, n + p + c
    return best


def _fake_assess(payload: dict[str, Any]) -> EvidenceAssessments:
    claim_text = payload.get("claim", {}).get("text", "")
    out = []
    for cand in payload.get("candidates", []):
        text = " ".join(filter(None, [cand.get("title"), cand.get("summary"), cand.get("content")]))
        relation = relation_from_text(claim_text, text) or "unrelated"
        out.append(
            EvidenceAssessment(
                article_id=cand["article_id"],
                relation=relation,
                quote=_best_sentence(claim_text, text) if relation != "unrelated" else None,
            )
        )
    return EvidenceAssessments(assessments=out)


FAKE_HANDLERS: dict[str, Callable[[dict], BaseModel]] = {
    "extract_claims": _fake_extract,
    "assess_evidence": _fake_assess,
}
