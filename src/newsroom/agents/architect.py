"""Knowledge Architect: files every report under ONE canonical event in the topic tree.

When Kaleva, Yle, BBC (or any outlets, in any language) report the same news, their
articles attach to the same ``Event``. The newsroom then investigates the event once and
can say how many *independent* outlets reported it.

Division of labour (the rule of this codebase):

* The model only PROPOSES: an event title, entities, topic suggestions.
* CODE decides: clustering uses language-independent signals (numbers, named entities,
  places, a small Finnish<->English concept map, title overlap) and needs two different
  kinds of signal to agree before two reports are called the same news.
* Tree changes are expressed as ``TreeProposal`` objects. Code validates them. Routine
  ones (new event, attaching a report, filing an event under an existing topic) are
  auto-committed; structural ones (a new top-level topic, any merge, anything that fails
  validation) wait for a named human. Only validated changes touch the authoritative
  collections, and every proposal is stored for audit.

Paywalled sources (``Article.access == "metadata_only"``) contribute headline and summary
only. They can start or join an event, but nothing is ever read from behind the paywall.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent, AgentContext
from newsroom.domain import (
    Article,
    Event,
    TimelineEntry,
    Topic,
    TreeChange,
    TreeProposal,
    new_id,
    utcnow,
)
from newsroom.llm.base import LLMError
from newsroom.store.cache import ReadCache
from newsroom.topics import slugify, topic_id_for

# ----------------------------------------------------------------------- thresholds

ATTACH_THRESHOLD = 0.30  # similarity needed to call two reports "the same news"
MERGE_THRESHOLD = 0.45  # score before two EVENTS are proposed for a merge...
MERGE_MIN_KINDS = 3  # ...and three independent signal kinds must agree
MIN_AGREEING_KINDS = 2  # independent signal kinds that must agree (number/entity/concept/title)
WINDOW_HOURS = 72  # a report only joins an event active within this window
CONFIRMED_MIN_OUTLETS = 2  # independent outlets needed for status "confirmed"
MAX_TOPICS_PER_EVENT = 2
QUARANTINE_MARK = "quarantine:injection"

NUMBER_SHARE = 0.6  # score per shared number (capped at 1)
ENTITY_SHARE = 0.5  # score per shared specific entity (capped at 1)
CONCEPT_SHARE = 0.35  # score per shared concept (capped at 1)
TITLE_AGREE = 0.34  # title-stem Jaccard needed for the title signal to count
WEIGHTS = {"number": 0.30, "entity": 0.30, "concept": 0.25, "title": 0.15}
BROAD_PLACE_BONUS = 0.05

AGENT_NAMES = {"agent", "ai", "bot", "system", "model", "architect", "knowledge_architect"}

# ------------------------------------------------------------------ language-aware data

# concept -> terms (matched at the START of a word, so Finnish inflections still match)
CONCEPTS: dict[str, tuple[str, ...]] = {
    "residence_permit": (
        "oleskelulup",
        "residence permit",
        "residence card",
        "work permit",
        "työlup",
        "permit",
        "viisumi",
        "visa",
    ),
    "immigration_authority": ("migri", "maahanmuuttovirasto", "immigration service"),
    "processing_time": ("käsittelyaik", "käsittely", "processing time", "processing"),
    "housing": (
        "asunto",
        "vuokra",
        "housing",
        "rental",
        "accommodation",
        "majoitus",
        "apartment",
        "landlord",
        "vuokraisäntä",
    ),
    "seasonal_work": ("kausityö", "seasonal"),
    "union": ("ammattiliit", "työehtosopimus", "union", "collective agreement"),
    "council": ("valtuusto", "kaupunginhallitus", "council", "kaupunki ", "city of"),
    "transport": (
        "joukkoliikenne",
        "public transport",
        "bussi",
        "bus ",
        "aikataulu",
        "timetable",
        "kausilip",
        "season ticket",
    ),
    "aviation": (
        "lentoasema",
        "lentokenttä",
        "lento",
        "airport",
        "flight",
        "airline",
        "reitti",
        "route",
    ),
    "daycare": ("päiväkoti", "daycare", "kindergarten", "varhaiskasvatus", "early childhood"),
    "sami": ("saamel", "saamen", "sámi", "sami ", "saami", "saamenkiel"),
    "price": ("hinta", "hinn", "price", "cost", "maksu", "fare"),
    "funding": ("rahoitus", "avustus", "funding", "grant"),
    "health": ("terveys", "lääkäri", "sairaala", "health", "doctor", "hospital"),
    "education": ("yliopisto", "opiskel", "university", "student", "koulu", "school"),
    "winter_driving": (
        "talvirengas",
        "winter tyre",
        "winter tire",
        "winter driving",
        "liukas",
        "black ice",
    ),
    "energy": ("sähkö", "electricity", "energia", "energy"),
    "sport": ("kiekko", "jääkiekko", "ottelu", "hockey", "match ", "football", "jalkapallo"),
    "ukraine_war": ("ukraina", "ukraine", "sota ", "war "),
}

# aliases that make Finnish and English place/body names the same entity key
ALIASES: dict[str, str] = {
    "lappi": "lapland",
    "lapin": "lapland",
    "lapissa": "lapland",
    "suomi": "finland",
    "suomen": "finland",
    "suomessa": "finland",
    "ukraina": "ukraine",
    "ukrainan": "ukraine",
    "venäjä": "russia",
    "venäjän": "russia",
    "ruotsi": "sweden",
    "ruotsin": "sweden",
    "eurooppa": "europe",
    "euroopan": "europe",
    "saamelaiskäräjät": "sami-parliament",
}

# places broad enough that sharing one proves nothing about "the same news"
BROAD_PLACES = {
    "oulu",
    "finland",
    "lapland",
    "helsinki",
    "europe",
    "eu",
    "sweden",
    "russia",
    "ukraine",
    "pohjois-pohjanmaa",
    "uusimaa",
}

PLACES = (
    "Oulu",
    "Levi",
    "Rovaniemi",
    "Utsjoki",
    "Lapland",
    "Helsinki",
    "Finland",
    "Ukraine",
    "Inari",
    "Ylläs",
    "Saariselkä",
    "Tampere",
    "Turku",
    "Kemi",
    "Kuusamo",
    "Sodankylä",
)

_STOP_ENTITIES = {
    # outlets are not part of what happened
    "kaleva",
    "yle",
    "bbc",
    "reuters",
    "stt",
    "news",
    "uutiset",
    "fixture",
    "lehti",
    # calendar
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "maanantai",
    "tiistai",
    "keskiviikko",
    "torstai",
    "perjantai",
    "lauantai",
    "sunnuntai",
    "tammikuu",
    "helmikuu",
    "maaliskuu",
    "huhtikuu",
    "toukokuu",
    "kesäkuu",
    "heinäkuu",
    "elokuu",
    "syyskuu",
    "lokakuu",
    "marraskuu",
    "joulukuu",
    # adjectives / generic capitalised words
    "finnish",
    "english",
    "swedish",
    "russian",
    "european",
    "service",
    "services",
    "the",
    "this",
    "that",
    "last",
    "first",
    "new",
    "free",
    "ignore",
    "approve",
    "reveal",
}

_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# ----------------------------------------------------------------------- model schema


class TopicSuggestion(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)
    parent_hint: str | None = None  # name of an existing topic to file it under


class Analysis(BaseModel):
    """What the model PROPOSES about one article. Code validates every field."""

    event_title: str = ""
    entities: list[str] = Field(default_factory=list)
    place: str | None = None
    keywords: list[str] = Field(default_factory=list)
    topic_suggestions: list[TopicSuggestion] = Field(default_factory=list)


class ArchitectResult(BaseModel):
    events: list[Event] = Field(default_factory=list)
    proposals: list[TreeProposal] = Field(default_factory=list)
    attached: dict[str, str] = Field(default_factory=dict)  # article_id -> event_id


# ------------------------------------------------------------------ signal extraction


def _stem(token: str, n: int = 5) -> str:
    return token.casefold()[:n]


def canonical_entity(token: str) -> str:
    key = token.casefold()
    return ALIASES.get(key, key)


def extract_numbers(text: str | None) -> set[str]:
    """Numbers that can identify an occurrence: 2+ digits, decimals; never years."""

    found: set[str] = set()
    for raw in _NUMBER.findall(text or ""):
        value = raw.replace(",", ".")
        digits = value.replace(".", "")
        if len(digits) < 2:
            continue
        if "." not in value and 1900 <= int(value) <= 2100:
            continue
        found.add(value)
    return found


def extract_entities(text: str | None) -> set[str]:
    """Capitalised proper-noun-like tokens as canonical keys.

    Sentence-initial words are skipped (they are capitalised anyway) unless they are
    acronyms, known places or also capitalised later in the text. Title-case headlines
    (most words capitalised) are skipped entirely.
    """

    clean = guards.sanitize(text)
    entities: set[str] = set()
    capitalised_elsewhere: set[str] = set()
    initial: list[str] = []
    for sentence in _SENTENCE.split(clean):
        tokens = _WORD.findall(sentence)
        if not tokens:
            continue
        caps = sum(1 for t in tokens if t[0].isupper())
        title_case = len(tokens) >= 4 and caps / len(tokens) >= 0.6
        for index, token in enumerate(tokens):
            if not token[0].isupper() or len(token) < 3 or title_case:
                continue
            key = canonical_entity(token)
            if key in _STOP_ENTITIES or token.casefold() in _STOP_ENTITIES:
                continue
            if index == 0:
                initial.append(token)
            else:
                capitalised_elsewhere.add(key)
                entities.add(key)
    for token in initial:
        key = canonical_entity(token)
        acronym = token.isupper() and len(token) <= 5
        known_place = any(entity_match(token.casefold(), p.casefold()) for p in PLACES)
        if acronym or known_place or key in capitalised_elsewhere:
            entities.add(key)
    return entities


def extract_concepts(text: str | None) -> set[str]:
    lowered = f" {guards.sanitize(text).casefold()} "
    found = set()
    for name, terms in CONCEPTS.items():
        for term in terms:
            if re.search(r"(?<!\w)" + re.escape(term.casefold()), lowered):
                found.add(name)
                break
    return found


def title_stems(title: str | None) -> set[str]:
    return {_stem(w) for w in _WORD.findall(title or "") if len(w) >= 4}


def entity_match(a: str, b: str) -> bool:
    """Same entity despite Finnish endings (Oulu/Oulun, Levi/Levillä, Utsjoki/Utsjoella)."""

    if a == b:
        return True
    shorter = min(len(a), len(b))
    if shorter < 4:
        return False
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    return common >= max(4, math.ceil(0.7 * shorter))


@dataclass
class Signals:
    numbers: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    concepts: set[str] = field(default_factory=set)
    stems: set[str] = field(default_factory=set)


@dataclass
class Similarity:
    score: float
    kinds: list[str]
    shared: dict[str, list[str]]


def compare(a: Signals, b: Signals) -> Similarity:
    shared_numbers = sorted(a.numbers & b.numbers)
    specific: list[str] = []
    broad: list[str] = []
    for ea in sorted(a.entities):
        if any(entity_match(ea, eb) for eb in b.entities):
            (broad if ea in BROAD_PLACES else specific).append(ea)
    shared_concepts = sorted(a.concepts & b.concepts)
    union = a.stems | b.stems
    jaccard = len(a.stems & b.stems) / len(union) if union else 0.0

    parts = {
        "number": min(1.0, NUMBER_SHARE * len(shared_numbers)),
        "entity": min(1.0, ENTITY_SHARE * len(specific)),
        "concept": min(1.0, CONCEPT_SHARE * len(shared_concepts)),
        "title": jaccard,
    }
    kinds = []
    if shared_numbers:
        kinds.append("number")
    if specific:
        kinds.append("entity")
    if len(shared_concepts) >= 2:
        kinds.append("concept")
    if jaccard >= TITLE_AGREE:
        kinds.append("title")
    score = sum(WEIGHTS[k] * v for k, v in parts.items())
    if broad:
        score += BROAD_PLACE_BONUS
    return Similarity(
        min(1.0, score),
        kinds,
        {"numbers": shared_numbers, "entities": specific + broad, "concepts": shared_concepts},
    )


def article_text(article: Article) -> str:
    """Everything we may read. Paywalled articles contribute headline and summary only."""

    if article.access == "metadata_only":
        return f"{article.title}. {article.summary or ''}"
    return f"{article.title}. {article.summary or ''}. {article.content or ''}"


def signals_for(article: Article, extra_entities: Iterable[str] = ()) -> Signals:
    text = article_text(article)
    entities = extract_entities(text) | {canonical_entity(e) for e in extra_entities}
    return Signals(
        numbers=extract_numbers(text),
        entities={e for e in entities if e and e not in _STOP_ENTITIES},
        concepts=extract_concepts(text),
        stems=title_stems(article.title),
    )


# --------------------------------------------------------------------------- outlets


def outlet_of(source_name: str) -> str:
    """One broadcaster or newspaper = one outlet, however many feeds it has."""

    try:  # the shared helper wins when it exists
        from newsroom.outlets import outlet_of as shared  # type: ignore[import-not-found]

        return shared(source_name)
    except ImportError:
        pass
    name = re.sub(r"\s*\(.*?\)\s*", " ", source_name or "").strip()
    folded = name.casefold()
    if folded.startswith(("yle", "svenska yle")) or " yle" in folded:
        return "Yle"
    if folded.startswith("bbc"):
        return "BBC"
    if folded.startswith("kaleva"):
        return "Kaleva"
    return re.split(r"\s[·|–-]\s", name)[0].strip() or "Unknown"


def _aware(moment: datetime | None) -> datetime:
    if moment is None:
        return utcnow()
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


# -------------------------------------------------------------------- fake (offline)

_SPORT_KEYWORDS = ["kiekko", "jääkiekko", "ottelu", "hockey", "football", "sport", "urheilu"]


def _fake_analyse(payload: dict[str, Any]) -> Analysis:
    title = str(payload.get("title", ""))
    text = f"{title}. {payload.get('summary', '')}. {payload.get('content', '')}"
    entities = sorted(extract_entities(text))
    place = next((p for p in PLACES if re.search(r"(?<!\w)" + p.casefold(), text.casefold())), None)
    suggestions: list[TopicSuggestion] = []
    if "sport" in extract_concepts(text):
        suggestions.append(
            TopicSuggestion(
                name="Sports",
                keywords=_SPORT_KEYWORDS,
                parent_hint="Culture and everyday life",
            )
        )
    return Analysis(
        event_title=title,
        entities=entities,
        place=place,
        keywords=sorted(extract_concepts(text)),
        topic_suggestions=suggestions,
    )


FAKE_HANDLERS = {"architect_analyse": _fake_analyse}

# ----------------------------------------------------------------------------- agent

SYSTEM = (
    "You are the knowledge architect of a newsroom. Given ONE news report, propose a short "
    "neutral event title, the named entities and place it is about, and, only if the report "
    "does not fit an existing topic, a topic suggestion. You only propose; a program "
    "validates everything. Answer with JSON matching the schema.\n" + guards.UNTRUSTED_PREAMBLE
)


class KnowledgeArchitectAgent(Agent):
    name = "knowledge_architect"

    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(ctx)

    # ------------------------------------------------------------------ public API

    def process(self, articles: list[Article]) -> ArchitectResult:
        result = ArchitectResult()
        # One batch reads the same events and topics for every article, so read them from memory.
        original = self.store
        cache = ReadCache(original)
        self.store = cache
        try:
            self._process_batch(articles, result)
        finally:
            cache.flush()
            self.store = original
        return result

    def _process_batch(self, articles: list[Article], result: ArchitectResult) -> None:
        for position, article in enumerate(articles):
            reason = self.ctx.limits.stop_reason()
            if reason:
                self.trace(
                    "cluster",
                    "stopped",
                    f"{reason}; {len(articles) - position} articles wait for the next cycle",
                )
                break
            self._process_one(article, result)

    def propose_topic(
        self,
        *,
        name: str,
        keywords: list[str] | None = None,
        parent_id: str | None = None,
        event_id: str | None = None,
        evidence_article_ids: list[str] | None = None,
        reason: str = "",
    ) -> TreeProposal:
        """Propose a new topic. Under an existing parent it is auto-committed as 'proposed'
        (active only after a human approves); a new top-level topic waits for a human."""

        change = TreeChange(
            kind="create_topic",
            payload={
                "name": name.strip()[:60],
                "slug": slugify(name),
                "parent_id": parent_id,
                "keywords": _clean_keywords(keywords or []),
                "event_id": event_id,
            },
            evidence_article_ids=evidence_article_ids or [],
            reason=reason or "no existing topic fits",
        )
        return self._submit([change])

    def validate_proposal(self, proposal: TreeProposal) -> TreeProposal:
        """Decide by CODE whether a proposal may be committed without a human."""

        notes: list[str] = []
        needs_human = False
        for change in proposal.changes:
            problems, human = self._check(change)
            notes.extend(problems)
            needs_human = needs_human or human
        proposal.validation_notes = notes
        proposal.status = "needs_human" if (needs_human or notes) else "auto_committed"
        return proposal

    def approve_proposal(self, proposal_id: str, *, reviewer: str, approve: bool) -> TreeProposal:
        self._require_human(reviewer)
        proposal = self.store.get(TreeProposal, proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal '{proposal_id}'.")
        if proposal.status != "needs_human":
            raise ValueError(f"Proposal '{proposal_id}' is not waiting for a human.")
        proposal.decided_by = reviewer.strip()
        if approve:
            for change in proposal.changes:
                self._apply(change, human_approved=True)
            proposal.status = "approved"
        else:
            proposal.status = "rejected"
        self.store.put(proposal)
        self.trace(
            "approve_proposal",
            proposal.status,
            f"decided by {proposal.decided_by}",
            proposal_id=proposal.id,
        )
        return proposal

    def merge_events(self, keep_id: str, merge_id: str, *, reviewer: str) -> Event:
        """Human-approved merge of two events. History is preserved on the merged event."""

        self._require_human(reviewer)
        event = self._do_merge(keep_id, merge_id)
        self.trace(
            "merge_events",
            "merged",
            f"approved by {reviewer.strip()}",
            keep=keep_id,
            merged=merge_id,
        )
        return event

    def detect_duplicate_events(self) -> list[TreeProposal]:
        """Propose merges (always for a human) for events that look like the same news."""

        events = [
            e
            for e in self.store.find(Event)
            if e.status != "merged" and QUARANTINE_MARK not in e.entities
        ]
        already = {
            (c.payload.get("keep_id"), c.payload.get("merge_id"))
            for p in self.store.find(TreeProposal)
            if p.status in ("needs_human", "approved", "rejected")
            for c in p.changes
            if c.kind == "merge_events"
        }
        proposals: list[TreeProposal] = []
        for i, first in enumerate(events):
            for second in events[i + 1 :]:
                if (first.id, second.id) in already or (second.id, first.id) in already:
                    continue
                if not self._within_window_events(first, second):
                    continue
                best = self._event_similarity(first, second)
                if best.score >= MERGE_THRESHOLD and len(best.kinds) >= MERGE_MIN_KINDS:
                    keep, merge = sorted((first, second), key=lambda e: e.first_seen)
                    proposals.append(
                        self._submit(
                            [
                                TreeChange(
                                    kind="merge_events",
                                    payload={
                                        "keep_id": keep.id,
                                        "merge_id": merge.id,
                                        "score": round(best.score, 3),
                                        "signals": best.kinds,
                                    },
                                    evidence_article_ids=keep.article_ids[:2]
                                    + merge.article_ids[:2],
                                    reason=f"events look identical ({', '.join(best.kinds)})",
                                )
                            ]
                        )
                    )
        return proposals

    def event_report(self, event_id: str) -> dict[str, Any]:
        event = self.store.get(Event, event_id)
        if event is None:
            raise ValueError(f"Unknown event '{event_id}'.")
        articles = self._articles_of(event)
        outlets, independent = _independence(articles)
        topics = [t for t in (self.store.get(Topic, tid) for tid in event.topic_ids) if t]
        return {
            "id": event.id,
            "title": event.title,
            "status": event.status,
            "quarantined": QUARANTINE_MARK in event.entities,
            "topics": [t.name for t in topics],
            "outlets": outlets,
            "languages": sorted(event.languages),
            "timeline": [
                {"at": t.at.isoformat(), "text": t.text, "article_id": t.article_id}
                for t in sorted(event.timeline, key=lambda t: t.at)
            ],
            "article_count": len(articles),
            "independent_outlet_count": independent,
        }

    # ---------------------------------------------------------------- processing

    def _process_one(self, article: Article, result: ArchitectResult) -> None:
        if self.store.get(Article, article.id) is None:
            self.store.put(article)
        stored = self.store.get(Article, article.id) or article
        if stored.event_id and self.store.get(Event, stored.event_id):
            result.attached[stored.id] = stored.event_id
            self.trace(
                "skip",
                "already_filed",
                "article already belongs to an event",
                article_id=stored.id,
                event_id=stored.event_id,
            )
            return

        flagged = bool(stored.injection_flags) or bool(
            guards.scan_injection(f"{stored.title} {stored.summary} {stored.content}", self.policy)
        )
        analysis = self._analyse(
            stored, skip_model=flagged or not self.settings.architect_uses_model
        )
        signals = signals_for(stored, analysis.entities)

        if flagged:
            self.trace(
                "quarantine",
                "isolated",
                "instruction-like text; never clustered",
                article_id=stored.id,
                flags=stored.injection_flags,
            )
            match, similarity = None, None
        else:
            match, similarity = self._best_event(stored, signals)

        if match is not None and similarity is not None:
            changes = [self._attach_change(stored, match, similarity, analysis)]
            event_id = match.id
            self.trace(
                "attach",
                "joined_event",
                f"score {similarity.score:.2f}, signals {similarity.kinds}",
                article_id=stored.id,
                event_id=event_id,
            )
        else:
            event = self._new_event(stored, analysis, quarantined=flagged)
            changes = [
                TreeChange(
                    kind="create_event",
                    payload={"event": event.model_dump(mode="json"), "article_id": stored.id},
                    evidence_article_ids=[stored.id],
                    reason="no existing event matches" if not flagged else "quarantined article",
                )
            ]
            event_id = event.id
            self.trace(
                "create_event",
                "new_event",
                changes[0].reason,
                article_id=stored.id,
                event_id=event_id,
            )

        # Topic changes ride in a second step because they depend on the event existing.
        proposal = self._submit(changes)
        result.proposals.append(proposal)
        if proposal.status != "auto_committed":
            return
        result.attached[stored.id] = event_id

        if not flagged:
            for p in self._topic_proposals(event_id, stored, analysis):
                result.proposals.append(p)
        event = self.store.get(Event, event_id)
        if event is not None and all(e.id != event.id for e in result.events):
            result.events.append(event)
        elif event is not None:
            result.events = [event if e.id == event.id else e for e in result.events]

    def _analyse(self, article: Article, *, skip_model: bool) -> Analysis:
        payload = {
            "title": article.title,
            "summary": article.summary or "",
            "content": "" if article.access == "metadata_only" else (article.content or "")[:1500],
            "language": article.language,
        }
        fallback = Analysis(event_title=article.title)
        if skip_model:
            return fallback
        prompt = "Analyse this news report.\n" + guards.wrap_untrusted(
            f"{payload['title']}\n{payload['summary']}\n{payload['content']}",
            f"{article.source_name} ({article.language})",
        )
        try:
            raw = self.llm.generate(
                task="architect_analyse",
                system=SYSTEM,
                prompt=prompt,
                schema=Analysis,
                payload=payload,
            )
        except LLMError as exc:
            self.trace("analyse", "degraded", f"model unavailable: {exc}", article_id=article.id)
            return fallback
        return self._sanitise_analysis(raw, article)

    def _sanitise_analysis(self, raw: Analysis, article: Article) -> Analysis:
        """The model proposes; keep only what is grounded in the text we are allowed to read."""

        text = article_text(article).casefold()
        entities = [e for e in raw.entities[:20] if e and len(e) < 60 and e.casefold()[:5] in text]
        title = guards.sanitize(raw.event_title, max_chars=140) or article.title
        suggestions = []
        for s in raw.topic_suggestions[:3]:
            name = guards.sanitize(s.name, max_chars=60)
            if name and not guards.scan_injection(name, self.policy):
                suggestions.append(
                    TopicSuggestion(
                        name=name,
                        keywords=_clean_keywords(s.keywords),
                        parent_hint=s.parent_hint,
                    )
                )
        place = raw.place if raw.place and raw.place.casefold()[:4] in text else None
        return Analysis(
            event_title=title,
            entities=entities,
            place=place,
            keywords=_clean_keywords(raw.keywords),
            topic_suggestions=suggestions,
        )

    # ------------------------------------------------------------------- matching

    def _articles_of(self, event: Event) -> list[Article]:
        articles = [self.store.get(Article, aid) for aid in event.article_ids]
        found = [a for a in articles if a is not None]
        return sorted(found, key=lambda a: _aware(a.published_at))

    def _best_event(
        self, article: Article, signals: Signals
    ) -> tuple[Event | None, Similarity | None]:
        when = _aware(article.published_at)
        best: tuple[Event, Similarity] | None = None
        for event in self.store.find(Event):
            if event.status == "merged" or QUARANTINE_MARK in event.entities:
                continue
            members = self._articles_of(event)
            if not any(
                abs(_aware(m.published_at) - when) <= timedelta(hours=WINDOW_HOURS) for m in members
            ):
                continue
            top: Similarity | None = None
            for member in members:
                sim = compare(signals, signals_for(member))
                if top is None or sim.score > top.score:
                    top = sim
            if top and top.score >= ATTACH_THRESHOLD and len(top.kinds) >= MIN_AGREEING_KINDS:
                if best is None or top.score > best[1].score:
                    best = (event, top)
        return (best[0], best[1]) if best else (None, None)

    def _event_signals(self, event: Event) -> list[Signals]:
        return [signals_for(a) for a in self._articles_of(event)]

    def _event_similarity(self, first: Event, second: Event) -> Similarity:
        best = Similarity(0.0, [], {})
        for a in self._event_signals(first):
            for b in self._event_signals(second):
                sim = compare(a, b)
                if sim.score > best.score:
                    best = sim
        return best

    def _within_window_events(self, first: Event, second: Event) -> bool:
        times_a = [_aware(a.published_at) for a in self._articles_of(first)]
        times_b = [_aware(a.published_at) for a in self._articles_of(second)]
        return any(abs(x - y) <= timedelta(hours=WINDOW_HOURS) for x in times_a for y in times_b)

    # ------------------------------------------------------------- change building

    def _new_event(self, article: Article, analysis: Analysis, *, quarantined: bool) -> Event:
        when = _aware(article.published_at)
        entities = sorted(
            extract_entities(article_text(article))
            | {canonical_entity(e) for e in analysis.entities}
        )
        if quarantined:
            entities = [QUARANTINE_MARK]
        return Event(
            id=new_id("evt"),
            title=analysis.event_title or article.title,
            summary=article.summary or "",
            topic_ids=[],
            article_ids=[article.id],
            source_names=[outlet_of(article.source_name)],
            languages=[article.language],
            entities=entities,
            place=analysis.place,
            timeline=[TimelineEntry(at=when, text=article.title, article_id=article.id)],
            status="candidate",
            first_seen=when,
            last_updated=utcnow(),
        )

    def _attach_change(
        self, article: Article, event: Event, similarity: Similarity, analysis: Analysis
    ) -> TreeChange:
        return TreeChange(
            kind="attach_article",
            payload={
                "article_id": article.id,
                "event_id": event.id,
                "score": round(similarity.score, 3),
                "signals": similarity.kinds,
                "shared": similarity.shared,
                "entities": sorted(canonical_entity(e) for e in analysis.entities),
                "place": analysis.place,
            },
            evidence_article_ids=[article.id, *event.article_ids[:3]],
            reason=f"same news: {', '.join(similarity.kinds)} agree",
        )

    def _topic_proposals(
        self, event_id: str, article: Article, analysis: Analysis
    ) -> list[TreeProposal]:
        event = self.store.get(Event, event_id)
        if event is None:
            return []
        text = " ".join(
            [event.title, event.summary, " ".join(event.entities)]
            + [article_text(a) for a in self._articles_of(event)]
        )
        matched = self._match_topics(text)
        proposals: list[TreeProposal] = []
        fresh = [t for t in matched if t.id not in event.topic_ids]
        if fresh:
            proposals.append(
                self._submit(
                    [
                        TreeChange(
                            kind="attach_event_to_topic",
                            payload={"event_id": event_id, "topic_id": t.id},
                            evidence_article_ids=[article.id],
                            reason=f"topic keywords match '{t.name}'",
                        )
                        for t in fresh
                    ]
                )
            )
        if not matched and not event.topic_ids:
            by_name = {t.name.casefold(): t for t in self.store.find(Topic)}
            for suggestion in analysis.topic_suggestions[:1]:
                parent = by_name.get((suggestion.parent_hint or "").casefold())
                exists = any(slugify(suggestion.name) == t.slug for t in by_name.values())
                if exists:
                    continue
                proposals.append(
                    self.propose_topic(
                        name=suggestion.name,
                        keywords=suggestion.keywords,
                        parent_id=parent.id if parent else None,
                        event_id=event_id,
                        evidence_article_ids=[article.id],
                        reason="no existing topic fits this event",
                    )
                )
        return proposals

    def _match_topics(self, text: str) -> list[Topic]:
        topics = [t for t in self.store.find(Topic) if t.status == "active"]
        scored: list[tuple[int, Topic]] = []
        lowered = text.casefold()
        for topic in topics:
            hits = {kw for kw in topic.keywords if _keyword_hit(lowered, kw)}
            if hits:
                scored.append((len(hits), topic))
        if not scored:
            return []
        best = max(score for score, _ in scored)
        matched = [(s, t) for s, t in scored if s >= max(1, math.ceil(best * 0.5))]
        # Prefer the specific branch, but only when it matches at least as strongly as its
        # parent; a weak child must not hide a parent that fits much better.
        by_id = {t.id: s for s, t in matched}
        matched = [
            (s, t)
            for s, t in matched
            if not any(c.parent_id == t.id and by_id.get(c.id, -1) >= s for _, c in matched)
        ]
        matched.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [t for _, t in matched[:MAX_TOPICS_PER_EVENT]]

    # ----------------------------------------------------------- validate & commit

    def _submit(self, changes: list[TreeChange]) -> TreeProposal:
        proposal = TreeProposal(id=new_id("prop"), changes=changes)
        self.validate_proposal(proposal)
        if proposal.status == "auto_committed":
            for change in changes:
                self._apply(change)
        self.store.put(proposal)
        self.trace(
            "proposal",
            proposal.status,
            "; ".join(proposal.validation_notes) or "validated by rules",
            proposal_id=proposal.id,
            kinds=[c.kind for c in changes],
        )
        return proposal

    def _check(self, change: TreeChange) -> tuple[list[str], bool]:
        """Return (problems, needs_human) for one change."""

        problems: list[str] = []
        payload = change.payload
        for aid in change.evidence_article_ids:
            if self.store.get(Article, aid) is None:
                problems.append(f"evidence article '{aid}' does not exist")
        kind = change.kind
        if kind == "create_event":
            if not change.evidence_article_ids:
                problems.append("create_event needs an evidence article")
        elif kind == "attach_article":
            if self.store.get(Event, payload.get("event_id", "")) is None:
                problems.append("event to attach to does not exist")
            article = self.store.get(Article, payload.get("article_id", ""))
            if article is None:
                problems.append("article to attach does not exist")
            elif article.injection_flags:
                problems.append("flagged article may not join an event")
            if len(payload.get("signals", [])) < MIN_AGREEING_KINDS:
                problems.append("fewer than two independent signals agree")
            if float(payload.get("score", 0)) < ATTACH_THRESHOLD:
                problems.append("similarity below the attach threshold")
        elif kind == "attach_event_to_topic":
            if self.store.get(Event, payload.get("event_id", "")) is None:
                problems.append("event does not exist")
            if self.store.get(Topic, payload.get("topic_id", "")) is None:
                problems.append("topic does not exist")
        elif kind == "create_topic":
            if not payload.get("name"):
                problems.append("topic needs a name")
            if self.store.get(Topic, topic_id_for(payload.get("slug", ""))) is not None:
                problems.append("a topic with this slug already exists")
            parent_id = payload.get("parent_id")
            if parent_id is None:
                return problems + ["new top-level topic needs a human editor"], True
            if self.store.get(Topic, parent_id) is None:
                problems.append("parent topic does not exist")
        elif kind == "merge_events":
            for key in ("keep_id", "merge_id"):
                if self.store.get(Event, payload.get(key, "")) is None:
                    problems.append(f"{key} does not exist")
            return problems + ["event merges always need a human editor"], True
        return problems, False

    def _apply(self, change: TreeChange, *, human_approved: bool = False) -> None:
        payload = change.payload
        if change.kind == "create_event":
            event = Event.model_validate(payload["event"])
            self.store.put(event)
            article = self.store.get(Article, payload["article_id"])
            if article is not None:
                article.event_id = event.id
                self.store.put(article)
            self._refresh_event(event.id)
        elif change.kind == "attach_article":
            self._attach(payload)
        elif change.kind == "attach_event_to_topic":
            event = self.store.get(Event, payload["event_id"])
            if event is not None and payload["topic_id"] not in event.topic_ids:
                event.topic_ids.append(payload["topic_id"])
                event.last_updated = utcnow()
                self.store.put(event)
        elif change.kind == "create_topic":
            status = "active" if human_approved else "proposed"
            topic = Topic(
                id=topic_id_for(payload["slug"]),
                name=payload["name"],
                slug=payload["slug"],
                parent_id=payload.get("parent_id"),
                keywords=payload.get("keywords", []),
                status=status,
            )
            self.store.put(topic)
            event_id = payload.get("event_id")
            event = self.store.get(Event, event_id) if event_id else None
            if event is not None and topic.id not in event.topic_ids:
                event.topic_ids.append(topic.id)
                self.store.put(event)
        elif change.kind == "merge_events":
            self._do_merge(payload["keep_id"], payload["merge_id"])

    def _attach(self, payload: dict[str, Any]) -> None:
        event = self.store.get(Event, payload["event_id"])
        article = self.store.get(Article, payload["article_id"])
        if event is None or article is None:
            return
        if article.id not in event.article_ids:
            event.article_ids.append(article.id)
            event.timeline.append(
                TimelineEntry(
                    at=_aware(article.published_at), text=article.title, article_id=article.id
                )
            )
        event.entities = sorted(
            set(event.entities)
            | set(payload.get("entities", []))
            | extract_entities(article_text(article))
        )
        event.place = event.place or payload.get("place")
        if article.language not in event.languages:
            event.languages.append(article.language)
        article.event_id = event.id
        self.store.put(article)
        self.store.put(event)
        self._refresh_event(event.id)

    def _refresh_event(self, event_id: str) -> None:
        """Recompute outlets, languages and status from the member articles."""

        event = self.store.get(Event, event_id)
        if event is None:
            return
        articles = self._articles_of(event)
        outlets, independent = _independence(articles)
        event.source_names = [o["name"] for o in outlets]
        event.languages = sorted({a.language for a in articles})
        times = [_aware(a.published_at) for a in articles]
        if times:
            event.first_seen = min(times)
        event.status = "confirmed" if independent >= CONFIRMED_MIN_OUTLETS else "candidate"
        event.last_updated = utcnow()
        self.store.put(event)

    def _do_merge(self, keep_id: str, merge_id: str) -> Event:
        if keep_id == merge_id:
            raise ValueError("Cannot merge an event into itself.")
        keep = self.store.get(Event, keep_id)
        gone = self.store.get(Event, merge_id)
        if keep is None or gone is None:
            raise ValueError("Both events must exist to merge them.")
        if QUARANTINE_MARK in keep.entities or QUARANTINE_MARK in gone.entities:
            raise ValueError("A quarantined event cannot be merged.")
        for article_id in gone.article_ids:
            article = self.store.get(Article, article_id)
            if article is not None:
                article.event_id = keep.id
                self.store.put(article)
            if article_id not in keep.article_ids:
                keep.article_ids.append(article_id)
        have = {(t.article_id, t.text) for t in keep.timeline}
        keep.timeline.extend(t for t in gone.timeline if (t.article_id, t.text) not in have)
        keep.entities = sorted(set(keep.entities) | set(gone.entities))
        keep.topic_ids = list(dict.fromkeys([*keep.topic_ids, *gone.topic_ids]))
        keep.place = keep.place or gone.place
        self.store.put(keep)
        gone.status = "merged"
        gone.merged_into = keep.id  # history is kept: the merged event stays readable
        self.store.put(gone)
        self._refresh_event(keep.id)
        return self.store.get(Event, keep.id) or keep

    @staticmethod
    def _require_human(reviewer: str) -> None:
        if not reviewer or not reviewer.strip():
            raise ValueError("A named human reviewer is required.")
        if reviewer.strip().casefold() in AGENT_NAMES:
            raise ValueError("Agents cannot approve tree changes.")


# ------------------------------------------------------------------------- helpers


def _clean_keywords(words: Iterable[str]) -> list[str]:
    out: list[str] = []
    for word in words:
        cleaned = guards.sanitize(str(word), max_chars=40).casefold()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out[:8]


def _keyword_hit(lowered: str, keyword: str) -> bool:
    kw = keyword.casefold()
    if len(kw) <= 3:  # EU, ice: whole words only, otherwise "euroa" would match "eu"
        return re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", lowered) is not None
    return re.search(r"(?<!\w)" + re.escape(kw), lowered) is not None


def _independence(articles: list[Article]) -> tuple[list[dict[str, Any]], int]:
    """Outlets with an independence flag.

    One broadcaster or paper is one outlet. A copy that shares a syndication key with an
    earlier report is repetition, not corroboration, so it never raises the count.
    """

    seen_outlets: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    for article in articles:  # sorted by publication time
        outlet = outlet_of(article.source_name)
        copy = bool(article.syndication_key) and article.syndication_key in seen_keys
        if article.syndication_key:
            seen_keys.add(article.syndication_key)
        if outlet in seen_outlets:
            continue
        seen_outlets[outlet] = {
            "name": outlet,
            "language": article.language,
            "access": article.access,
            "url": article.url,
            "independent": not copy,
        }
    outlets = list(seen_outlets.values())
    return outlets, sum(1 for o in outlets if o["independent"])
