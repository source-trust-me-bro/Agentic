"""Topic Connector: links semantically related topics through the backend API.

The graph lives in the application backend (KalevaShorts API). This agent reads
the topic tree over REST, shortlists candidate pairs with deterministic lexical
scores, lets the model confirm which candidates are *meaningfully* related, and
submits each accepted relationship with the backend's documented
``POST /topics/{id}/branches``. The backend validates, stores the relationship
symmetrically and stays idempotent; the agent never writes to any database.

Division of labour (the rule of this codebase):

* Candidate retrieval is CODE: cheap token/keyword overlap over topic names,
  descriptions and keywords, deduplicated and capped per run so one run never
  grows into an exhaustive model comparison as the topic count grows.
* The model only JUDGES the shortlisted pairs (one batched call). If it is
  unavailable or disabled, code falls back to a conservative lexical threshold.
* Persistence goes through the backend API, which owns integrity: repeated runs
  converge because already-related pairs are recognised from the tree and never
  resubmitted, and backend rejections never touch other topics.

Every stage is a separate method so retrieval, evaluation and persistence can be
tested independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent
from newsroom.backend_client import BackendClient, BackendError, client_from_settings
from newsroom.llm.base import LLMError

_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)

MIN_TOKEN_LEN = 3
MAX_REASON_CHARS = 200

# ------------------------------------------------------------------- model schema


class LinkJudgement(BaseModel):
    """The model's verdict on one candidate pair. Code validates every field."""

    topic_a: str
    topic_b: str
    related: bool = False
    reason: str = ""


class LinkEvaluation(BaseModel):
    judgements: list[LinkJudgement] = Field(default_factory=list)


class AcceptedPair(BaseModel):
    first_id: str
    second_id: str
    score: float
    reason: str


class FailedPair(BaseModel):
    first_id: str
    second_id: str
    error: str


class ConnectorResult(BaseModel):
    backend_configured: bool = False
    topics_considered: int = 0
    candidates: int = 0
    accepted: list[AcceptedPair] = Field(default_factory=list)
    skipped_existing: int = 0
    submitted: list[AcceptedPair] = Field(default_factory=list)
    failed: list[FailedPair] = Field(default_factory=list)


# ------------------------------------------------------- candidate scoring (code)


def normalize_pair(first_id: str, second_id: str) -> tuple[str, str] | None:
    """Canonical, duplicate-free pair key. Returns None for self-links or bad ids."""

    if not first_id or not second_id or first_id == second_id:
        return None
    return tuple(sorted((first_id, second_id)))


@dataclass
class TopicSignature:
    id: str
    name: str
    stems: set[str]
    keywords: set[str]
    text: str


def topic_text(topic: dict[str, Any]) -> str:
    parts = [
        str(topic.get("name", "")),
        str(topic.get("description", "") or ""),
        " ".join(str(k) for k in topic.get("keywords") or []),
    ]
    return guards.sanitize(" . ".join(p for p in parts if p))


def _stems(text: str, n: int = 5) -> set[str]:
    return {
        token.casefold()[:n].strip("-")
        for token in _WORD.findall(text)
        if len(token) >= MIN_TOKEN_LEN
    }


def signature_for(topic: dict[str, Any]) -> TopicSignature:
    text = str(topic.get("text") or topic_text(topic))
    return TopicSignature(
        id=str(topic.get("id", "")),
        name=str(topic.get("name", "")),
        stems=_stems(text),
        keywords={str(k).strip().casefold() for k in topic.get("keywords") or [] if str(k).strip()},
        text=text,
    )


def candidate_score(first: TopicSignature, second: TopicSignature) -> float:
    """Lexical affinity in [0, 1]. Cheap prefilter: it shortlists, never promises."""

    stems_union = first.stems | second.stems
    stems = len(first.stems & second.stems) / len(stems_union) if stems_union else 0.0
    kw_union = first.keywords | second.keywords
    keywords = len(first.keywords & second.keywords) / len(kw_union) if kw_union else 0.0
    return round(min(1.0, 0.5 * stems + 0.5 * keywords), 4)


def shortlist_candidates(
    topics: list[dict[str, Any]],
    *,
    prefilter_score: float,
    max_candidates: int,
) -> list[AcceptedPair]:
    """Deterministic shortlist of possibly-related pairs (no model calls, no repeats).

    Pairs are deduplicated (A-B and B-A are the same relationship), self-links
    are impossible by construction, and the list is capped so the cost per run
    stays bounded as the topic count grows.
    """

    signatures = {topic["id"]: signature_for(topic) for topic in topics}
    scored: list[AcceptedPair] = []
    seen: set[tuple[str, str]] = set()
    ids = sorted(topics, key=lambda t: t["id"])
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            key = normalize_pair(first["id"], second["id"])
            if key is None or key in seen:
                continue
            seen.add(key)
            score = candidate_score(signatures[first["id"]], signatures[second["id"]])
            if score >= prefilter_score:
                scored.append(
                    AcceptedPair(
                        first_id=key[0],
                        second_id=key[1],
                        score=score,
                        reason=f"lexical prefilter score {score}",
                    )
                )
    scored.sort(key=lambda p: (-p.score, p.first_id, p.second_id))
    return scored[:max_candidates]


# ------------------------------------------------------------------- fake (offline)


def _fake_evaluate(payload: dict[str, Any]) -> LinkEvaluation:
    """Deterministic offline stand-in: related only on >= 2 shared specific stems."""

    judgements = []
    for candidate in payload.get("candidates") or []:
        shared = _stems(str(candidate.get("topic_a_details", ""))) & _stems(
            str(candidate.get("topic_b_details", ""))
        )
        judgements.append(
            LinkJudgement(
                topic_a=str(candidate.get("topic_a", "")),
                topic_b=str(candidate.get("topic_b", "")),
                related=len(shared) >= 2,
                reason=("shared specific stems: " + ", ".join(sorted(shared)))
                if shared
                else "no specific shared subject matter",
            )
        )
    return LinkEvaluation(judgements=judgements)


FAKE_HANDLERS = {"connector_evaluate": _fake_evaluate}

# --------------------------------------------------------------------------- model

SYSTEM = (
    "You are the topic connector of a newsroom. You get EXISTING topic-tree nodes and "
    "candidate pairs between them. For every pair decide whether the two topics have a "
    "meaningful semantic relationship: shared events, shared named subject matter, or a "
    "contextual connection a reader would follow. Topics that only share generic words, "
    "broad categories, or sit near each other in the tree are NOT related. Never invent "
    "topic ids: answer with the ids exactly as given. Answer with JSON matching the "
    "schema.\n" + guards.UNTRUSTED_PREAMBLE
)


# --------------------------------------------------------------------------- agent


def _pair_of(judgement: LinkJudgement) -> tuple[str, str] | None:
    return normalize_pair(judgement.topic_a, judgement.topic_b)


class TopicConnectorAgent(Agent):
    name = "topic_connector"

    def run(self, client: BackendClient | None = None) -> ConnectorResult:
        """Retrieve topics, judge relationships and submit the accepted ones.

        A prepared ``client`` can be injected (tests, runbooks); otherwise one is
        built from settings. Without ``backend_api_url`` the agent reports that
        persistence is not configured and submits nothing.
        """

        result = ConnectorResult()
        client = client if client is not None else client_from_settings(self.settings)
        if client is None:
            self.trace(
                "run", "not_configured", "backend_api_url is empty; nothing retrieved or submitted"
            )
            return result
        result.backend_configured = True

        topics = self._retrieve(client, result)
        if result.topics_considered < 2:
            self.trace(
                "run",
                "not_enough_topics",
                f"{result.topics_considered} topic(s) retrieved; nothing to compare",
            )
            return result

        known = self._existing_pairs(topics)
        by_id = {t["id"]: t for t in topics}
        candidates = shortlist_candidates(
            topics,
            prefilter_score=self.settings.connector_prefilter_score,
            max_candidates=self.settings.connector_max_candidates,
        )
        result.candidates = len(candidates)
        self.trace(
            "candidates",
            "shortlisted",
            f"{len(candidates)} candidate pair(s) from {result.topics_considered} topics "
            f"(prefilter >= {self.settings.connector_prefilter_score})",
        )
        if not candidates:
            self.trace("run", "done", "no candidate pairs passed the prefilter")
            return result

        accepted = self._evaluate(candidates, by_id)
        result.accepted = accepted
        fresh = [
            pair for pair in accepted if normalize_pair(pair.first_id, pair.second_id) not in known
        ]
        result.skipped_existing = len(accepted) - len(fresh)
        if result.skipped_existing:
            self.trace(
                "skip_existing",
                "skipped",
                f"{result.skipped_existing} accepted pair(s) already related in the backend",
            )
        if not fresh:
            self.trace("run", "done", "accepted relationships are all already linked")
            return result

        self._submit(fresh, client, result)
        self.trace(
            "run",
            "done",
            f"submitted {len(result.submitted)}, failed {len(result.failed)}, "
            f"skipped existing {result.skipped_existing}",
        )
        return result

    # ------------------------------------------------------------------ retrieval

    def _retrieve(self, client: BackendClient, result: ConnectorResult) -> list[dict[str, Any]]:
        try:
            topics = client.list_topics()
        except BackendError as exc:
            result.failed.append(
                FailedPair(first_id="", second_id="", error=f"retrieval failed: {exc}")
            )
            self.trace("retrieve", "failed", f"could not read the backend topic tree: {exc}")
            return []
        result.topics_considered = len(topics)
        self.trace(
            "retrieve",
            "retrieved",
            f"{len(topics)} topic(s) readable from the backend tree",
            topic_count=len(topics),
        )
        return topics

    @staticmethod
    def _existing_pairs(topics: list[dict[str, Any]]) -> set[tuple[str, str]]:
        """Relationships the backend already stores (symmetric on both documents)."""

        known: set[tuple[str, str]] = set()
        for topic in topics:
            for branch_id in topic.get("branches") or []:
                key = normalize_pair(str(topic["id"]), str(branch_id))
                if key is not None:
                    known.add(key)
        return known

    # ------------------------------------------------------------------ evaluation

    def _evaluate(
        self, candidates: list[AcceptedPair], by_id: dict[str, dict[str, Any]]
    ) -> list[AcceptedPair]:
        """Judge each candidate pair. Model when enabled, conservative fallback otherwise."""

        threshold = self.settings.connector_min_similarity
        lexical = [p for p in candidates if p.score >= threshold]
        if not self.settings.connector_uses_model:
            self.trace(
                "evaluate",
                "lexical_only",
                f"model disabled; {len(lexical)} pair(s) pass threshold {threshold}",
            )
            return lexical
        try:
            judgements = self._model_judgement(candidates, by_id)
        except LLMError as exc:
            self.trace("evaluate", "degraded", f"model unavailable: {exc}; using lexical threshold")
            return lexical
        return self._validate_judgements(judgements, candidates, by_id)

    def _model_judgement(
        self, candidates: list[AcceptedPair], by_id: dict[str, dict[str, Any]]
    ) -> list[LinkJudgement]:
        payload_candidates = []
        for pair in candidates:
            payload_candidates.append(
                {
                    "topic_a": pair.first_id,
                    "topic_a_name": by_id.get(pair.first_id, {}).get("name", ""),
                    "topic_a_details": topic_text(by_id.get(pair.first_id, {})),
                    "topic_b": pair.second_id,
                    "topic_b_name": by_id.get(pair.second_id, {}).get("name", ""),
                    "topic_b_details": topic_text(by_id.get(pair.second_id, {})),
                }
            )
        prompt = "Judge these topic pairs.\n" + guards.wrap_untrusted(
            "\n".join(
                f"{c['topic_a']} <t>{c['topic_a_name']} — {c['topic_a_details']}</t> :: "
                f"{c['topic_b']} <t>{c['topic_b_name']} — {c['topic_b_details']}</t>"
                for c in payload_candidates
            ),
            "topic tree from the application backend",
        )
        raw = self.llm.generate(
            task="connector_evaluate",
            system=SYSTEM,
            prompt=prompt,
            schema=LinkEvaluation,
            payload={"candidates": payload_candidates},
        )
        return list(raw.judgements)[: len(candidates) * 2]

    def _validate_judgements(
        self,
        judgements: list[LinkJudgement],
        candidates: list[AcceptedPair],
        by_id: dict[str, dict[str, Any]],
    ) -> list[AcceptedPair]:
        """Keep verdicts that reference real, shortlisted pairs; the model only proposes."""

        allowed = {(normalize_pair(p.first_id, p.second_id)): p for p in candidates}
        accepted: dict[tuple[str, str], AcceptedPair] = {}
        for judgement in judgements:
            key = _pair_of(judgement)
            if key is None or key not in allowed:
                continue  # self-links, invented pairs or malformed ids are dropped here
            if not judgement.related:
                continue
            candidate = allowed[key]
            if key in accepted:
                continue
            reason = guards.sanitize(judgement.reason, max_chars=MAX_REASON_CHARS)
            accepted[key] = AcceptedPair(
                first_id=key[0],
                second_id=key[1],
                score=candidate.score,
                reason=reason or candidate.reason,
            )
        self.trace(
            "evaluate",
            "model_judged",
            f"{len(judgements)} verdict(s); "
            f"{sum(1 for j in judgements if j.related and _pair_of(j) in allowed)} valid related, "
            f"{len(accepted)} accepted",
        )
        return [accepted[key] for key in sorted(accepted)]

    # ------------------------------------------------------------------ persistence

    def _submit(
        self,
        pairs: list[AcceptedPair],
        client: BackendClient,
        result: ConnectorResult,
    ) -> None:
        """Hand each accepted pair to the backend. The backend owns idempotency."""

        for pair in pairs:
            try:
                client.link_topics(pair.first_id, pair.second_id)
            except BackendError as exc:
                result.failed.append(
                    FailedPair(
                        first_id=pair.first_id,
                        second_id=pair.second_id,
                        error=str(exc)[:300],
                    )
                )
                self.trace(
                    "submit",
                    "failed",
                    f"{pair.first_id} <-> {pair.second_id}: {exc}",
                    first_id=pair.first_id,
                    second_id=pair.second_id,
                )
                continue
            result.submitted.append(pair)
            self.trace(
                "submit",
                "linked",
                f"{pair.first_id} <-> {pair.second_id} ({pair.reason})",
                first_id=pair.first_id,
                second_id=pair.second_id,
                score=pair.score,
            )
