"""Standards editor (copy chief): the last checkpoint before a human sees the story.

Checks are deterministic Python mapped to the Finnish Journalist's Guidelines of the
Council for Mass Media (in force 1 Oct 2024), cited by section range:

  7–17   obtaining and publishing information (check facts, acknowledge sources,
         fact vs opinion, headlines match substance, no hidden advertising, update)
  18–22  interviewees and the right of reply
  23–29  human dignity and privacy
  35–37  corrections
  L1–L4  public (community) content on the medium's own platform

An optional model "second opinion" can only ADD failures, never clear one, and if it is
unavailable the check fails closed. This agent never approves a story: it decides which
humans must.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from newsroom import guards
from newsroom.agents.base import Agent, AgentContext
from newsroom.agents.explainer import UNKNOWN_HEADING, verify_story_claims
from newsroom.domain import (
    Check,
    Claim,
    ClaimStatus,
    Community,
    ReviewRole,
    RiskTier,
    StandardsReport,
    Story,
    Translation,
    new_id,
)
from newsroom.llm.base import LLMError

_RISK_ORDER = {RiskTier.LOW: 0, RiskTier.ELEVATED: 1, RiskTier.HIGH: 2}
# Blocking even in hackathon mode: no story goes out empty or without a linked source.
ALWAYS_BLOCKING = {"Story has a body", "Sources acknowledged with links"}
_STOP = {
    "what", "this", "that", "with", "your", "from", "about", "will", "have", "they", "their",
    "means", "mean", "here", "next", "into", "more", "than", "when", "which", "for", "the",
    "and", "you", "are", "was", "how",
}  # fmt: skip

_SENSATIONAL = re.compile(
    r"(!{2,}|\bshocking\b|you won'?t believe|\boutrage(ous)?\b|\bscandal\b|\bexposed\b)", re.I
)
_AD_MARKERS = re.compile(
    r"\b(sponsored|promo code|discount code|use code|affiliate|buy now|limited offer|advertorial)\b",
    re.I,
)
_OPINION = re.compile(
    r"\b(we believe|in our opinion|in my opinion|the best|the worst|clearly wrong|outrageous)\b",
    re.I,
)
_NEGATIVE = re.compile(
    r"\b(fraud|illegal(ly)?|scam|exploit(s|ed|ing|ation)?|abus(e|es|ed|ive)|violat(e|es|ed|ion)|"
    r"unlawful(ly)?|discriminat\w*|negligen\w*|corrupt\w*|charges? more|breach(es|ed)?)\b",
    re.I,
)
_STIGMA = re.compile(
    r"(\billegal (immigrants?|aliens?)\b|\billegals\b|\bfreeloaders?\b|\bparasites?\b|"
    r"\bthese people\b|\bforeigners are\b|"
    r"\b(refugees|immigrants|muslims|romani|gypsies|sámi|sami|foreigners) are (all )?"
    r"(criminals|lazy|dangerous|terrorists|a burden)\b)",
    re.I,
)
_PRIVATE = [
    ("an e-mail address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    (
        "a phone number",
        re.compile(
            r"(?<!\d)(?:\+\d{1,3}(?:[\s-]?\d{2,4}){2,4}|0\d{1,2}[\s-]?\d{3,4}[\s-]?\d{2,4})(?!\d)"
        ),
    ),
    ("a personal identity code", re.compile(r"\b\d{6}[-+A]\d{3}[0-9A-Y]\b")),
]
_AI_WORDS = re.compile(r"\b(ai|artificial intelligence|tekoäly\w*|machine)\b", re.I)
_MISLEADING_AI = re.compile(r"\b(written (entirely )?by humans?|human[- ]written|no ai)\b", re.I)


class SecondOpinion(BaseModel):
    headline_matches: bool = True
    tone_ok: bool = True
    concerns: list[str] = Field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.casefold()) if len(t) >= 4 and t not in _STOP}


def _max_risk(*tiers: RiskTier) -> RiskTier:
    return max(tiers, key=lambda tier: _RISK_ORDER[tier])


class StandardsEditorAgent(Agent):
    name = "standards_editor"

    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(ctx)

    def review(
        self,
        story: Story,
        claims: list[Claim],
        translations: list[Translation],
        community: Community,
    ) -> StandardsReport:
        text = f"{story.headline}\n\n{story.standfirst}\n\n{story.body_md}"
        risk = _max_risk(story.risk_tier, guards.classify_risk(text, self.policy)[0])
        checks: list[Check] = []
        required: list[ReviewRole] = [ReviewRole.EDITOR_IN_CHIEF]

        def add(
            guideline: str,
            name: str,
            passed: bool,
            note: str = "",
            blocking: bool = False,
            info: bool = False,
        ):
            # Notes explain failures; passing checks stay quiet unless the note is information.
            checks.append(
                Check(
                    guideline=guideline,
                    name=name,
                    passed=passed,
                    note=note if (info or not passed) else "",
                    # Hackathon mode: findings are shown to the editor but do not stop a
                    # publish, except an empty draft. STRICT_STANDARDS=true restores blocking.
                    blocking=blocking
                    and (self.ctx.settings.strict_standards or name in ALWAYS_BLOCKING),
                )
            )

        # -- 7–17 obtaining and publishing information ---------------------------------
        add("7–17", "Story has a body", bool(story.body_md.strip()), "Empty draft.", blocking=True)

        problems = verify_story_claims(text, claims)
        add(
            "7–17", "Claims worded within the evidence", not problems,
            "; ".join(problems), blocking=True,
        )  # fmt: skip

        unsourced = [
            c.text
            for c in claims
            if c.status in (ClaimStatus.VERIFIED, ClaimStatus.ATTRIBUTED)
            and not c.evidence
            and not c.source_article_id
        ]
        add(
            "7–17", "Stated claims have sources", not unsourced,
            "No source recorded for: " + "; ".join(unsourced), blocking=True,
        )  # fmt: skip

        bad_links = [
            s.name for s in story.sources if not s.url.lower().startswith(("http://", "https://"))
        ]
        add(
            "7–17", "Sources acknowledged with links",
            bool(story.sources) and not bad_links,
            "No sources listed." if not story.sources else f"Missing link: {', '.join(bad_links)}",
            blocking=True,
        )  # fmt: skip

        outlets = {s.name for s in story.sources if s.origin != "agent_reference"}
        add(
            "7–17", "Two or more sources where available", len(outlets) >= 2,
            f"Only {len(outlets)} source reported this. Look for a second, independent source, "
            "or keep the story clearly attributed to the one it has.",
        )  # fmt: skip

        opinion = _OPINION.findall(text)
        add(
            "7–17", "Fact and opinion kept apart", not opinion,
            "Unattributed opinion wording found; attribute it or remove it.",
        )  # fmt: skip

        overlap_ok, overlap_note = self._headline_overlap(story)
        add("7–17", "Headline matches substance", overlap_ok, overlap_note, blocking=True)

        sensational = _SENSATIONAL.search(story.headline)
        add(
            "7–17", "Headline is not sensational", sensational is None,
            f"Sensational wording: {sensational.group(0)!r}" if sensational else "", blocking=True,
        )  # fmt: skip

        second = self._second_opinion(story)
        add("7–17", "Second reading of headline and tone", second[0], second[1], blocking=True)

        ad = _AD_MARKERS.search(text)
        add(
            "7–17", "No hidden advertising", ad is None,
            f"Promotional wording: {ad.group(0)!r}" if ad else "", blocking=True,
        )  # fmt: skip

        unknowns = [u for c in claims for u in c.unknowns] + [
            c.text for c in claims if c.status is ClaimStatus.UNKNOWN
        ]
        add(
            "7–17", "Open questions are shown to readers",
            not unknowns or UNKNOWN_HEADING in story.body_md.casefold(),
            "Add a 'What we don't know yet' section; update the story as facts emerge.",
        )  # fmt: skip

        # -- 18–22 interviewees and the right of reply ----------------------------------
        reply_targets = self._reply_targets(claims, story)
        add(
            "18–22", "Right of reply considered", not reply_targets,
            "Strongly negative statements about a named party; seek a reply before "
            "publishing: " + "; ".join(reply_targets),
        )  # fmt: skip

        # -- 23–29 dignity and privacy --------------------------------------------------
        stigma = _STIGMA.search(text)
        add(
            "23–29", "No stigmatising wording", stigma is None,
            f"Wording that stigmatises a group: {stigma.group(0)!r}" if stigma else "",
            blocking=True,
        )  # fmt: skip
        private = [label for label, pattern in _PRIVATE if pattern.search(text)]
        add(
            "23–29", "No private details", not private,
            "Contains " + ", ".join(private) if private else "", blocking=True,
        )  # fmt: skip

        # -- 35–37 corrections ------------------------------------------------------------
        disclosure = story.ai_disclosure or ""
        add(
            "35–37", "Readers can report an error",
            bool(re.search(r"(wrong|correct|virhe|oikais)", disclosure, re.I)),
            "The disclosure must tell readers how to flag a mistake so corrections can be "
            "published.", blocking=True,
        )  # fmt: skip

        # -- L1–L4 public content ---------------------------------------------------------
        unlabeled = [
            n.text
            for n in story.context_notes
            if n.kind == "community_knowledge" and n.text.casefold() not in story.body_md.casefold()
        ]
        labelled = "community note" in story.body_md.casefold() or not any(
            n.kind == "community_knowledge" for n in story.context_notes
        )
        add(
            "L1–L4", "Community content labelled", labelled and not unlabeled,
            "Community-contributed context must appear as a labelled community note.",
            blocking=True,
        )  # fmt: skip

        # -- AI disclosure ----------------------------------------------------------------
        add(
            "AI", "AI role is disclosed",
            bool(_AI_WORDS.search(disclosure)) and not _MISLEADING_AI.search(disclosure),
            "Every story must say it was prepared with AI assistance and approved by a human.",
            blocking=True,
        )  # fmt: skip

        # -- RISK -------------------------------------------------------------------------
        official = any(s.official for s in story.sources)
        if risk is RiskTier.HIGH:
            add(
                "RISK", "High-risk story links an official source", official,
                "Permit, health, employment, legal or safety stories must link the official "
                "source.", blocking=True,
            )  # fmt: skip
            for role_name in self.policy.review.high_risk_roles:
                role = ReviewRole(role_name)
                if role is ReviewRole.COMMUNITY_REVIEWER and not community.trusted_voices:
                    continue
                required.append(role)
        add(
            "RISK", f"Risk tier {risk.value}", True,
            "High risk: extra named reviewers must approve." if risk is RiskTier.HIGH else "",
            info=True,
        )  # fmt: skip
        if community.trusted_voices and risk is not RiskTier.HIGH:
            add(
                "RISK", "Community review recommended", True,
                "A trusted voice from the community can review this story.",
                info=True,
            )  # fmt: skip

        human_needed = [n.text for n in story.context_notes if n.kind == "human_needed"]
        add(
            "RISK", "Ungrounded context resolved", not human_needed,
            "A human must supply or remove: " + "; ".join(human_needed),
        )  # fmt: skip

        # -- translations -----------------------------------------------------------------
        for translation in translations:
            lane = community.lane(translation.language)
            needs_language_reviewer = (
                translation.status == "needs_human_translation"
                or bool(translation.backtranslation_flags)
                or (
                    translation.status == "machine_draft"
                    and lane is not None
                    and lane.human_review in ("required", "human_only")
                )
            )
            if needs_language_reviewer and translation.status != "needs_human_translation":
                add(
                    "7–17", f"Language review ({translation.language})", True,
                    "This language goes out when a language reviewer from the community "
                    "approves it. The story is not held back for it.",
                    info=True,
                )  # fmt: skip
            if translation.backtranslation_flags:
                add(
                    "7–17", f"Translation checks ({translation.language})", False,
                    "; ".join(translation.backtranslation_flags),
                )  # fmt: skip
            if translation.status == "needs_human_translation":
                add(
                    "7–17", f"Human translation needed ({translation.language})", True,
                    "Not publishable until a language reviewer supplies and approves the text.",
                    info=True,
                )  # fmt: skip

        required = list(dict.fromkeys(required))
        report = StandardsReport(
            id=new_id("std"),
            story_id=story.id,
            story_version=story.version,
            checks=checks,
            risk_tier=risk,
            required_reviews=required,
            revision_feedback=self._feedback(checks, human_needed, reply_targets),
        )
        self.store.put(report)
        failures = [c.name for c in report.blocking_failures]
        self.trace(
            "review",
            "blocking issues" if failures else "ready for human review",
            "; ".join(failures) or f"requires {', '.join(r.value for r in required)}",
            story_id=story.id, version=story.version, risk=risk.value,
        )  # fmt: skip
        return report

    # ---------------------------------------------------------------------------------

    @staticmethod
    def _headline_overlap(story: Story) -> tuple[bool, str]:
        headline = _tokens(story.headline)
        if not headline:
            return True, ""
        rest = _tokens(f"{story.standfirst} {story.body_md}")
        ratio = len(headline & rest) / len(headline)
        return ratio >= 0.5, f"Only {ratio:.0%} of the headline's words appear in the story."

    def _second_opinion(self, story: Story) -> tuple[bool, str]:
        try:
            opinion = self.llm.generate(
                task="standards.second_opinion",
                system=(
                    "You are a copy editor. Say whether the headline matches the story and whether "
                    "the tone is neutral. Report concerns; do not rewrite.\n"
                    + guards.UNTRUSTED_PREAMBLE
                ),
                prompt=guards.wrap_untrusted(
                    f"{story.headline}\n\n{story.standfirst}\n\n{story.body_md}", "draft"
                ),
                schema=SecondOpinion,
                payload={
                    "headline": story.headline,
                    "standfirst": story.standfirst,
                    "body_md": story.body_md,
                },
            )
        except LLMError as exc:
            self.trace("second_opinion", "unavailable, failing closed", str(exc)[:300])
            return False, "Second reading unavailable, so the story cannot be cleared."
        ok = opinion.headline_matches and opinion.tone_ok
        return ok, "; ".join(opinion.concerns) or ("" if ok else "Headline or tone concern.")

    @staticmethod
    def _reply_targets(claims: list[Claim], story: Story) -> list[str]:
        targets: list[str] = []
        pool = [c.text for c in claims] + re.split(r"(?<=[.!?])\s+", story.body_md)
        for sentence in pool:
            if not _NEGATIVE.search(sentence):
                continue
            has_party = re.search(
                r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b", sentence
            ) or re.search(
                r"\b(employers?|compan(y|ies)|municipality|authorit(y|ies)|landlords?)\b",
                sentence,
                re.I,
            )
            if has_party and sentence.strip() not in targets:
                targets.append(sentence.strip()[:160])
        return targets[:3]

    @staticmethod
    def _feedback(
        checks: list[Check], human_needed: list[str], reply_targets: list[str]
    ) -> list[str]:
        feedback = [
            f"Fix '{c.name}' ({c.guideline}): {c.note}".strip()
            for c in checks
            if c.blocking and not c.passed
        ]
        feedback += [f"Human needed: supply or remove this context: {t}" for t in human_needed]
        feedback += [f"Seek a reply before publishing: {t}" for t in reply_targets]
        return feedback


# ------------------------------------------------------------------- offline stand-in


def _fake_second_opinion(payload: dict[str, Any]) -> SecondOpinion:
    headline = payload["headline"]
    concerns = []
    if _SENSATIONAL.search(headline):
        concerns.append("Headline wording is sensational.")
    overlap = _tokens(headline)
    rest = _tokens(f"{payload['standfirst']} {payload['body_md']}")
    if overlap and len(overlap & rest) / len(overlap) < 0.5:
        concerns.append("Headline words do not appear in the story.")
    return SecondOpinion(
        headline_matches=not concerns, tone_ok=not _SENSATIONAL.search(headline), concerns=concerns
    )


FAKE_HANDLERS: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
    "standards.second_opinion": _fake_second_opinion,
}
