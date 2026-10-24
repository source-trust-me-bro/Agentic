"""Distributor: turns an APPROVED story into channel-ready output.

Defence in depth: the graph already waits for human approval, and this agent checks the
approvals again and refuses (``PublishRefused``) when anything is missing. Nothing is
silently machine-published in a language that lacks human review.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from newsroom import guards
from newsroom.agents.base import Agent, AgentContext
from newsroom.domain import (
    Channel,
    Community,
    Decision,
    Publication,
    Rendition,
    ReviewDecision,
    ReviewRole,
    RiskTier,
    StandardsReport,
    Story,
    StoryStatus,
    Translation,
    new_id,
    utcnow,
)

# Names that belong to software, never to a human reviewer.
_NON_HUMAN_REVIEWERS = {
    "explainer", "translator", "standards_editor", "distributor", "monitor", "news_editor",
    "investigator", "community_desk", "agent", "ai", "bot", "system", "llm",
}  # fmt: skip

SHORT_DISCLOSURE = {
    "en": "AI-assisted, approved by a human editor.",
    "fi": "Tekoälyavusteinen, toimittajan hyväksymä.",
}
CORRECTION_LINE = {
    "en": "Spotted a mistake? Tell us and we will publish a correction.",
    "fi": "Huomasitko virheen? Kerro meille, niin julkaisemme oikaisun.",
}


class PublishRefused(ValueError):
    """The story is not approved well enough to be published."""


# ---------------------------------------------------------------------- text helpers

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell_int(number: int) -> str:
    """English words for 0 <= number < 1,000,000 (read-aloud scripts)."""

    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, rest = divmod(number, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {spell_int(rest)}" if rest else "")
    if number < 1_000_000:
        thousands, rest = divmod(number, 1000)
        return f"{spell_int(thousands)} thousand" + (f" {spell_int(rest)}" if rest else "")
    return str(number)


def _spell_numbers(text: str) -> str:
    text = re.sub(
        r"€\s?(\d+)|(\d+)\s?(?:€|euros?|EUR)\b",
        lambda m: f"{spell_int(int(m.group(1) or m.group(2)))} euros",
        text,
    )
    text = re.sub(r"(\d+)\s?%", lambda m: f"{spell_int(int(m.group(1)))} percent", text)
    text = re.sub(
        r"(\d+)\.(\d+)",
        lambda m: (
            f"{spell_int(int(m.group(1)))} point " + " ".join(spell_int(int(d)) for d in m.group(2))
        ),
        text,
    )
    return re.sub(
        r"\d+", lambda m: spell_int(int(m.group(0))) if len(m.group(0)) < 7 else m.group(0), text
    )


def _plain(text: str) -> str:
    """Strip markdown and URLs."""

    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^[>#\s*-]+", "", text, flags=re.M)
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _short_sentences(text: str, limit: int = 22) -> list[str]:
    out: list[str] = []
    for sentence in _sentences(text):
        if len(sentence.split()) <= limit:
            out.append(sentence)
            continue
        for part in re.split(r";\s+|,\s+(?=and |but |while |which )", sentence):
            part = part.strip().rstrip(",;")
            if part:
                out.append(part if part[-1] in ".!?" else part + ".")
    return out


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 1, 0)].rstrip() + "…"


# ------------------------------------------------------------------------------ agent


class DistributorAgent(Agent):
    name = "distributor"

    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(ctx)

    # -- publication ----------------------------------------------------------------

    def render(
        self,
        story: Story,
        translations: list[Translation],
        community: Community,
        approvals: list[ReviewDecision],
    ) -> Publication:
        used = self._require_approvals(story, approvals)
        report = self._latest_report(story)
        risk = max(
            (story.risk_tier, report.risk_tier),
            key=lambda tier: {RiskTier.LOW: 0, RiskTier.ELEVATED: 1, RiskTier.HIGH: 2}[tier],
        )

        channels = community.channels or [Channel(kind="web", label="Web feed")]
        languages = self._publishable_languages(story, translations, community, risk)

        renditions: list[Rendition] = []
        for channel in channels:
            for code, text in languages:
                renditions.append(self._render_channel(story, channel, code, text))

        publication = Publication(
            id=new_id("pub"),
            story_id=story.id,
            story_version=story.version,
            community_id=community.id,
            renditions=renditions,
            approval_ids=[d.id for d in used],
        )
        self.store.put(publication)

        story.status = StoryStatus.PUBLISHED
        story.updated_at = utcnow()
        self.store.put(story)
        self.trace(
            "publish", "published", f"{len(renditions)} renditions",
            story_id=story.id, version=story.version, approvals=[d.id for d in used],
        )  # fmt: skip
        return publication

    def onboarding_pack(
        self, community: Community, publications: list[Publication], *, max_items: int = 5
    ) -> list[Rendition]:
        """First-week messages for people who have just arrived for a short season."""

        if not community.lifecycle.seasonal:
            return []
        candidates: list[tuple[int, Any, Rendition]] = []
        for publication in publications:
            if publication.community_id != community.id:
                continue
            story = self.store.get(Story, publication.story_id)
            if story is None:
                continue
            message = next(
                (
                    r
                    for r in publication.renditions
                    if r.channel == "messaging" and r.language == story.language
                ),
                None,
            )
            if message is None:
                continue
            useful = 1 if any(str(n) == "usefulness" for n in story.needs) else 0
            candidates.append((useful, publication.published_at, message))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        pack = [rendition for _, _, rendition in candidates[:max_items]]
        self.trace("onboarding_pack", "assembled", f"{len(pack)} item(s)", community=community.id)
        return pack

    def bridge_digest(
        self,
        communities: list[Community],
        publications: list[Publication],
        *,
        language: str = "fi",
    ) -> Rendition | None:
        """Stories ABOUT the communities, for Finnish-speaking readers."""

        names = {c.id: c.name for c in communities}
        lines: list[str] = []
        for publication in sorted(publications, key=lambda p: p.published_at, reverse=True)[:10]:
            if publication.community_id not in names:
                continue
            story = self.store.get(Story, publication.story_id)
            if story is None:
                continue
            reviewed = next(
                (
                    t
                    for t in self.store.find(Translation, story_id=story.id, language=language)
                    if t.status == "human_reviewed" and t.story_version == story.version
                ),
                None,
            )
            if reviewed and reviewed.headline:
                lines.append(f"- {names[publication.community_id]}: {reviewed.headline}")
            else:
                suffix = " (englanniksi)" if language == "fi" else ""
                lines.append(f"- {names[publication.community_id]}: {story.headline}{suffix}")
        if not lines:
            return None
        heading = (
            "Näin kerroimme yhteisöistä viime aikoina:"
            if language == "fi"
            else "Recent stories about the communities:"
        )
        text = (
            heading
            + "\n"
            + "\n".join(lines)
            + "\n"
            + SHORT_DISCLOSURE.get(language, SHORT_DISCLOSURE["en"])
        )
        return Rendition(channel="digest", language=language, text=text)

    # -- checks -----------------------------------------------------------------------

    def _latest_report(self, story: Story) -> StandardsReport:
        reports = [
            r
            for r in self.store.find(StandardsReport, story_id=story.id)
            if r.story_version == story.version
        ]
        if not reports:
            raise PublishRefused("No standards review exists for this story version.")
        report = reports[-1]
        if report.blocking_failures:
            raise PublishRefused(
                "The standards review has blocking failures: "
                + "; ".join(c.name for c in report.blocking_failures)
            )
        return report

    def _require_approvals(
        self, story: Story, approvals: list[ReviewDecision]
    ) -> list[ReviewDecision]:
        if story.status is StoryStatus.REJECTED:
            raise PublishRefused("A rejected story cannot be published.")
        report = self._latest_report(story)
        required = list(dict.fromkeys([ReviewRole.EDITOR_IN_CHIEF, *report.required_reviews]))

        latest: dict[ReviewRole, ReviewDecision] = {}
        for decision in sorted(approvals, key=lambda d: d.decided_at):
            if decision.story_id != story.id or decision.story_version != story.version:
                continue
            name = decision.reviewer.strip()
            if not name or name.casefold() in _NON_HUMAN_REVIEWERS:
                continue  # an empty or software name never counts as a human reviewer
            latest[decision.role] = decision

        used: list[ReviewDecision] = []
        for role in required:
            decision = latest.get(role)
            if decision is None or decision.decision is not Decision.APPROVE:
                self.trace(
                    "publish", "refused", f"missing approval: {role.value}", story_id=story.id
                )
                raise PublishRefused(
                    f"Missing a named human approval from {role.value} for version {story.version}."
                )
            used.append(decision)
        return used

    def _publishable_languages(
        self, story: Story, translations: list[Translation], community: Community, risk: RiskTier
    ) -> list[tuple[str, tuple[str, str, str]]]:
        """(code, (headline, standfirst, body)) for every language allowed to go out."""

        out = [(story.language, (story.headline, story.standfirst, story.body_md))]
        for lane in community.languages:
            if lane.code == story.language:
                continue
            candidates = [
                t
                for t in translations
                if t.language == lane.code
                and t.story_version == story.version
                and t.body_md.strip()
            ]
            translation = candidates[-1] if candidates else None
            if translation is None:
                self.trace(
                    "language", "skipped", f"no usable {lane.code} translation", lane=lane.code
                )
                continue
            human_ok = translation.status == "human_reviewed"
            machine_ok = (
                translation.status == "machine_draft"
                and lane.tier == "A"
                and lane.human_review == "spot_check"
                and risk is not RiskTier.HIGH
                and not translation.backtranslation_flags
            )
            if not (human_ok or machine_ok):
                self.trace(
                    "language", "skipped",
                    f"{lane.code} needs human review before it can go out", lane=lane.code,
                )  # fmt: skip
                continue
            out.append(
                (lane.code, (translation.headline, translation.standfirst, translation.body_md))
            )
        return out

    # -- renditions -------------------------------------------------------------------

    def _render_channel(
        self, story: Story, channel: Channel, code: str, text: tuple[str, str, str]
    ) -> Rendition:
        headline, standfirst, body = text
        link = next((s.url for s in story.sources if s.official), None) or (
            story.sources[0].url if story.sources else None
        )
        short = SHORT_DISCLOSURE.get(code, SHORT_DISCLOSURE["en"])
        full = guards.ai_disclosure(code if code in self.policy.disclosure else "en", self.policy)

        if channel.kind == "web":
            sources = "\n".join(f"- {s.name}: {s.url}" for s in story.sources)
            rendered = (
                f"# {headline}\n\n{standfirst}\n\n{body}\n\n## Sources\n{sources}\n\n"
                f"{full}\n\n{CORRECTION_LINE.get(code, CORRECTION_LINE['en'])}"
            )
        elif channel.kind == "messaging":
            rendered = self._messaging(headline, standfirst, body, link, short, channel.max_chars)
        elif channel.kind == "audio":
            rendered = self._audio(headline, standfirst, body, code, short)
        else:
            rendered = f"{headline}\n{standfirst}" + (f"\n{link}" if link else "")
        return Rendition(channel=channel.kind, language=code, text=rendered, link=link)

    @staticmethod
    def _messaging(
        headline: str, standfirst: str, body: str, link: str | None, short: str, limit: int | None
    ) -> str:
        limit = limit or 700
        head = _plain(headline)
        tail = "\n".join(part for part in (link, short) if part)
        sentences = _sentences(_plain(standfirst)) + _sentences(_plain(body.split("\n## ")[0]))
        budget = limit - len(head) - len(tail) - 4
        chosen: list[str] = []
        for sentence in sentences:
            if len(chosen) == 3:
                break
            used = sum(len(s) + 1 for s in chosen)
            if used + len(sentence) > budget:
                break
            chosen.append(sentence)
        message = head + ("\n\n" + " ".join(chosen) if chosen else "") + "\n\n" + tail
        return _truncate(message, limit)

    @staticmethod
    def _audio(headline: str, standfirst: str, body: str, code: str, short: str) -> str:
        text = f"{_plain(headline)}. {_plain(standfirst)}\n{_plain(body.split(chr(10) + '## Sources')[0])}"
        sentences = _short_sentences(text)
        script = " ".join(sentences)
        if code == "en":
            script = _spell_numbers(script)
        return f"{script} {short}".strip()


# ------------------------------------------------------------------- offline stand-in

FAKE_HANDLERS: dict[str, Callable[[dict[str, Any]], BaseModel]] = {}
