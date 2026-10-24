"""Translator: machine drafts with guard rails, never a publication decision.

* Tier C / ``human_only`` languages (North Sámi...) are NEVER machine-translated for
  publication. The agent only prepares a glossary for the human translator.
* Everything else is a *draft*. Numbers, dates, money and protected terms must survive; a
  back-translation check (HIGH risk or tier B languages) lists differences for a human.
* Only the human review layer can mark a translation ``human_reviewed``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from newsroom import guards
from newsroom.agents.base import Agent, AgentContext
from newsroom.domain import Community, LanguageLane, RiskTier, Story, Translation, new_id
from newsroom.llm.base import LLMError

_OFFLINE_PREFIX = re.compile(r"^\[offline stand-in · [^\]]*\]\s*")
_NUMBER = re.compile(r"\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|\d[\d.,:/\-]*\d|\d")


class TranslationDraft(BaseModel):
    headline: str
    standfirst: str
    body_md: str


def normalize_digits(text: str) -> str:
    """Map digits of any script (e.g. Arabic-Indic) to ASCII so numbers can be compared."""

    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() else ch for ch in text)


def extract_numbers(text: str) -> set[str]:
    """Numbers, dates, times and money amounts as digit-only strings (separators removed)."""

    found: set[str] = set()
    for match in _NUMBER.findall(normalize_digits(text)):
        digits = re.sub(r"\D", "", match)
        if digits:
            found.add(digits)
    return found


def compare_texts(original: str, other: str, protected: list[str], label: str) -> list[str]:
    """Differences a human must check between ``original`` and ``other``."""

    flags: list[str] = []
    before, after = extract_numbers(original), extract_numbers(other)
    for number in sorted(before - after):
        flags.append(f"{label}: number {number} from the source is missing or changed")
    for number in sorted(after - before):
        flags.append(f"{label}: number {number} was added and is not in the source")
    lowered = other.casefold()
    for term in protected:
        if term.casefold() not in lowered:
            flags.append(f"{label}: protected term '{term}' is missing")
    return flags


class TranslatorAgent(Agent):
    name = "translator"

    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(ctx)

    def translate(self, story: Story, lane: LanguageLane, community: Community) -> Translation:
        source_text = f"{story.headline}\n{story.standfirst}\n{story.body_md}"
        protected = guards.protected_terms_in(source_text, self.policy)

        if lane.code == story.language:
            translation = Translation(
                id=new_id("tr"),
                story_id=story.id,
                story_version=story.version,
                language=lane.code,
                headline=story.headline,
                standfirst=story.standfirst,
                body_md=story.body_md,
                machine_translated=False,
            )
            self.store.put(translation)
            self.trace("translate", "source language, no translation needed", lane.code)
            return translation

        if lane.human_review == "human_only":
            translation = Translation(
                id=new_id("tr"),
                story_id=story.id,
                story_version=story.version,
                language=lane.code,
                status="needs_human_translation",
                protected_terms=protected,
                machine_translated=False,
            )
            self.store.put(translation)
            self.trace(
                "translate", "handed to a human translator",
                f"{lane.label} ({lane.code}) is never machine-translated for publication",
                story_id=story.id,
            )  # fmt: skip
            return translation

        try:
            draft = self._translate(story, lane, protected)
        except LLMError as exc:
            self.trace("translate", "model unavailable", str(exc)[:300], language=lane.code)
            translation = Translation(
                id=new_id("tr"),
                story_id=story.id,
                story_version=story.version,
                language=lane.code,
                status="needs_human_translation",
                protected_terms=protected,
                machine_translated=False,
                backtranslation_flags=["Machine translation failed; a human must translate."],
            )
            self.store.put(translation)
            return translation

        translated_text = f"{draft.headline}\n{draft.standfirst}\n{draft.body_md}"
        flags = compare_texts(source_text, translated_text, protected, "translation")

        high_risk = story.risk_tier is RiskTier.HIGH
        # A second model call per language: only where a mistake could hurt someone.
        if high_risk or (lane.tier == "B" and story.risk_tier is not RiskTier.LOW):
            flags += self._backtranslation_flags(source_text, draft, lane, protected)

        translation = Translation(
            id=new_id("tr"),
            story_id=story.id,
            story_version=story.version,
            language=lane.code,
            headline=draft.headline,
            standfirst=draft.standfirst,
            body_md=draft.body_md,
            status="machine_draft",
            protected_terms=protected,
            backtranslation_flags=flags,
            machine_translated=True,
        )
        self.store.put(translation)
        self.trace(
            "translate", "machine draft ready", f"{len(flags)} flag(s) for the language reviewer",
            story_id=story.id, language=lane.code, risk=story.risk_tier.value,
        )  # fmt: skip
        return translation

    # ------------------------------------------------------------------------------

    def _system(self, extra: str) -> str:
        return f"{extra}\n{guards.UNTRUSTED_PREAMBLE}"

    def _translate(
        self, story: Story, lane: LanguageLane, protected: list[str]
    ) -> TranslationDraft:
        system = self._system(
            "You are a careful news translator. Translate faithfully into the target language. "
            "Keep every number, date and amount exactly. Keep protected terms recognisable: "
            "write the Finnish term in brackets after your translation. Do not add or remove facts. "
            "Keep the Markdown structure: every '## ' heading stays a heading, translated."
        )
        prompt = (
            f"Target language: {lane.label} ({lane.code}). Protected terms: {protected}.\n"
            + guards.wrap_untrusted(
                f"{story.headline}\n\n{story.standfirst}\n\n{story.body_md}", "story"
            )
        )
        return self.llm.generate(
            task="translator.translate_story",
            system=system,
            prompt=prompt,
            schema=TranslationDraft,
            payload={
                "source_language": story.language,
                "language": lane.code,
                "label": lane.label,
                "headline": story.headline,
                "standfirst": story.standfirst,
                "body_md": story.body_md,
                "protected_terms": protected,
            },
        )

    def _backtranslation_flags(
        self, source_text: str, draft: TranslationDraft, lane: LanguageLane, protected: list[str]
    ) -> list[str]:
        try:
            back = self.llm.generate(
                task="translator.backtranslate",
                system=self._system(
                    "Translate the text back into English as literally as possible, without "
                    "improving or correcting it."
                ),
                prompt=guards.wrap_untrusted(
                    f"{draft.headline}\n\n{draft.standfirst}\n\n{draft.body_md}", "translation"
                ),
                schema=TranslationDraft,
                payload={
                    "language": lane.code,
                    "headline": draft.headline,
                    "standfirst": draft.standfirst,
                    "body_md": draft.body_md,
                },
            )
        except LLMError as exc:
            self.trace("backtranslate", "check could not run", str(exc)[:300])
            return ["Back-translation check could not run; a human must verify this translation."]
        back_text = f"{back.headline}\n{back.standfirst}\n{back.body_md}"
        flags = compare_texts(source_text, back_text, protected, "back-translation")
        self.trace("backtranslate", "compared", f"{len(flags)} difference(s)", language=lane.code)
        return flags


# ------------------------------------------------------------------- offline stand-in


def _fake_translate(payload: dict[str, Any]) -> TranslationDraft:
    prefix = f"[offline stand-in · {payload['source_language']}→{payload['language']}] "
    return TranslationDraft(
        headline=prefix + payload["headline"],
        standfirst=prefix + payload["standfirst"],
        body_md=prefix + payload["body_md"],
    )


def _fake_backtranslate(payload: dict[str, Any]) -> TranslationDraft:
    return TranslationDraft(
        headline=_OFFLINE_PREFIX.sub("", payload["headline"]),
        standfirst=_OFFLINE_PREFIX.sub("", payload["standfirst"]),
        body_md=_OFFLINE_PREFIX.sub("", payload["body_md"]),
    )


FAKE_HANDLERS: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
    "translator.translate_story": _fake_translate,
    "translator.backtranslate": _fake_backtranslate,
}
