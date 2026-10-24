"""A small, dependency-free check of which language a text is written in.

It exists to catch one specific mistake: a model asked for English that answers in the language of
its sources (Finnish, Swedish, North Sámi...). It tells apart the languages this newsroom handles,
and says "unsure" (``None``) rather than guess on short or mixed text.
"""

from __future__ import annotations

import re

LANGUAGE_NAMES = {
    "en": "English",
    "fi": "Finnish",
    "sv": "Swedish",
    "se": "North Sámi",
    "ar": "Arabic",
    "ru": "Russian",
    "uk": "Ukrainian",
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_STOPWORDS: dict[str, set[str]] = {
    "en": set(
        "the and of to in is that for with on are as was it this by from be at you your will can "
        "what how new more their has have not but they".split()
    ),
    "fi": set(
        "ja on että ei ovat se oli mutta kun tai jos joka myös kuin vain tämä sekä kanssa ole "
        "mitä miten kaikki sinun meidän tulee voi uusi sen ne tässä niin jotka".split()
    ),
    "sv": set(
        "och att det som är på för med inte till av den har ett om de vi du kan ska men från "
        "eller också vid under mer".split()
    ),
    "se": set(
        "ja lea leat ii ahte dát mii go dan dahje muhto maid gii mas dál juo ge ovdal maŋŋel "
        "buot nu eai oktan dán dat manin olbmot sáhttet vejolaš".split()
    ),
}

# Letters that point at a language (Sámi has several that no other language here uses).
_SAMI_STRONG = set("đŋŧ")
_SAMI_WEAK = set("čšž")


def _script_share(text: str, low: int, high: int) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if low <= ord(ch) <= high) / len(letters)


def detect_language(text: str | None) -> str | None:
    """The language code of ``text`` (en, fi, sv, se, ar, ru, uk), or ``None`` when unsure."""

    text = text or ""
    if _script_share(text, 0x0600, 0x06FF) > 0.4:
        return "ar"
    if _script_share(text, 0x0400, 0x04FF) > 0.4:
        return "uk" if set("іїєґІЇЄҐ") & set(text) else "ru"

    words = [w.casefold() for w in _WORD.findall(text)]
    if len(words) < 8:
        return None

    scores = {code: float(sum(1 for w in words if w in vocab)) for code, vocab in _STOPWORDS.items()}
    lowered = text.casefold()
    scores["se"] += 3 * sum(lowered.count(ch) for ch in _SAMI_STRONG)
    scores["se"] += min(5, sum(lowered.count(ch) for ch in _SAMI_WEAK))
    scores["fi"] += 0.5 * min(8, lowered.count("ä") + lowered.count("ö"))
    scores["sv"] += 0.3 * min(8, lowered.count("å"))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < 3 or top < 1.4 * second:
        return None
    return best
