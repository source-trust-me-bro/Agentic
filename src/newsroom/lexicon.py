"""Small bilingual (Finnish/English) helpers shared by the editor and the reporter.

They exist because Finnish inflects at the end of words ("Levi" -> "Levillä") and because
sources and communities use different languages. Matching is by *word prefix*, never by
plain substring, so "jää" is not found inside "työntekijää".

These helpers only support ranking and retrieval. Editorial decisions (status,
independence, risk) are made elsewhere, in plain code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# concept -> terms that signal it in either language (matched at the start of a word)
CONCEPTS: dict[str, list[str]] = {
    "permit": [
        "permit",
        "oleskelulup",
        "migri",
        "maahanmuuttovirasto",
        "immigration",
        "visa",
        "viisumi",
        "residence card",
    ],
    "processing": ["processing", "käsittely", "application", "hakemus"],
    "housing": ["housing", "accommodation", "rent", "apartment", "asunto", "vuokra", "majoitus"],
    "seasonal": ["seasonal", "kausityö", "kausi"],
    "union": ["union", "collective agreement", "ammattiliit", "työehtosopimus"],
    "transport": [
        "bus",
        "transport",
        "timetable",
        "joukkoliikenne",
        "bussi",
        "aikataulu",
        "kausilip",
        "season ticket",
    ],
    "work": ["work", "job", "employ", "wage", "salary", "työ", "palkka"],
    "health": ["health", "doctor", "hospital", "terveys", "lääkäri", "sairaala"],
    "education": [
        "student",
        "university",
        "school",
        "daycare",
        "kindergarten",
        "opiskel",
        "yliopisto",
        "koulu",
        "päiväkoti",
        "varhaiskasvatus",
    ],
    "language": ["sámi", "sami", "saame", "saamen", "kieli", "language"],
    "tax": ["tax", "vero"],
    "safety": ["safety", "rescue", "avalanche", "emergency", "turvallisuus", "pelastus", "hätä"],
    "winter": ["winter tyre", "winter driving", "talvirengas", "black ice", "liukas"],
    "culture": ["culture", "festival", "kulttuuri", "festivaali"],
    "reindeer": ["reindeer", "poro"],
    "services": ["service", "palvelu"],
    "council": ["council", "valtuusto", "kaupunginhallitus", "decision", "päätös"],
    "daily": ["daily life", "everyday", "arki"],
    "funding": ["funding", "grant", "rahoitus", "avustus"],
}

# community interest (lower case) -> concept names that count as a hit
INTEREST_CONCEPTS: dict[str, list[str]] = {
    "housing": ["housing"],
    "work": ["work"],
    "healthcare": ["health"],
    "education": ["education"],
    "transport": ["transport"],
    "taxes": ["tax"],
    "residence permits": ["permit"],
    "daily life": ["daily"],
    "working conditions": ["work", "union"],
    "safety": ["safety"],
    "winter driving": ["winter"],
    "language": ["language"],
    "culture": ["culture"],
    "reindeer herding": ["reindeer"],
    "services": ["services"],
    "local decisions": ["council"],
}

# Interests that apply everywhere in Finland, so a hit needs no mention of the region.
NATIONWIDE_INTERESTS: frozenset[str] = frozenset({"residence permits", "taxes"})

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def words(text: str | None) -> list[str]:
    return [w.casefold() for w in _WORD.findall(text or "")]


def has_prefix_term(text: str, term: str) -> bool:
    """True if ``term`` occurs at the start of a word (so inflected forms match)."""

    return re.search(r"(?<!\w)" + re.escape(term.casefold()), text.casefold()) is not None


def concepts(text: str | None) -> set[str]:
    haystack = text or ""
    return {
        name
        for name, terms in CONCEPTS.items()
        if any(has_prefix_term(haystack, term) for term in terms)
    }


def numbers(text: str | None) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text or "")}


def proper_nouns(text: str | None) -> set[str]:
    """Capitalised words (>= 4 letters), lower-cased. Sentence starts add some noise."""

    return {w.casefold() for w in _WORD.findall(text or "") if w[0].isupper() and len(w) >= 4}


def nouns_overlap(a: Iterable[str], b: Iterable[str]) -> set[str]:
    """Proper nouns that match by prefix ("levi" ~ "levillä", "migri" ~ "migrin")."""

    b_list = list(b)
    hits: set[str] = set()
    for x in a:
        for y in b_list:
            short, long_ = (x, y) if len(x) <= len(y) else (y, x)
            if len(short) >= 4 and long_.startswith(short):
                hits.add(short)
    return hits


def query_terms(text: str, extra: Iterable[str] = ()) -> list[str]:
    """Search terms for finding coverage of a claim across languages.

    Numbers and proper nouns come first; stem variants cover Finnish endings; concept
    terms bridge Finnish <-> English wording.
    """

    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.casefold()
        if len(term) >= 3 and term not in seen:
            seen.add(term)
            terms.append(term)

    for n in sorted(numbers(text)):
        add(n)
    for noun in sorted(proper_nouns(text)):
        add(noun)
        if len(noun) >= 6:
            add(noun[:-2])
            add(noun[:-3])
    for word in words(text):
        if len(word) >= 5:
            add(word)
    for name in concepts(text):
        for term in CONCEPTS[name]:
            for part in term.split():
                add(part)
    for hint in extra:
        for word in words(hint):
            add(word)
    return terms
