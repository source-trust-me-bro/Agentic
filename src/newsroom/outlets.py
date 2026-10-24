"""Outlet identity.

``Article.source_name`` is the *outlet* name, but feeds come in many flavours: Kaleva has
several regional feeds, Yle has national, regional, Swedish, Sámi, Ukrainian and Russian
services. For independence, one broadcaster or publisher is ONE outlet: five Yle feeds
repeating a story are repetition, not corroboration.
"""

from __future__ import annotations

import re

# canonical outlet -> pattern that identifies it anywhere in a source name
_ALIASES: list[tuple[str, re.Pattern[str]]] = [
    ("Yle", re.compile(r"(?<!\w)yle(?!\w)", re.IGNORECASE)),
    ("Kaleva", re.compile(r"(?<!\w)kaleva(?!\w)", re.IGNORECASE)),
    ("BBC", re.compile(r"(?<!\w)bbc(?!\w)", re.IGNORECASE)),
    ("Migri", re.compile(r"(?<!\w)migri(?!\w)|immigration service", re.IGNORECASE)),
    ("Kela", re.compile(r"(?<!\w)kela(?!\w)|social insurance institution", re.IGNORECASE)),
    # The city's own channels (news, notices, its magazine, its business agency) are all
    # the city speaking: one publisher, not several independent confirmations.
    (
        "City of Oulu",
        re.compile(
            r"city of oulu|oulun kaupunki|mun oulu|munoulu|businessoulu|business oulu",
            re.IGNORECASE,
        ),
    ),
    ("OSL", re.compile(r"(?<!\w)osl(?!\w)|oulu region public transport", re.IGNORECASE)),
]

_FIXTURE_SUFFIX = re.compile(r"\s*\((?:fixture|test)\)\s*$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def outlet_of(source_name: str | None) -> str:
    """Canonical outlet for a source/feed name ("Yle Pohjois-Pohjanmaa" -> "Yle")."""

    name = _SPACE.sub(" ", _FIXTURE_SUFFIX.sub("", source_name or "")).strip()
    for outlet, pattern in _ALIASES:
        if pattern.search(name):
            return outlet
    # Unknown outlet: drop a trailing feed/section after a separator.
    base = re.split(r"\s[·|–—]\s", name, maxsplit=1)[0].strip()
    return base or "unknown"


def same_outlet(a: str | None, b: str | None) -> bool:
    return outlet_of(a).casefold() == outlet_of(b).casefold()
