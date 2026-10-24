"""Curated registry of official bodies, matched to stories by topic."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from newsroom.config import get_settings


class OfficialSource(BaseModel):
    id: str
    name: str
    url: str
    topics: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    note: str = ""


def load_official_sources(path: str | Path | None = None) -> list[OfficialSource]:
    file = Path(path or get_settings().official_sources_file)
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return [OfficialSource.model_validate(item) for item in raw.get("official_sources", [])]


def match_official_sources(
    topics: list[str],
    regions: list[str] | None = None,
    *,
    registry: list[OfficialSource] | None = None,
    limit: int = 3,
) -> list[OfficialSource]:
    """Most specific first: a body that covers fewer topics but matches is more precise than
    a general portal. Region-bound bodies only match communities in that region."""

    registry = registry if registry is not None else load_official_sources()
    wanted = set(topics)
    region_set = {r.casefold() for r in (regions or [])}
    scored: list[tuple[int, int, OfficialSource]] = []
    for source in registry:
        overlap = len(wanted & set(source.topics))
        if not overlap:
            continue
        if source.regions and not region_set & {r.casefold() for r in source.regions}:
            continue
        scored.append((-overlap, len(source.topics), source))
    scored.sort(key=lambda item: (item[0], item[1], item[2].id))
    return [source for _, _, source in scored[:limit]]
