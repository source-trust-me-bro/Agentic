"""Loads communities and sources from config files and fixtures from disk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from newsroom.config import ROOT, get_settings
from newsroom.domain import Article, Community, SourceConfig, new_id

FIXTURE_ARTICLES = ROOT / "fixtures" / "articles.json"


def load_communities(path: str | Path | None = None) -> list[Community]:
    file = Path(path or get_settings().communities_file)
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return [Community.model_validate(item) for item in raw.get("communities", [])]


def load_sources(path: str | Path | None = None) -> list[SourceConfig]:
    file = Path(path or get_settings().sources_file)
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return [SourceConfig.model_validate(item) for item in raw.get("sources", [])]


def load_fixture_articles(
    path: str | Path | None = None, *, paywalled: bool = True
) -> list[Article]:
    """Synthetic articles for offline demos and tests. They are labelled as fixtures.

    By default Kaleva items are paywalled (headline and summary only), as the real feeds
    are. ``paywalled=False`` restores the full-text copy kept in ``content_full`` so tests
    can exercise logic that needs article text."""

    file = Path(path or FIXTURE_ARTICLES)
    items = json.loads(file.read_text(encoding="utf-8"))
    articles = []
    for item in items:
        published = item.get("published_at")
        articles.append(
            Article(
                id=item.get("id") or new_id("art"),
                source_id=item["source_id"],
                source_name=item["source_name"],
                url=item["url"],
                title=item["title"],
                summary=item.get("summary"),
                content=(
                    item.get("content_full")
                    if not paywalled and item.get("content_full")
                    else item.get("content")
                ),
                language=item.get("language", "fi"),
                published_at=datetime.fromisoformat(published) if published else None,
                official=item.get("official", False),
                access="full"
                if (not paywalled and item.get("content_full"))
                else item.get("access", "full"),
                syndication_key=item.get("syndication_key"),
            )
        )
    return articles
