"""Assignment desk: watches sources and brings new reports into the newsroom.

It only gathers and labels. It never decides what is true or what to cover, and it
never obeys text found inside a report: instruction-like text is flagged for humans.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from pydantic import BaseModel

from newsroom import guards
from newsroom.agents.base import Agent
from newsroom.domain import Article, SourceConfig
from newsroom.lexicon import words
from newsroom.sources.rss import extract_content, fetch_feed

FAKE_HANDLERS: dict[str, Callable[[dict], BaseModel]] = {}  # the monitor makes no model calls

_STOP = {
  "the", "and", "for", "with", "that", "this", "from", "are", "was", "will", "has", "have",
  "ja", "on", "että", "oli", "ovat", "myös", "kun", "joka", "mutta", "sekä",
}  # fmt: skip


def normalise_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def content_hash(article: Article) -> str:
    basis = normalise_text(article.content) or normalise_text(article.title)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def syndication_key(title: str) -> str | None:
    """Same title tokens -> same key (wire copies and reposts). Unrelated stories differ.

    Titles with fewer than three meaningful words get no key, so very short headlines
    cannot collide by accident.
    """

    tokens = sorted({w for w in words(title) if len(w) >= 3 and w not in _STOP})
    if len(tokens) < 3:
        return None
    return "syn_" + hashlib.sha1(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


class MonitorAgent(Agent):
    name = "assignment_desk"

    def scan(
        self,
        sources: list[SourceConfig],
        *,
        limit_per_source: int = 10,
        max_new: int | None = None,
        fixtures: list[Article] | None = None,
    ) -> list[Article]:
        """Fetch, label and store reports. Returns only the newly stored ones."""

        by_id = {source.id: source for source in sources}
        fresh: list[Article] = []

        if fixtures is not None:
            batches = [(by_id.get(a.source_id), [a]) for a in fixtures]
            for source, articles in batches:
                fresh.extend(self._ingest(articles, source, fetch_text=False))
            return fresh

        # Official bodies first: city decisions and permit news must not be cut by the cap.
        for source in sorted(sources, key=lambda s: 0 if s.official else 1):
            if max_new is not None and len(fresh) >= max_new:
                self.trace("fetch", "limit", f"article limit of {max_new} reached")
                break
            if not source.enabled or not source.feed_url:
                self.trace(
                    "skip_source",
                    "skipped",
                    "disabled or no feed_url configured",
                    source_id=source.id,
                )
                continue
            try:
                articles = fetch_feed(source, limit=limit_per_source, settings=self.settings)
                if not articles:
                    self.trace("fetch", "source_failed", "no entries returned", source_id=source.id)
                    continue
                fresh.extend(self._ingest(articles, source, fetch_text=True))
            except Exception as exc:  # one broken source must not stop the cycle
                self.trace("fetch", "source_failed", str(exc)[:200], source_id=source.id)
        return fresh

    # ------------------------------------------------------------------ internals

    def _ingest(
        self, articles: list[Article], source: SourceConfig | None, *, fetch_text: bool
    ) -> list[Article]:
        created: list[Article] = []
        metadata_only = bool(source and source.access == "metadata_only")
        for raw in articles:
            access = "metadata_only" if metadata_only or raw.access == "metadata_only" else "full"
            content = raw.content
            if access == "metadata_only":
                # Paywalled outlet: headline, summary and link only. Never fetch the page
                # and never keep body text, even if a feed or fixture happens to carry it.
                content = None
                self.trace(
                    "access",
                    "metadata_only",
                    "paywalled outlet: only headline, summary and link are used",
                    source_id=raw.source_id,
                    url=raw.url,
                )
            elif fetch_text and self.settings.fetch_article_content and not content:
                content = extract_content(raw.url, settings=self.settings)

            article = raw.model_copy(
                update={
                    "title": guards.sanitize(raw.title, max_chars=500),
                    "summary": guards.sanitize(raw.summary, max_chars=2000) or None,
                    "content": guards.sanitize(content) or None,
                    "access": access,
                }
            )
            article.injection_flags = guards.scan_injection(
                f"{article.title} {article.summary or ''} {article.content or ''}",
                self.policy,
            )
            article.content_hash = content_hash(article)
            if not article.syndication_key:
                article.syndication_key = syndication_key(article.title)

            stored, was_created = self.store.upsert_article(article)
            if not was_created:
                self.trace("ingest", "duplicate", "already stored", url=article.url)
                continue

            created.append(stored)
            self.trace(
                "ingest", "created", "new report stored", article_id=stored.id, url=stored.url
            )
            if stored.injection_flags:
                self.trace(
                    "ingest",
                    "flagged_injection",
                    "text addressed to an AI was found; kept for humans, never obeyed",
                    article_id=stored.id,
                    flags=stored.injection_flags,
                )
        return created
