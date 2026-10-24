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
from newsroom.parallel import run_parallel
from newsroom.sources.rss import extract_page, fetch_feed

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
        progress = self.ctx.progress

        if fixtures is not None:
            batches = [(by_id.get(a.source_id), [a]) for a in fixtures]
            progress.begin("collect", total=len(batches), detail="Reading the saved sample news")
            for source, articles in batches:
                fresh.extend(self._ingest(articles, source, fetch_text=False))
                progress.advance("collect")
            progress.finish_stage("collect", detail=f"{len(fresh)} new articles")
            return fresh

        # Official bodies first: city decisions and permit news must not be cut by the cap.
        ordered = sorted(sources, key=lambda s: 0 if s.official else 1)
        active = sum(1 for s in ordered if s.enabled and s.feed_url)
        # Every feed gets an equal share of the cap, so the last feeds are not starved by the
        # first ones. Items that are already stored do not use up a feed's share.
        quota = max(1, -(-max_new // max(1, active))) if max_new is not None else None
        progress.begin("collect", total=active, detail="Reading news feeds")

        # Feeds are downloaded a wave at a time, official bodies first: everything in a wave at the
        # same moment, then the full text of its new articles at the same moment. Waiting on 20+
        # websites one after another was most of the time a cycle spent here. Waves keep the
        # priority order and stop early: once the cap is reached the later feeds are never fetched.
        wave = max(1, self.settings.fetch_workers)
        for first in range(0, len(ordered), wave):
            if max_new is not None and len(fresh) >= max_new:
                self.trace("fetch", "limit", f"article limit of {max_new} reached")
                break
            chunk = ordered[first : first + wave]
            fetchable = [s for s in chunk if s.enabled and s.feed_url]
            fetched = dict(
                zip(
                    (s.id for s in fetchable),
                    run_parallel(
                        lambda s: self._download_feed(s, limit_per_source),
                        fetchable,
                        self.settings.fetch_workers,
                        "feeds",
                    ),
                    strict=True,
                )
            )
            pages = self._download_pages(fetched, by_id, cap=(max_new or 100) * 2)

            for source in chunk:
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
                progress.advance("collect", detail=f"Read {source.name}")
                try:
                    articles = fetched[source.id]
                    if isinstance(articles, Exception):
                        raise articles
                    if not articles:
                        self.trace(
                            "fetch", "source_failed", "no entries returned", source_id=source.id
                        )
                        continue
                    taken = 0
                    for raw in articles:
                        if quota is not None and taken >= quota:
                            break
                        if max_new is not None and len(fresh) >= max_new:
                            break
                        created = self._ingest([raw], source, fetch_text=True, pages=pages)
                        taken += len(created)
                        fresh.extend(created)
                except Exception as exc:  # one broken source must not stop the cycle
                    self.trace("fetch", "source_failed", str(exc)[:200], source_id=source.id)
        return fresh

    # ------------------------------------------------------------------ internals

    def _download_feed(self, source: SourceConfig, limit: int) -> list[Article] | Exception:
        try:
            return fetch_feed(source, limit=limit, settings=self.settings)
        except Exception as exc:  # one broken source must not stop the others
            return exc

    def _download_pages(
        self,
        fetched: dict[str, list[Article] | Exception],
        by_id: dict[str, SourceConfig],
        *,
        cap: int,
    ) -> dict[str, tuple[str | None, str | None]]:
        """Full text and page photo for the new articles of every feed, downloaded side by side.

        Only articles that are not stored yet are fetched (one batch query tells which), and never
        those of paywalled outlets."""

        if not self.settings.fetch_article_content:
            return {}
        wanted: list[str] = []
        for source_id, articles in fetched.items():
            source = by_id.get(source_id)
            if isinstance(articles, Exception) or (source and source.access == "metadata_only"):
                continue
            wanted.extend(
                a.url
                for a in articles
                if a.url and not a.content and a.access != "metadata_only"
            )
        wanted = list(dict.fromkeys(wanted))
        known = {a.url for a in self.store.find_in(Article, "url", wanted)}
        todo = [url for url in wanted if url not in known][:cap]

        def page(url: str) -> tuple[str | None, str | None]:
            try:
                return extract_page(url, settings=self.settings)
            except Exception:  # noqa: BLE001 - a page that will not load is simply skipped
                return None, None

        texts = run_parallel(page, todo, self.settings.fetch_workers, "pages")
        return dict(zip(todo, texts, strict=True))

    def _ingest(
        self,
        articles: list[Article],
        source: SourceConfig | None,
        *,
        fetch_text: bool,
        pages: dict[str, tuple[str | None, str | None]] | None = None,
    ) -> list[Article]:
        created: list[Article] = []
        metadata_only = bool(source and source.access == "metadata_only")
        for raw in articles:
            access = "metadata_only" if metadata_only or raw.access == "metadata_only" else "full"
            content = raw.content
            page_image: str | None = None
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
                # The page's photo is used only when the feed has none. Paywalled pages are
                # never fetched (above), so their photo can only come from the feed.
                if pages is not None and raw.url in pages:
                    content, page_image = pages[raw.url]  # downloaded alongside the others
                else:
                    content, page_image = extract_page(raw.url, settings=self.settings)

            article = raw.model_copy(
                update={
                    "title": guards.sanitize(raw.title, max_chars=500),
                    "summary": guards.sanitize(raw.summary, max_chars=2000) or None,
                    "content": guards.sanitize(content) or None,
                    "access": access,
                    "image_url": raw.image_url or page_image,
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
