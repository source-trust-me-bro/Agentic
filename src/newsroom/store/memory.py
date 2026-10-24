from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from newsroom.domain import Article, Doc, TraceEvent
from newsroom.store.base import D

_TOKEN = re.compile(r"[^\W_]{3,}", re.UNICODE)


def tokenize(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN.findall(text)}


class MemoryStore:
    """In-memory store: tests, demos and the default when no MongoDB is configured."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._traces: list[TraceEvent] = []

    def put(self, doc: Doc) -> None:
        self._data[doc.COLLECTION][doc.id] = doc.model_dump(mode="python")

    def get(self, cls: type[D], doc_id: str) -> D | None:
        raw = self._data[cls.COLLECTION].get(doc_id)
        return cls.model_validate(raw) if raw else None

    def find(self, cls: type[D], **equals: Any) -> list[D]:
        out = []
        for raw in self._data[cls.COLLECTION].values():
            if all(raw.get(key) == value for key, value in equals.items()):
                out.append(cls.model_validate(raw))
        return out

    def delete(self, cls: type[Doc], doc_id: str) -> bool:
        return self._data[cls.COLLECTION].pop(doc_id, None) is not None

    def upsert_article(self, article: Article) -> tuple[Article, bool]:
        for raw in self._data[Article.COLLECTION].values():
            same_url = raw["url"] == article.url
            same_hash = (
                bool(article.content_hash) and raw.get("content_hash") == article.content_hash
            )
            if same_url or same_hash:
                return Article.model_validate(raw), False
        self.put(article)
        return article, True

    def search_articles(
        self,
        query: str,
        *,
        language: str | None = None,
        exclude_ids: set[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
    ) -> list[Article]:
        wanted = tokenize(query)
        if not wanted:
            return []
        scored: list[tuple[int, Article]] = []
        for raw in self._data[Article.COLLECTION].values():
            article = Article.model_validate(raw)
            if exclude_ids and article.id in exclude_ids:
                continue
            if language and article.language != language:
                continue
            if since and article.published_at and article.published_at < since:
                continue
            haystack = tokenize(f"{article.title} {article.summary or ''} {article.content or ''}")
            overlap = len(wanted & haystack)
            if overlap:
                scored.append((overlap, article))
        scored.sort(key=lambda pair: (-pair[0], pair[1].published_at is None))
        return [article for _, article in scored[:limit]]

    def trace(self, event: TraceEvent) -> None:
        self._traces.append(event)

    def traces(self, run_id: str | None = None) -> list[TraceEvent]:
        if run_id is None:
            return list(self._traces)
        return [t for t in self._traces if t.run_id == run_id]
