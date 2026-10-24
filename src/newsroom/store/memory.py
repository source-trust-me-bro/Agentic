from __future__ import annotations

import re
import threading
from collections import defaultdict
from collections.abc import Iterable
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
        self._lock = threading.RLock()  # stories are processed side by side

    def put(self, doc: Doc) -> None:
        with self._lock:
            self._data[doc.COLLECTION][doc.id] = doc.model_dump(mode="python")

    def get(self, cls: type[D], doc_id: str) -> D | None:
        with self._lock:
            raw = self._data[cls.COLLECTION].get(doc_id)
        return cls.model_validate(raw) if raw else None

    def find(self, cls: type[D], **equals: Any) -> list[D]:
        with self._lock:
            rows = list(self._data[cls.COLLECTION].values())
        return [
            cls.model_validate(raw)
            for raw in rows
            if all(raw.get(key) == value for key, value in equals.items())
        ]

    def get_many(self, cls: type[D], ids: Iterable[str]) -> dict[str, D]:
        with self._lock:
            rows = {i: self._data[cls.COLLECTION].get(i) for i in dict.fromkeys(ids)}
        return {i: cls.model_validate(raw) for i, raw in rows.items() if raw}

    def find_in(self, cls: type[D], field: str, values: Iterable[Any], **equals: Any) -> list[D]:
        wanted = set(values)
        with self._lock:
            rows = list(self._data[cls.COLLECTION].values())
        return [
            cls.model_validate(raw)
            for raw in rows
            if raw.get(field) in wanted and all(raw.get(k) == v for k, v in equals.items())
        ]

    def delete(self, cls: type[Doc], doc_id: str) -> bool:
        with self._lock:
            return self._data[cls.COLLECTION].pop(doc_id, None) is not None

    def upsert_article(self, article: Article) -> tuple[Article, bool]:
        with self._lock:
            return self._upsert_article(article)

    def _upsert_article(self, article: Article) -> tuple[Article, bool]:
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
        with self._lock:
            rows = list(self._data[Article.COLLECTION].values())
        for raw in rows:
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
        with self._lock:
            self._traces.append(event)

    def traces(self, run_id: str | None = None) -> list[TraceEvent]:
        with self._lock:
            traces = list(self._traces)
        return traces if run_id is None else [t for t in traces if t.run_id == run_id]
