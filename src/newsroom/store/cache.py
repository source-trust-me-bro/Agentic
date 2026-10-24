"""A short-lived cache around a store, for one batch of work.

Clustering reads the same events and topics for every article. Against a remote database that is
dozens of round trips per article. For the length of one batch this wrapper keeps reads in memory,
lets every write through (and drops the cached reads it affects), and writes the audit entries
together at the end. Callers get copies, so changing a returned object never changes the cache.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from newsroom.domain import Article, Doc, TraceEvent
from newsroom.store.base import D, Store


class ReadCache:
    def __init__(self, inner: Store) -> None:
        self._inner = inner
        self._get: dict[tuple[str, str], Doc | None] = {}
        self._find: dict[tuple[str, tuple[tuple[str, Any], ...]], list[Doc]] = {}
        self._traces: list[TraceEvent] = []

    # ---------------------------------------------------------------- reads (cached)

    def get(self, cls: type[D], doc_id: str) -> D | None:
        key = (cls.COLLECTION, doc_id)
        if key not in self._get:
            self._get[key] = self._inner.get(cls, doc_id)
        found = self._get[key]
        return found.model_copy(deep=True) if found is not None else None  # type: ignore[return-value]

    def find(self, cls: type[D], **equals: Any) -> list[D]:
        key = (cls.COLLECTION, tuple(sorted(equals.items())))
        if key not in self._find:
            self._find[key] = list(self._inner.find(cls, **equals))
        return [doc.model_copy(deep=True) for doc in self._find[key]]  # type: ignore[misc]

    # ---------------------------------------------------------------- writes (pass through)

    def _drop(self, collection: str) -> None:
        for key in [k for k in self._find if k[0] == collection]:
            del self._find[key]

    def put(self, doc: Doc) -> None:
        self._inner.put(doc)
        self._drop(doc.COLLECTION)
        self._get[(doc.COLLECTION, doc.id)] = doc.model_copy(deep=True)

    def delete(self, cls: type[Doc], doc_id: str) -> bool:
        removed = self._inner.delete(cls, doc_id)
        self._drop(cls.COLLECTION)
        self._get.pop((cls.COLLECTION, doc_id), None)
        return removed

    def upsert_article(self, article: Article) -> tuple[Article, bool]:
        stored, created = self._inner.upsert_article(article)
        self._drop(Article.COLLECTION)
        self._get.pop((Article.COLLECTION, stored.id), None)
        return stored, created

    def search_articles(
        self,
        query: str,
        *,
        language: str | None = None,
        exclude_ids: set[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
    ) -> list[Article]:
        return self._inner.search_articles(
            query, language=language, exclude_ids=exclude_ids, since=since, limit=limit
        )

    # ---------------------------------------------------------------- audit trail (buffered)

    def trace(self, event: TraceEvent) -> None:
        self._traces.append(event)

    def traces(self, run_id: str | None = None) -> list[TraceEvent]:
        self.flush()
        return self._inner.traces(run_id)

    def flush(self) -> None:
        """Write the buffered audit entries, in order."""

        pending, self._traces = self._traces, []
        writer = getattr(self._inner, "trace_many", None)
        if callable(writer):
            writer(pending)
        else:
            for event in pending:
                self._inner.trace(event)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
