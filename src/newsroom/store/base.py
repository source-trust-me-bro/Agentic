"""Storage abstraction. Durable knowledge and history live here, never in the
conversation: LangGraph state carries ids and small values, the store holds the facts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, TypeVar

from newsroom.domain import Article, Doc, TraceEvent

D = TypeVar("D", bound=Doc)


class Store(Protocol):
    def put(self, doc: Doc) -> None:
        """Insert or replace a document by id."""

    def get(self, cls: type[D], doc_id: str) -> D | None: ...

    def find(self, cls: type[D], **equals: Any) -> list[D]:
        """All documents whose top-level fields equal the given values."""

    def delete(self, cls: type[Doc], doc_id: str) -> bool: ...

    # Articles get dedicated methods because ingestion and corroboration need them.
    def upsert_article(self, article: Article) -> tuple[Article, bool]:
        """Deduplicate by url / content hash. Returns (stored article, created)."""

    def search_articles(
        self,
        query: str,
        *,
        language: str | None = None,
        exclude_ids: set[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
    ) -> list[Article]:
        """Keyword search used to look for independent coverage of a claim."""

    def trace(self, event: TraceEvent) -> None: ...

    def traces(self, run_id: str | None = None) -> list[TraceEvent]: ...
