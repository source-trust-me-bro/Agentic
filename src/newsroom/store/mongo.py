from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

from newsroom.domain import Article, Doc, TraceEvent
from newsroom.store.base import D
from newsroom.store.memory import tokenize

_COLLECTIONS = [
    "articles",
    "communities",
    "pitches",
    "claims",
    "stories",
    "translations",
    "standards_reports",
    "reviews",
    "publications",
    "corrections",
    "community_messages",
    "topics",
    "events",
    "tree_proposals",
    "agent_runs",
]


class MongoStore:
    """MongoDB persistence. Documents use the domain ``id`` as ``_id``."""

    def __init__(self, database: Database) -> None:
        self.db = database
        self.ensure_indexes()

    @classmethod
    def from_uri(cls, uri: str, database: str) -> MongoStore:
        return cls(MongoClient(uri, tz_aware=True)[database])

    def ensure_indexes(self) -> None:
        self.db["articles"].create_index([("url", ASCENDING)], unique=True)
        self.db["articles"].create_index([("content_hash", ASCENDING)])
        self.db["articles"].create_index([("language", ASCENDING), ("published_at", -1)])
        self.db["claims"].create_index([("pitch_id", ASCENDING)])
        self.db["topics"].create_index([("slug", ASCENDING)], unique=True)
        self.db["events"].create_index([("status", ASCENDING), ("last_updated", -1)])
        self.db["stories"].create_index([("community_id", ASCENDING), ("status", ASCENDING)])
        self.db["reviews"].create_index([("story_id", ASCENDING), ("story_version", ASCENDING)])
        self.db["agent_runs"].create_index([("run_id", ASCENDING), ("created_at", ASCENDING)])

    def put(self, doc: Doc) -> None:
        data = doc.model_dump(mode="python")
        data["_id"] = data.pop("id")
        self.db[doc.COLLECTION].replace_one({"_id": data["_id"]}, data, upsert=True)

    @staticmethod
    def _load(cls: type[D], raw: dict[str, Any] | None) -> D | None:
        if raw is None:
            return None
        raw = dict(raw)
        raw["id"] = raw.pop("_id")
        return cls.model_validate(raw)

    def get(self, cls: type[D], doc_id: str) -> D | None:
        return self._load(cls, self.db[cls.COLLECTION].find_one({"_id": doc_id}))

    def find(self, cls: type[D], **equals: Any) -> list[D]:
        query = {("_id" if k == "id" else k): v for k, v in equals.items()}
        return [self._load(cls, raw) for raw in self.db[cls.COLLECTION].find(query)]  # type: ignore[misc]

    def delete(self, cls: type[Doc], doc_id: str) -> bool:
        return self.db[cls.COLLECTION].delete_one({"_id": doc_id}).deleted_count > 0

    def upsert_article(self, article: Article) -> tuple[Article, bool]:
        clauses: list[dict[str, Any]] = [{"url": article.url}]
        if article.content_hash:
            clauses.append({"content_hash": article.content_hash})
        existing = self.db["articles"].find_one({"$or": clauses})
        if existing:
            return self._load(Article, existing), False  # type: ignore[return-value]
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
        tokens = sorted(tokenize(query))
        if not tokens:
            return []
        pattern = re.compile("|".join(re.escape(t) for t in tokens), re.IGNORECASE)
        filters: dict[str, Any] = {
            "$or": [{"title": pattern}, {"summary": pattern}, {"content": pattern}]
        }
        if language:
            filters["language"] = language
        if exclude_ids:
            filters["_id"] = {"$nin": list(exclude_ids)}
        if since:
            filters["published_at"] = {"$gte": since}
        cursor = self.db["articles"].find(filters).sort("published_at", -1).limit(limit * 3)
        articles = [self._load(Article, raw) for raw in cursor]
        wanted = set(tokens)

        def overlap(article: Article) -> int:
            hay = tokenize(f"{article.title} {article.summary or ''} {article.content or ''}")
            return len(wanted & hay)

        ranked = sorted((a for a in articles if a), key=overlap, reverse=True)
        return ranked[:limit]

    def trace(self, event: TraceEvent) -> None:
        self.put(event)

    def trace_many(self, events: list[TraceEvent]) -> None:
        if not events:
            return
        rows = []
        for event in events:
            data = event.model_dump(mode="python")
            data["_id"] = data.pop("id")
            rows.append(data)
        self.db["agent_runs"].insert_many(rows, ordered=False)

    def traces(self, run_id: str | None = None) -> list[TraceEvent]:
        query = {"run_id": run_id} if run_id else {}
        cursor = self.db["agent_runs"].find(query).sort("created_at", ASCENDING)
        return [self._load(TraceEvent, raw) for raw in cursor]  # type: ignore[misc]
