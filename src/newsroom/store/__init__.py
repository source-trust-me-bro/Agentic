from __future__ import annotations

from newsroom.config import Settings, get_settings
from newsroom.store.base import Store
from newsroom.store.memory import MemoryStore

__all__ = ["Store", "MemoryStore", "build_store"]


def build_store(settings: Settings | None = None) -> Store:
    settings = settings or get_settings()
    if settings.mongodb_uri:
        from newsroom.store.mongo import MongoStore

        return MongoStore.from_uri(settings.mongodb_uri, settings.mongodb_database)
    return MemoryStore()
