from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model layer. "fake" runs fully offline with deterministic rules (tests, demos).
    llm_provider: Literal["fake", "gemini", "huggingface"] = "fake"
    gemini_api_key: str = ""
    # Any OpenAI-compatible endpoint. The default is the Hugging Face router; the model runs on
    # a third-party host (Novita), so community text is masked and never sent with contact details.
    hf_token: str = ""
    hf_model: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017:novita"
    hf_base_url: str = "https://router.huggingface.co/v1"
    # Measured on a real claim-extraction task, counting only valid output: gemini-3.1-flash-lite 3.3 s,
    # gemini-3-flash-preview 4.2 s, gemini-3.8-flash 38 s. gemini-2.5-flash is not available to new keys.
    # A cycle makes dozens of calls, so speed decides what is usable.
    gemini_model: str = "gemini-3.1-flash-lite"
    # "minimal" is not supported by this model. "low" keeps calls to ~30 s instead of
    # ~80 s; raise it (medium/high) only where reasoning quality matters more than time.
    gemini_thinking_level: Literal["low", "medium", "high"] = "low"
    llm_timeout_seconds: float = 120.0

    # Storage. Without MONGODB_URI the newsroom keeps everything in memory.
    mongodb_uri: str = ""
    mongodb_database: str = "agentic_newsroom"

    # LangGraph checkpoints make human-review pauses survive restarts.
    checkpoint_db: str = str(ROOT / "data" / "checkpoints.sqlite")

    # Bounds that stop investigation loops (explicit stopping conditions).
    max_investigation_rounds: int = 3
    max_revision_rounds: int = 2
    # Changes a human asks for are always honoured, up to this many rounds per story.
    max_human_revision_rounds: int = 5
    max_pitches_per_cycle: int = 2  # per community
    # Speed limits for one cycle. This is a demo-sized newsroom: small, bounded and stoppable.
    # Clustering is decided by code. Asking the model about every article costs seconds each and
    # adds little; the model is spent on the stories instead. Turn on to let it suggest entities.
    architect_uses_model: bool = False
    max_items_per_source: int = 2  # newest items read from each feed
    max_articles_per_cycle: int = 24  # articles clustered and considered per cycle
    cycle_max_seconds: int = 420  # hard time budget for one cycle
    cycle_max_model_calls: int = 120  # hard model-call budget for one cycle
    # Agents run a cycle on their own every N minutes (0 = off). Only with a real model:
    # the offline stand-in never runs on live feeds.
    cycle_interval_minutes: int = 0
    http_timeout_seconds: float = 20.0
    fetch_article_content: bool = True
    user_agent: str = "AgenticNewsroomBot/0.1 (+contact the editors before crawling)"

    # When set, review and publishing endpoints require this value in X-Api-Key. Real
    # deployments must put proper staff authentication in front of these routes.
    api_key: str = ""

    # Allowed browser origins for the community web app.
    cors_origins: str = "http://localhost:3000"

    sources_file: str = str(ROOT / "config" / "sources.yaml")
    communities_file: str = str(ROOT / "config" / "communities.yaml")
    official_sources_file: str = str(ROOT / "config" / "official_sources.yaml")
    policy_file: str = str(ROOT / "config" / "policy.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()
