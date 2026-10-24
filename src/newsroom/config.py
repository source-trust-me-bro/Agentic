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
    # Context-window protection. A prompt above this estimate is refused by the model layer
    # (chars/4 is a deliberately cheap approximation for mixed-language text). Agents also
    # trim their own inputs well below this: this is the backstop, not the target.
    llm_max_input_tokens: int = 30000
    # How many articles one prompt may contain. Events grow across cycles, so prompts must
    # not grow with them; newest official/full-text reports are kept first.
    max_articles_per_prompt: int = 12
    # Chars of one article offered to the claim extractor / the explainer.
    extract_article_chars: int = 4000
    explain_article_chars: int = 1500
    # Cap on evidence-checking model calls for one story, across all investigation rounds.
    max_evidence_assessments_per_story: int = 12
    # Article bodies are fetched in parallel (robots.txt is cached per host).
    max_fetch_workers: int = 4
    # Topic matching is bounded too: an event may have hundreds of reports over time.
    topic_match_chars: int = 20000

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

    # Topic-graph integration with the application backend's REST API (see the Backend
    # README, "Topic relationships"). When the URL is empty the connector only reports
    # what it would link, in the agent trace. The token is a bearer credential the
    # backend accepts with the editor role; the backend owns integrity, so a failed or
    # unauthorized call can never corrupt the graph.
    backend_api_url: str = ""  # e.g. http://127.0.0.1:8000 (/api/v1 is appended by the client)
    backend_api_token: str = ""
    # The connector narrows candidates with deterministic lexical scores (no model) and
    # spends at most one model call per run judging the shortlist.
    connector_uses_model: bool = False
    connector_prefilter_score: float = 0.12  # candidate shortlist threshold
    connector_min_similarity: float = 0.34  # accepted without a model judgement
    connector_max_candidates: int = 20  # pairs judged per run (scalability, no O(n²) model calls)


@lru_cache
def get_settings() -> Settings:
    return Settings()
