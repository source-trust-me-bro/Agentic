"""Shared plumbing for agents.

Every agent receives an ``AgentContext`` and writes its decisions to the audit trail
with ``self.trace(...)``. Agents return proposals (domain objects); they never publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from newsroom.config import Settings, get_settings
from newsroom.domain import Article, TraceEvent, new_id
from newsroom.limits import CycleLimits
from newsroom.llm.base import LLM
from newsroom.policy import Policy, get_policy
from newsroom.store.base import Store


def prompt_articles(articles: list[Article], limit: int) -> list[Article]:
    """The reports one prompt may carry: official and full-text first, then newest.

    A story can accumulate reports across many cycles. Without this bound its prompts grow
    with the archive until they no longer fit the model's context window.
    """

    if limit <= 0:
        return []
    ordered = sorted(
        articles,
        key=lambda a: (
            not a.official,
            a.access != "full",
            -(a.published_at.timestamp() if a.published_at else 0.0),
        ),
    )
    return ordered[:limit]


def article_excerpt(article: Article, max_chars: int) -> str:
    """Headline, summary and (for full-text sources only) a bounded body."""

    body = "" if article.access == "metadata_only" else (article.content or "")[:max_chars]
    return f"{article.title}\n{article.summary or ''}\n{body}"


@dataclass
class AgentContext:
    llm: LLM
    store: Store
    settings: Settings = field(default_factory=get_settings)
    policy: Policy = field(default_factory=get_policy)
    run_id: str = field(default_factory=lambda: new_id("run"))
    limits: CycleLimits = field(default_factory=CycleLimits.unlimited)


class Agent:
    name = "agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.llm = ctx.llm
        self.store = ctx.store
        self.settings = ctx.settings
        self.policy = ctx.policy

    def trace(self, step: str, decision: str, reason: str = "", **refs: Any) -> None:
        self.store.trace(
            TraceEvent(
                id=new_id("trace"),
                run_id=self.ctx.run_id,
                agent=self.name,
                step=step,
                decision=decision,
                reason=reason,
                refs=refs,
            )
        )
