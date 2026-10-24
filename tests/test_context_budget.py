"""Context-window and time protection: prompts are bounded, claims are not re-assessed
for nothing, and a call cannot outlive the cycle budget."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from helpers_agents import art, make_ctx, pitch_for, store_articles
from newsroom.agents.explainer import ExplainerAgent
from newsroom.agents.investigator import InvestigatorAgent
from newsroom.config import Settings
from newsroom.domain import Claim, ClaimStatus, InvestigationResult, new_id
from newsroom.limits import CycleLimits
from newsroom.llm.base import LLMError
from newsroom.llm.budget import BudgetedLLM, estimate_tokens
from newsroom.sources.rss import clear_robots_cache, extract_content


class Out(BaseModel):
    answer: str = "ok"


class Recording:
    def __init__(self) -> None:
        self.name = "recording"
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Out:
        self.calls.append(kwargs)
        return Out()


def test_estimator_is_a_conservative_character_ratio():
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("") == 0


def test_an_oversized_prompt_is_refused_before_the_provider_is_called():
    limits = CycleLimits()
    limits.start()
    inner = Recording()
    llm = BudgetedLLM(inner, limits, max_input_tokens=10)

    with pytest.raises(LLMError, match="Input budget"):
        llm.generate(task="huge", system="s", prompt="x" * 400, schema=Out, payload={"schema": Out})

    assert inner.calls == []  # no provider call, no retries, no timeout
    usage = limits.usage["huge"]
    assert usage["calls"] == 1 and usage["input_tokens"] > 10


def test_each_call_gets_only_the_time_left_in_the_cycle():
    limits = CycleLimits(max_seconds=30)
    limits.start()
    inner = Recording()
    llm = BudgetedLLM(inner, limits, max_input_tokens=30000)

    llm.generate(task="echo", system="s", prompt="p", schema=Out, payload={"schema": Out})

    timeout = inner.calls[0]["timeout_seconds"]
    assert timeout is not None and 0 < timeout <= 30
    assert limits.usage["echo"]["seconds"] >= 0


def test_article_selection_keeps_official_and_newest_reports():
    ctx = make_ctx()
    ctx.settings = Settings(max_articles_per_prompt=2)
    old = art("old", "Old Paper", "Old report", published="2026-10-01T10:00:00+00:00")
    new = art("new", "New Paper", "New report", published="2026-10-22T10:00:00+00:00")
    official = art(
        "official",
        "City",
        "Official decision",
        official=True,
        published="2026-09-01T10:00:00+00:00",
    )
    store_articles(ctx, old, new, official)
    pitch = pitch_for("oulu-international", old, new, official)

    InvestigatorAgent(ctx)._extract(pitch, [old, new, official])

    asked = ctx.llm.calls[-1][1]["articles"]
    assert [a["id"] for a in asked] == ["official", "new"]


def test_extraction_prompt_contains_at_most_the_configured_number_of_articles():
    ctx = make_ctx()
    ctx.settings = Settings(max_articles_per_prompt=3)
    articles = [
        art(f"a{i}", f"Outlet {i}", f"Report {i}", content="A body. " * 20) for i in range(10)
    ]
    store_articles(ctx, *articles)
    pitch = pitch_for("oulu-international", *articles)

    InvestigatorAgent(ctx)._extract(pitch, articles)

    assert len(ctx.llm.calls[-1][1]["articles"]) == 3


def test_explainer_prompt_contains_at_most_the_configured_number_of_sources():
    ctx = make_ctx()
    ctx.settings = Settings(max_articles_per_prompt=3)
    articles = [
        art(f"a{i}", f"Outlet {i}", f"Report {i}", content="A body. " * 20) for i in range(10)
    ]
    pitch = pitch_for("oulu-international", *articles)

    from helpers_agents import community

    ExplainerAgent(ctx)._draft(pitch, [], community("oulu-international"), articles, [], 1)
    assert len(ctx.llm.calls[-1][1]["articles"]) == 3


def test_an_unchanged_claim_is_not_assessed_again_in_the_next_round():
    ctx = make_ctx()
    article = art("a1", "Paper", "A report", content="The permit takes 45 days.")
    store_articles(ctx, article)
    pitch = pitch_for("oulu-international", article)
    claim = Claim(
        id=new_id("claim"),
        pitch_id=pitch.id,
        text="The permit takes 45 days.",
        status=ClaimStatus.ATTRIBUTED,
        source_article_id=article.id,
    )
    ctx.store.put(claim)

    result = InvestigatorAgent(ctx).investigate(
        pitch,
        round_no=2,
        prior=InvestigationResult(claims=[], sufficient=False, search_hints=[], round_no=1),
    )

    assert ctx.llm.calls == []  # same empty hints as the previous round: nothing new to ask
    assert result.claims[0].id == claim.id


def test_assessment_calls_per_story_are_capped():
    ctx = make_ctx()
    ctx.settings = Settings(max_evidence_assessments_per_story=0)
    article = art("a1", "Paper", "A report", content="The permit takes 45 days.")
    store_articles(ctx, article)
    pitch = pitch_for("oulu-international", article)
    claim = Claim(
        id=new_id("claim"),
        pitch_id=pitch.id,
        text="The permit takes 45 days.",
        status=ClaimStatus.ATTRIBUTED,
        source_article_id=article.id,
        assessed_hints=["old"],
    )
    ctx.store.put(claim)

    InvestigatorAgent(ctx).investigate(pitch, round_no=2)

    assert ctx.llm.calls == []
    assert any(t.decision == "capped" for t in ctx.store.traces(ctx.run_id))


def test_the_cycle_records_prompt_size_and_latency(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert report.model_usage
    traces = newsroom.store.traces(report.run_id)
    usage = next(t for t in traces if t.step == "usage")
    assert usage.agent == "model_layer"
    assert usage.refs["usage"] == report.model_usage


def test_robots_txt_is_fetched_once_per_host():
    clear_robots_cache()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200, text="<html><body><article><p>" + "Word " * 400 + "</p></article></body></html>"
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    extract_content("https://example.test/a", client=http)
    extract_content("https://example.test/b", client=http)

    assert seen.count("https://example.test/robots.txt") == 1
