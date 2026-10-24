"""Loads config/policy.yaml: the editorial rules, kept as data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from newsroom.config import get_settings


class TopicRule(BaseModel):
    keywords: list[str] = Field(default_factory=list)


class ProtectedTerm(BaseModel):
    term: str
    note: str = ""


class ReviewPolicy(BaseModel):
    """Which extra humans a HIGH-risk story needs besides the editor-in-chief.

    The default is deliberately strict: permits, health, employment, legal rights and safety
    are where a mistake harms people. Set ``high_risk_roles: []`` in policy.yaml to let the
    editor-in-chief alone approve them. Language review never blocks a story: each language
    goes out when its own reviewer approves it.
    """

    high_risk_roles: list[str] = Field(
        default_factory=lambda: ["subject_expert", "community_reviewer"]
    )


class Policy(BaseModel):
    high_risk_topics: dict[str, TopicRule] = Field(default_factory=dict)
    elevated_topics: dict[str, TopicRule] = Field(default_factory=dict)
    injection_patterns: list[str] = Field(default_factory=list)
    protected_terms: list[ProtectedTerm] = Field(default_factory=list)
    disclosure: dict[str, str] = Field(default_factory=dict)
    review: ReviewPolicy = Field(default_factory=ReviewPolicy)

    def disclosure_text(self, language: str) -> str:
        text = self.disclosure.get(language) or self.disclosure.get("en", "")
        return " ".join(text.split())


def load_policy(path: str | Path | None = None) -> Policy:
    file = Path(path or get_settings().policy_file)
    return Policy.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")) or {})


@lru_cache
def get_policy() -> Policy:
    return load_policy()
