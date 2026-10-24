"""Shared plumbing for agents.

Every agent receives an ``AgentContext`` and writes its decisions to the audit trail
with ``self.trace(...)``. Agents return proposals (domain objects); they never publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from newsroom.config import Settings, get_settings
from newsroom.domain import TraceEvent, new_id
from newsroom.limits import CycleLimits
from newsroom.llm.base import LLM
from newsroom.policy import Policy, get_policy
from newsroom.store.base import Store


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
