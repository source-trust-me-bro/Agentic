"""The topic connector reads the backend topic tree and links related topics through its API.

All tests are hermetic: the backend API is a scripted fake (or httpx MockTransport
for the client itself), the model is the registered offline handler or an
explicit script, and no live service is contacted.
"""

from __future__ import annotations

import json

import httpx
import pytest

from newsroom.agents.base import AgentContext
from newsroom.agents.connector import (
    LinkEvaluation,
    LinkJudgement,
    TopicConnectorAgent,
    normalize_pair,
    shortlist_candidates,
)
from newsroom.backend_client import BackendClient, BackendError, BackendTimeout
from newsroom.config import Settings
from newsroom.llm import FakeLLM, collect_fake_handlers
from newsroom.service import Newsroom
from newsroom.store.memory import MemoryStore

# ----------------------------------------------------------------------- fakes


class FakeBackend:
    """Scripted stand-in for BackendClient: records submissions, can fail on demand."""

    provider = "fake_backend"

    def __init__(
        self,
        topics: list[dict],
        *,
        fail_ids: set[str] | None = None,
        timeout_ids: set[str] | None = None,
    ) -> None:
        self._topics = [dict(t, branches=list(t.get("branches") or [])) for t in topics]
        self.calls: list[tuple[str, str]] = []
        self.fail_ids = fail_ids or set()
        self.timeout_ids = timeout_ids or set()

    def list_topics(self) -> list[dict]:
        # like the real backend: every relationship is stored symmetrically
        branches = {t["id"]: list(t["branches"]) for t in self._topics}
        for topic_id, linked in branches.items():
            for other in linked:
                if topic_id not in branches.get(other, []):
                    branches[other].append(topic_id)
        return [dict(t, branches=branches[t["id"]]) for t in self._topics]

    def link_topics(self, first_id: str, second_id: str) -> dict:
        self.calls.append((first_id, second_id))
        if first_id in self.fail_ids or second_id in self.fail_ids:
            raise BackendError("backend POST /api/v1/topics/x/branches -> 404: unknown", status=404)
        if first_id in self.timeout_ids or second_id in self.timeout_ids:
            raise BackendTimeout("backend POST /api/v1/topics/x/branches timed out")
        for topic in self._topics:
            if topic["id"] == first_id and second_id not in topic["branches"]:
                topic["branches"].append(second_id)
            if topic["id"] == second_id and first_id not in topic["branches"]:
                topic["branches"].append(first_id)
        return {"id": first_id, "branches": [second_id]}


def _topic(
    id: str,
    name: str,
    keywords: list[str],
    *,
    parent_id: str | None = None,
    branches: list[str] | None = None,
    description: str = "",
) -> dict:
    return {
        "id": id,
        "name": name,
        "slug": name.lower().replace(" ", "-"),
        "keywords": keywords,
        "description": description,
        "language": "en",
        "parent_id": parent_id,
        "branches": list(branches or []),
    }


def _agent(
    settings: Settings | None = None, llm: FakeLLM | None = None
) -> tuple[TopicConnectorAgent, FakeLLM]:
    llm = llm or FakeLLM(collect_fake_handlers())
    settings = settings or Settings(
        llm_provider="fake",
        backend_api_url="http://backend.test",
        connector_uses_model=True,
        connector_prefilter_score=0.0,  # judge every shortlisted pair in these tests
        connector_max_candidates=10,
    )
    agent = TopicConnectorAgent(AgentContext(llm=llm, store=MemoryStore(), settings=settings))
    return agent, llm


def _scripted_llm(related: list[tuple[str, str]]) -> FakeLLM:
    """Offline judge that accepts exactly the given pairs (keyed like the agent does)."""

    llm = FakeLLM()
    wanted = {normalize_pair(a, b) for a, b in related}

    def handler(payload):
        judgements = []
        for candidate in payload["candidates"]:
            key = normalize_pair(candidate["topic_a"], candidate["topic_b"])
            related_here = key in wanted
            judgements.append(
                LinkJudgement(
                    topic_a=candidate["topic_a"],
                    topic_b=candidate["topic_b"],
                    related=related_here,
                    reason="scripted: shared subject matter" if related_here else "unrelated",
                )
            )
        return LinkEvaluation(judgements=judgements)

    llm.register("connector_evaluate", handler)
    return llm


PERMIT_TOPICS = [
    _topic("t1", "Residence permits and visas", ["visa", "permit", "migri", "oleskelulupa"]),
    _topic("t2", "Immigration and rights", ["immigration", "permit", "asylum", "migri"]),
]


# ------------------------------------------------------ BackendClient (real HTTP)


TREE = [
    {
        "id": "t1",
        "name": "Residence permits and visas",
        "children": [{"id": "t1a", "name": "Processing times", "children": [], "branches": []}],
    },
    {"id": "t2", "name": "Transport", "children": [], "branches": []},
]


def _mock_client(handler) -> BackendClient:
    return BackendClient(
        "http://backend.test", token="secret-token", transport=httpx.MockTransport(handler)
    )


def test_client_flattens_the_tree_and_uses_the_documented_paths():
    routes: dict[tuple[str, str], dict[str, str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        routes[(request.method, request.url.path)] = dict(request.headers)
        if request.method == "GET":  # the tree endpoint, reader access
            return httpx.Response(200, json=TREE)
        return httpx.Response(200, json={"id": "t2", "branches": ["t1"]})

    client = _mock_client(handler)

    topics = client.list_topics()

    assert [t["id"] for t in topics] == ["t1", "t1a", "t2"]
    assert all("branches" in t for t in topics)
    assert routes[("GET", "/api/v1/topics/tree")]["authorization"] == "Bearer secret-token"

    client.link_topics("t2", "t1")

    assert routes[("POST", "/api/v1/topics/t2/branches")]["authorization"] == "Bearer secret-token"


def test_client_maps_backend_failures_and_timeouts():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":  # the tree endpoint, reader access
            return httpx.Response(200, json=TREE)
        branch_id = request.read() and json.loads(request.content).get("branch_id", "")
        if branch_id == "bad":
            return httpx.Response(404, json={"detail": "Topic 'bad' was not found."})
        if branch_id == "loop":
            return httpx.Response(400, json={"detail": "A topic cannot reference itself."})
        raise httpx.ConnectTimeout("backend did not answer")

    client = _mock_client(handler)

    with pytest.raises(BackendError) as missing:
        client.link_topics("t1", "bad")
    assert missing.value.status == 404

    with pytest.raises(BackendError) as bad_request:
        client.link_topics("t1", "loop")
    assert bad_request.value.status == 400

    with pytest.raises(BackendTimeout):
        client.link_topics("t1", "slow")

    with pytest.raises(BackendError):
        client.link_topics("t1", "t1")  # self-links never even reach the wire


# ------------------------------------------------------------ candidate retrieval


def test_candidate_shortlist_is_capped_and_deduplicated():
    first = _topic("a", "Residence permit processing", ["visa", "permit", "processing"])
    second = _topic("b", "Visa processing times", ["visa", "permit", "minor"])
    twin = _topic("c", "Residence permit processing", ["visa", "permit", "processing"])
    unrelated = _topic("d", "Local bus timetables", ["bus", "timetable", "route"])

    pairs = shortlist_candidates(
        [first, second, twin, unrelated], prefilter_score=0.05, max_candidates=2
    )

    assert len(pairs) == 2
    keys = [normalize_pair(p.first_id, p.second_id) for p in pairs]
    assert len(keys) == len(set(keys))  # one candidate per relationship
    assert normalize_pair("a", "b") in keys
    assert all(p.first_id != p.second_id for p in pairs)


def test_normalize_pair_rejects_self_links_and_orders_stably():
    assert normalize_pair("t1", "t1") is None
    assert normalize_pair("", "t2") is None
    assert normalize_pair("b", "a") == normalize_pair("a", "b") == ("a", "b")


# --------------------------------------------------------------------- agent runs


def test_agent_links_related_topics_and_submits_them_to_the_backend():
    agent, llm = _agent()
    related = _topic(
        "t1",
        "Residence permits and visas",
        ["visa", "permit", "migri", "oleskelulupa"],
        description="processing times and rules",
    )
    other = _topic("t2", "Immigration and rights", ["immigration", "permit", "asylum", "migri"])
    buses = _topic("t3", "Public transport", ["bus", "timetable", "ticket"])
    backend = FakeBackend([related, other, buses])

    result = agent.run(backend)  # the offline judge keeps meaningful pairs

    assert [task for task, _ in llm.calls] == ["connector_evaluate"]  # one batched call
    submitted = [normalize_pair(p.first_id, p.second_id) for p in result.submitted]
    assert submitted == [("t1", "t2")]  # the unrelated transport pair stays unlinked
    assert backend.calls == [("t1", "t2")]
    assert result.topics_considered == 3 and result.candidates == 3
    assert result.skipped_existing == 0
    linked = [(t.step, t.decision) for t in agent.store.traces() if t.agent == "topic_connector"]
    assert ("submit", "linked") in linked


def test_agent_rejects_unrelated_pairs_without_submitting():
    agent, llm = _agent(llm=_scripted_llm([]))  # judge: nothing is related
    health = _topic("t1", "Healthcare and wellbeing", ["health", "hospital", "doctor"])
    buses = _topic("t2", "Public transport", ["bus", "timetable", "ticket"])
    backend = FakeBackend([health, buses])

    result = agent.run(backend)

    assert result.accepted == [] and result.submitted == [] and result.failed == []
    assert backend.calls == []
    assert [task for task, _ in llm.calls] == ["connector_evaluate"]
    traces = [(t.step, t.decision) for t in agent.store.traces() if t.agent == "topic_connector"]
    assert ("evaluate", "model_judged") in traces


def test_predefined_model_output_never_creates_self_links_or_invented_pairs():
    t1 = _topic("t1", "Residence permits", ["visa", "permit", "migri"])
    t2 = _topic("t2", "Immigration and rights", ["immigration", "permit", "asylum"])
    t3 = _topic("t3", "Public transport", ["bus", "timetable", "ticket"])
    llm = FakeLLM()

    def misleading_handler(payload):
        del payload
        return LinkEvaluation(
            judgements=[
                LinkJudgement(topic_a="t1", topic_b="t1", related=True, reason="same id"),
                LinkJudgement(topic_a="t1", topic_b="ghost", related=True, reason="invented id"),
                LinkJudgement(topic_a="t1", topic_b="t3", related=True, reason="unchecked claim"),
            ]
        )

    llm.register("connector_evaluate", misleading_handler)
    agent, _ = _agent(llm=llm)

    result = agent.run(FakeBackend([t1, t2, t3]))

    # the code keeps to shortlisted candidate pairs: a self-link and an invented id
    # are dropped, while a valid shortlisted pair the model vouched for is kept
    assert [normalize_pair(p.first_id, p.second_id) for p in result.accepted] == [("t1", "t3")]
    assert all(p.first_id != p.second_id for p in result.accepted + result.submitted)


def test_known_relationships_are_never_resubmitted():
    t1 = _topic("t1", "Residence permits and visas", ["visa", "permit", "migri"], branches=["t2"])
    t2 = _topic(
        "t2", "Immigration and rights", ["immigration", "permit", "asylum"], branches=["t1"]
    )
    agent, _ = _agent()
    backend = FakeBackend([dict(t) for t in [t1, t2]])

    first = agent.run(backend)
    second = agent.run(backend)

    assert first.submitted == [] and first.skipped_existing == 1
    assert second.submitted == [] and second.skipped_existing == 1
    assert backend.calls == []  # repeated runs do not re-submit known relationships


def test_repeated_runs_submit_only_until_the_backend_knows_the_link():
    agent, _ = _agent()
    backend = FakeBackend([dict(t) for t in PERMIT_TOPICS])

    first = agent.run(backend)
    second = agent.run(backend)
    third = agent.run(backend)

    assert first.submitted and not second.submitted and not third.submitted
    assert backend.calls and len(set(backend.calls)) == 1  # converged after one submission


def test_backend_errors_are_recorded_and_never_break_other_pairs():
    agent, _ = _agent(llm=_scripted_llm([("t1", "t2"), ("t1", "t3")]))
    t1 = _topic("t1", "Residence permits and visas", ["visa", "permit", "migri"])
    t2 = _topic("t2", "Immigration and rights", ["immigration", "permit", "asylum"])
    t3 = _topic("t3", "Seasonal tourism and travel", ["tourism", "resort", "travel"])
    backend = FakeBackend([t1, t2, t3], fail_ids={"t2"})

    result = agent.run(backend)

    failed = [normalize_pair(p.first_id, p.second_id) for p in result.failed if p.first_id]
    assert ("t1", "t2") in failed
    assert all("404" == str(p.error)[-3:] or p.error for p in result.failed)
    # the healthy, accepted pair still reached the backend untouched
    healthy = [call for call in backend.calls if "t2" not in call]
    assert healthy == [("t1", "t3")]


def test_timeouts_and_invalid_topic_ids_are_handled_safely():
    agent, _ = _agent(llm=_scripted_llm([("t1", "t2")]))
    backend = FakeBackend([dict(t) for t in PERMIT_TOPICS], timeout_ids={"t2"})

    result = agent.run(backend)

    assert result.submitted == []
    assert backend.calls == [("t1", "t2")]
    assert "timed out" in result.failed[0].error
    assert backend._topics[0]["branches"] == []  # nothing was half-written

    backend.timeout_ids = set()  # the fault is gone; the graph converges on the next run
    retry = agent.run(backend)

    assert [normalize_pair(p.first_id, p.second_id) for p in retry.submitted] == [("t1", "t2")]
    assert backend.calls == [("t1", "t2"), ("t1", "t2")]


def test_run_with_too_few_topics_traces_and_submits_nothing():
    agent, llm = _agent()
    backend = FakeBackend([_topic("t1", "Only topic", ["one", "single"])])

    result = agent.run(backend)

    assert result.topics_considered == 1
    assert result.submitted == [] and backend.calls == []
    assert llm.calls == []  # no model call when there is nothing to compare
    traces = [(t.step, t.decision) for t in agent.store.traces() if t.agent == "topic_connector"]
    assert ("run", "not_enough_topics") in traces


def test_backend_retrieval_failure_is_reported():
    class BrokenBackend(FakeBackend):
        def list_topics(self):
            raise BackendError("connection refused")

    agent, _ = _agent()
    result = agent.run(BrokenBackend([dict(t) for t in PERMIT_TOPICS]))

    assert result.submitted == []
    assert result.failed[0].error.startswith("retrieval failed")
    retrieval = [(t.step, t.decision) for t in agent.store.traces() if t.agent == "topic_connector"]
    assert ("retrieve", "failed") in retrieval


def test_without_configuration_the_connector_reports_and_stays_idle():
    llm = FakeLLM(collect_fake_handlers())
    agent, llm = _agent(Settings(llm_provider="fake"), llm=llm)  # backend_api_url empty

    result = agent.run()

    assert result.backend_configured is False
    assert result.topics_considered == 0 and result.submitted == []
    assert llm.calls == []
    steps = [(t.step, t.decision) for t in agent.store.traces() if t.agent == "topic_connector"]
    assert ("run", "not_configured") in steps


# -------------------------------------------------------- execution integration


def test_newsroom_cycle_runs_the_connector_when_the_backend_is_configured(monkeypatch):
    """A normal cycle also refreshes topic relationships when the integration is on."""

    newsroom = Newsroom(
        settings=Settings(
            llm_provider="fake",
            backend_api_url="http://backend.test",
            connector_uses_model=True,
        ),
        store=MemoryStore(),
        llm=FakeLLM(collect_fake_handlers()),
        persistent_checkpoints=False,
    )
    backend = FakeBackend(
        [
            _topic(
                "topic_residence-permits-and-visas",
                "Residence permits and visas",
                ["visa", "permit", "migri", "oleskelulupa"],
            ),
            _topic(
                "topic_immigration-and-rights",
                "Immigration and rights",
                ["immigration", "permit", "asylum", "migri"],
            ),
        ]
    )
    monkeypatch.setattr("newsroom.agents.connector.client_from_settings", lambda *a, **k: backend)

    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert report.connector is not None
    assert report.connector["backend_configured"] is True
    assert report.connector["submitted"], report.connector
    assert backend.calls
    traces = [(t.step, t.decision) for t in newsroom.store.traces() if t.agent == "topic_connector"]
    assert ("submit", "linked") in traces


def test_newsroom_cycle_without_a_backend_stays_lightweight(newsroom):
    report = newsroom.run_cycle(["lapland-seasonal"], use_fixtures=True)

    assert report.connector is None  # the default cycle path never engages the connector
