# The Connector Agent: semantic relationships between topics

The topic tree answers "what subject does this belong to". Readers also need the sideways links:
"residence permits" is connected to "immigration and rights", "seasonal work" to "housing". The
**connector agent** finds those relationships in the existing topics and files them with the
application backend, which owns the topic graph.

## Purpose and responsibilities

`src/newsroom/agents/connector.py` defines `TopicConnectorAgent`. One run:

1. **Retrieves the topics** from the backend's REST endpoint `GET /api/v1/topics/tree`
   (reader access). The response is the whole tree; every node carries `branches`, the topic
   ids it is already related to, so the agent also learns what is already linked.
2. **Shortlists candidate pairs** with deterministic code — token/keyword overlap between
   topic names, descriptions and keywords. Pairs are deduplicated (A–B and B–A are the same
   relationship), self-links are impossible by construction, and the list is capped
   (`CONNECTOR_MAX_CANDIDATES`) so one run stays bounded as the topic count grows. No model
   calls happen here.
3. **Judges the shortlist** with one batched model call
   (task `connector_evaluate`, schema `LinkEvaluation`) that must answer for each pair
   whether there is a meaningful relationship: shared events, shared named subject matter,
   or a contextual connection a reader would follow. Sharing generic words or only a broad
   parent category is explicitly not enough. When the judge is important but unavailable
   (`LLMError`) or disabled (`CONNECTOR_USES_MODEL=false`), the code falls back to a
   conservative lexical threshold (`CONNECTOR_MIN_SIMILARITY`).
4. **Submits the accepted pairs** one by one with
   `POST /api/v1/topics/{topic_id}/branches` and body `{"branch_id": "<other topic id>"}`.
   The backend validates them, stores the relationship symmetrically on both topics, and is
   idempotent. Pairs the tree already reports as related are never resubmitted.
5. **Records everything** in the audit trail (`agent=topic_connector`: retrieve, candidates,
   evaluate, skip_existing, submit, run) and never raises on a single backend failure: the
   failing pair is recorded in `failed`, the rest of the run continues.

Every stage is a separate method (`_retrieve`, `shortlist_candidates`, `_evaluate`,
`_submit`), so retrieval, evaluation and persistence are tested independently. Model output
is validated like everywhere else in the newsroom: it only *proposes*, and code drops
verdicts that reference self-links, invented topic ids, or pairs the candidate stage never
produced.

## Backend configuration and what it must accept

| Setting (`Settings` / `.env`) | Meaning |
|---|---|
| `BACKEND_API_URL` | Base URL of the KalevaShorts API, e.g. `http://127.0.0.1:8000`. The client appends `/api/v1` itself. **Empty means the connector stays passive**: it runs nothing, traces `not_configured`, and normal cycles behave exactly as before. |
| `BACKEND_API_TOKEN` | Bearer credential sent as `Authorization` on every call. The backend requires the **editor** role for linking topics. |
| `CONNECTOR_USES_MODEL` | `false` (default): threshold-only acceptance. `true`: one batched model judgement of the shortlist. |
| `CONNECTOR_PREFILTER_SCORE` | Minimum lexical score for a pair to become a candidate (default `0.12`). |
| `CONNECTOR_MIN_SIMILARITY` | Threshold used when judging without a model (default `0.34`). |
| `CONNECTOR_MAX_CANDIDATES` | Pairs judged per run (default `20`). |

**Integration dependency to be provisioned.** The backend currently authenticates by Clerk
RS256 session tokens, so a runbook-driven machine cannot link topics without a credential the
backend accepts for an editor. Until the backend provides a service-account scheme, set
`BACKEND_API_TOKEN` to whichever credential your deployment has chosen and verify it works
against a staging backend. A wrong or expired token surfaces as `submit failed` entries in the
trace and in `ConnectorResult.failed` — the graph and the newsroom are unaffected, and the next
run converges because known relationships are recognised from the tree rather than retried blind.

## Running it

The connector is wired into the ordinary execution mechanism — it runs as part of a cycle
(same run id, same model budget, same stopping rules) whenever `BACKEND_API_URL` is set:

```powershell
# inside the running newsroom API: the next cycle links topics too
POST http://127.0.0.1:8100/cycle   (X-Api-Key: your staff key)
```

```python
from newsroom.service import Newsroom

nr = Newsroom()
nr.run_connector()          # on demand, without a full cycle (None when unconfigured)
result = nr.connector.run() # also accepts an injected backend client (tests, runbooks)
```

`result` is a `ConnectorResult`: `topics_considered`, `candidates`, `accepted`,
`skipped_existing`, `submitted`, `failed`, `backend_configured`.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/test_connector.py -q
```

The tests are hermetic: the backend API is a scripted fake (and `httpx.MockTransport`
exercises the real client paths), the model is the registered offline handler or an
explicit script. They cover tree retrieval and error mapping, candidate deduplication and
capping, acceptance of meaningful pairs, rejection of unrelated pairs, self-link and
invented-id prevention, idempotent repeated runs, backend failures/timeouts and too-few-topic
runs, plus the cycle integration (`backend configured` and the default lightweight path).

## What is deliberately not here

No direct database writes, no ORM, no migrations, no scheduler of its own: the backend owns
graph integrity, and persistence stays behind its documented REST interface.
