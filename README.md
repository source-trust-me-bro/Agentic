# Agentic newsroom for communities

A multilingual news medium for communities that local media reaches poorly, such as newcomers in
Oulu, seasonal workers in Lapland and North Sámi speakers. **Agents run the newsroom** (find, select,
investigate, explain, translate, check, distribute). **Humans approve everything that is published.**
**The community helps decide what matters**, contributes tips and local knowledge, reviews in its
own language and can demand corrections.

Built for the Kaleva Media challenge. It follows the Council for Mass Media's
[Guidelines for Journalists](https://jsn.fi/en/journalists-guidelines/).

## Read these first

| Doc | What it answers |
|---|---|
| [docs/CONFLICTS.md](docs/CONFLICTS.md) | Where the original plan conflicts with the challenge, and what we changed |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Roles, workflow, the event tree, bounds, model layer |
| [docs/JOURNALISM_GUIDELINES.md](docs/JOURNALISM_GUIDELINES.md) | The guidelines as concrete controls, and the human-in-the-loop rules |
| [docs/COMMUNITY.md](docs/COMMUNITY.md) | Which communities first, how they take part, trust, small languages |
| [docs/BUSINESS_MODEL.md](docs/BUSINESS_MODEL.md) | Payers, willingness to pay, independence, costs |
| [docs/EVALUATION.md](docs/EVALUATION.md) | How we test quality, errors and safety |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | How the community is connected in the app |

## Quick start (offline, no keys)

```powershell
cd D:\Agentic
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest                       # 148 tests
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m newsroom demo                # end-to-end walkthrough on fixtures
.venv\Scripts\python.exe -m newsroom serve               # API on http://127.0.0.1:8100/docs
```

The demo runs a full cycle on synthetic fixture articles: it clusters the same news from several
outlets into one event, drafts stories, attaches official references, stops at the human gate, then
simulates named reviewers so you can see publication. **Reviewers in the demo are simulated.**
Use `--no-approve` to stop at the gate.

## Using Gemini and the live feeds

The editor presses **Publish**; the agents do everything before it and build the outputs after it.
See [docs/INTEGRATION.md](docs/INTEGRATION.md).

Copy `.env.example` to `.env`, then set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`
(`GEMINI_MODEL` defaults to `gemini-3.1-flash-lite`, `GEMINI_THINKING_LEVEL` to `low`;
`CYCLE_INTERVAL_MINUTES` makes the agents run a cycle by themselves). Run
`python -m newsroom cycle` for one pass over the live Kaleva, City of Oulu and Migri feeds.

* Gemini is hosted by Google: it needs network access and an API key and may cost money.
* A live cycle makes dozens of model calls. With `gemini-3.1-flash-lite` they take a few seconds each, so a cycle takes a minute or two. Cycles run in the background.
* Set `MONGODB_URI` to persist everything in MongoDB. Without it the store is in memory.
* Kaleva is paywalled: only its public headline, summary and link are used.

## How it connects to the app

The web app and the KalevaShorts API connect members to communities, show approved stories, take
questions, tips and correction requests, and give editors a review desk. See
[docs/INTEGRATION.md](docs/INTEGRATION.md).

## What is real and what is a stand-in

| Part | State |
|---|---|
| Agents, graph, human gate, checkpoints, audit trail, rules, API | Implemented and tested |
| Event clustering across outlets and languages, topic tree | Implemented and tested |
| Kaleva (paywalled) handled as headline and link only | Implemented. Full text needs licensed access |
| Gemini provider | Implemented and exercised against the live API for the clustering task |
| Offline model | Rule-based stand-in. **Not real translation** |
| Messaging, audio (TTS) delivery | Renditions are generated; delivery is not wired up |
| Community admins and moderators as trusted voices | Implemented: appointment, scoped review rights, names shown to readers |
| Official feeds (City of Oulu, Migri) | Enabled; each feed verified to return RSS |
| Hosting | We host it. It is not open source |

## Layout

```
config/    communities, sources, topics, policy (all editable, no code)
src/newsroom/
  domain.py        data model       graph.py     the story workflow (LangGraph)
  hitl.py          who may approve  service.py   the newsroom, one cycle at a time
  guards.py        untrusted text   agents/      one module per newsroom role
  llm/ store/ sources/ api/ topics.py policy.py registry.py demo.py
tests/     behaviour tests, including the human gate and prompt injection
docs/      the answers the challenge asks for
```
