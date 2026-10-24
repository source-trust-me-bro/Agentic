# Architecture

## Principle

A real newsroom has roles, a hierarchy and a workflow. The agents copy that structure, and a
human holds the final say. LangGraph runs the workflow and the pauses. MongoDB (or an in-memory
store for tests) holds the facts and the audit trail. The model layer is replaceable.

## Roles

| Newsroom role | Agent | Does | Human counterpart |
|---|---|---|---|
| Assignment desk | `MonitorAgent` | Watches Kaleva, Yle, BBC and official feeds; deduplicates; flags text that tries to instruct an AI. | Editors maintain the source list. |
| Archive / researcher | `KnowledgeArchitectAgent` | Clusters reports of the **same news** (across outlets and languages) into one `Event` and files it in the topic tree. | Editors approve merges and new top-level branches. |
| News editor | `NewsEditorAgent` | Chooses what to pitch to each community using the Kaleva user-needs model; keeps variety. | Editor-in-chief. |
| Reporter | `InvestigatorAgent` | Extracts claims, looks for independent coverage, records evidence and uncertainty. | Subject experts for high-risk topics. |
| Writer | `ExplainerAgent` | Writes an explanatory story for the community, with sourced context. | Community reviewer. |
| Translator | `TranslatorAgent` | Drafts translations; refuses to machine-translate tier-C languages. | Language reviewer (a trusted voice). |
| Copy chief | `StandardsEditorAgent` | Checks the Journalist's Guidelines, classifies risk, decides which humans must review. | Editor-in-chief. |
| Publisher | `DistributorAgent` | Renders per channel; **refuses** to publish without approvals. | — |
| Community desk | `CommunityDeskAgent` | Moderates and triages questions, tips, corrections, local knowledge. | Editors. |
| Graph keeper | `TopicConnectorAgent` | Links semantically related topics (shared events, entities, subject matter) and submits the relationships through the backend API. See [CONNECTOR.md](CONNECTOR.md). | The backend validates; editors curate the tree. |

## Data flow

```
feeds ─► monitor ─► knowledge architect ─► news editor ─► [ per pitch: story graph ]
 (Kaleva, Yle,        (events + topic tree)    (pitches by         │
  BBC, officials)                              community)          ▼
                                                        investigate ⇄ (bounded rounds)
                                                              │
                                              explain ─► translate ─► standards ─► revise (bounded)
                                                                           │
                                                          ┌────────────────┘
                                                          ▼
                                                   HUMAN GATE (interrupt, checkpointed)
                                          reject │ request changes │ approve (all required roles)
                                                                           ▼
                                                                    publish (renditions per channel)
community messages ─► community desk ─► tip → pitch (a lead, never evidence) │ correction → open correction │ question → answer from published items or hand to a human
```

## Same news, many outlets: the event tree

* **Topics** are persistent subject areas (`config/topics.yaml`). **Events** are specific
  developments. **Articles** are single outlets' reports. **Claims** are factual assertions.
* Reports cluster by language-independent signals (numbers, dates, named entities, places,
  translated key terms) in a time window. Two independent kinds of signal must agree.
* Kaleva is `metadata_only` (paywalled), so it clusters from headline and summary alone.
* An outlet is counted once: all Yle feeds are one outlet, all Kaleva feeds are one outlet.
  Syndicated copies do not add independence.
* The architect only **proposes** tree changes. Code validates them. Attaching and creating
  events is auto-committed. New top-level topics and merges wait for a human.

## What the agents do alone, and what only a person does

| Agents, automatically | A person, always |
|---|---|
| Collect feeds on a schedule, group the same news, pick stories | Press Publish (editor-in-chief) |
| Check claims, attach official references, write the explainer | Approve high-risk stories (extra named reviewers, by policy) |
| Draft translations, run the standards check | Approve each non-source language (language reviewer) |
| Build the outputs for each channel once approved | Write Sámi text (human-only language) |
| Triage questions, tips and correction requests | Publish a correction, decide on local knowledge |

## Where the rules live

Business rules are plain code and data, never a prompt:

* claim status (verified / attributed / disputed / unknown) and independence: `agents/investigator.py`
* risk tiers and protected terms: `config/policy.yaml`, `guards.py`
* who may approve what, and when a story may publish: `hitl.py`
* disclosure text: `config/policy.yaml`

Models only propose structured text, scores and entities. Output is validated against schemas.

## Bounds and stopping conditions

`max_investigation_rounds`, `max_revision_rounds`, `max_pitches_per_cycle` (settings). Each loop
traces why it stopped. The graph is checkpointed in SQLite, so a review can happen days later.

## Model layer

`LLM.generate(task, system, prompt, schema, payload)`. Gemini uses prompts and the schema. The
offline `FakeLLM` uses `task` and `payload` with rule-based handlers, so demos and tests need no
network. Gemini inference is hosted by Google (network, API key, possible cost).

## Prototype limits (honest)

* Offline translation is a labelled stand-in, not real translation. Real quality needs Gemini and reviewers.
* Reviewers in the demo are simulated. Real deployments need staff authentication in front of the review API.
* Only the English story is drafted; other languages are translated from it.
* Audio is a script; text-to-speech and messaging delivery are not wired up.
